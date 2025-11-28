"""
This work by Justin Kunimune is marked with CC0 1.0 Universal.
To view a copy of this license, visit <https://creativecommons.org/publicdomain/zero/1.0>
"""
from __future__ import annotations

from time import time
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image
from numpy import shape, linspace, sqrt, meshgrid, stack, arange, transpose, concatenate, array, \
	ravel, size, newaxis, linalg, clip, ceil, hypot, zeros_like
from numpy.linalg import LinAlgError
from numpy.typing import NDArray
from scipy import optimize
from scipy.interpolate import RegularGridInterpolator
from torch import Tensor
from torch.autograd.functional import jacobian


NUM_OPTIMIZATION_ITERATIONS = 100
NUM_INVERSION_ITERATIONS = 10
REGULARIZATION_FACTOR = 1


def dewarp(image_warped: NDArray, point_sets_warped: List[PointSet], resolution: float) -> Tuple[NDArray, List[PointSet]]:
	num_y, num_x, num_channels = shape(image_warped)

	start_time = time()

	# do the optimization to define the flattening spline
	print("Solving for the optimal transformation...")
	x_spline, y_spline = optimize_spline_nodes(num_x, num_y, point_sets_warped, resolution)

	# recover the straightened point sets
	point_sets_flattened = []
	for point_set in point_sets_warped:
		points_flattened = stack([
			apply_spline(point_set.points[:, 0], point_set.points[:, 1], x_spline).numpy(),
			apply_spline(point_set.points[:, 0], point_set.points[:, 1], y_spline).numpy(),
		], axis=1)
		point_sets_flattened.append(PointSet(point_set.target, points_flattened))

	# recover the original image
	print("Inverting the optimal transformation...")
	x_pixel_flat, y_pixel_flat = meshgrid(0.5 + arange(num_x), 0.5 + arange(num_y), indexing="xy")
	cell_size = (x_spline.x_node[1] - x_spline.x_node[0] + x_spline.y_node[1] - y_spline.y_node[0])/2
	macropixel_size = max(1, cell_size/10)
	full_shape = (num_x, num_y)  # these shapes are xy indexing because they're going into PIL; everything else is yx indexing
	reduced_shape = (round(num_x/macropixel_size), round(num_y/macropixel_size))
	x_macropixel_flat = array(Image.fromarray(x_pixel_flat).resize(reduced_shape))
	y_macropixel_flat = array(Image.fromarray(y_pixel_flat).resize(reduced_shape))
	x_macropixel_warp, y_macropixel_warp = apply_inverse_splines(
		x_macropixel_flat, y_macropixel_flat, x_spline, y_spline)
	x_pixel_warp = array(Image.fromarray(x_macropixel_warp).resize(full_shape))
	y_pixel_warp = array(Image.fromarray(y_macropixel_warp).resize(full_shape))
	print("Applying the inverse transformation to the image...")
	image_flattened = stack([
		RegularGridInterpolator(
			(arange(num_x), arange(num_y)), transpose(image_warped[:, :, k]),
			method="linear", bounds_error=False, fill_value=0,
		)((x_pixel_warp, y_pixel_warp))
		for k in range(num_channels)
	], axis=2).astype(image_warped.dtype, casting="unsafe")

	end_time = time()
	print(f"Got it in {end_time - start_time:.0f} s!")

	return image_flattened, point_sets_flattened


def optimize_spline_nodes(width: int, height: int, point_sets: List[PointSet], resolution: float) -> Tuple[Spline, Spline]:
	# define the mesh grid that will be used to define the transformation
	cell_size = sqrt(width*height)/resolution
	x_node_warped = linspace(0, width, max(1, round(width/cell_size)) + 1)
	y_node_warped = linspace(0, height, max(1, round(height/cell_size)) + 1)

	# compress the state into a vector
	x_node_initial, y_node_initial = meshgrid(x_node_warped, y_node_warped, indexing="xy")
	initial_state = ravel(stack([x_node_initial, y_node_initial], axis=0))

	def unpack_state(state) -> Tuple[Spline, Spline]:
		x_node, y_node = torch.reshape(
			torch.as_tensor(state), (2, size(y_node_warped), size(x_node_warped)))
		x_spline = Spline(x_node_warped, y_node_warped, x_node)
		y_spline = Spline(x_node_warped, y_node_warped, y_node)
		return x_spline, y_spline

	# define the residuals function
	def residuals_function(state):
		x_spline, y_spline = unpack_state(state)
		residual_vectors = []
		# each point set has its own set of residuals
		for point_set in point_sets:
			x = apply_spline(point_set.points[:, 0], point_set.points[:, 1], x_spline)
			y = apply_spline(point_set.points[:, 0], point_set.points[:, 1], y_spline)
			if type(point_set.target) is Line:
				residuals, directions = fit_line(x, y, point_set.target)
			elif type(point_set.target) is Arc:
				residuals, directions = fit_arc(x, y)  # use squared radii so that we can solve it algebraicly
			else:
				raise ValueError(point_set.target)
			jacobians = torch.stack([
				spline_gradient(point_set.points[:, 0], point_set.points[:, 1], x_spline),
				spline_gradient(point_set.points[:, 0], point_set.points[:, 1], y_spline),
			], dim=-2)
			inv_jacobians = torch.linalg.inv(jacobians)
			scale = torch.linalg.vector_norm((inv_jacobians@directions[..., newaxis])[..., 0], dim=-1)  # scale the residuals so that we're measuring in warped image units
			residual_vectors.append(scale*residuals)
		# the second derivatives at each point can also be treated as residuals for regularization purposes
		if regularization_weight != 0:
			for spline in [x_spline, y_spline]:
				# dx^2 + dy^2
				i, j = meshgrid(
					arange(1, size(spline.y_node) - 1),
					arange(1, size(spline.x_node) - 1),
					indexing="xy")
				curvature = (
					spline.z_node[i - 1, j] + spline.z_node[i, j - 1] +
					spline.z_node[i + 1, j] + spline.z_node[i, j + 1] -
					4*spline.z_node[i, j]
				)/cell_size**2
				residual_vectors.append(torch.ravel(regularization_weight*curvature))
				# dxdy
				i, j = meshgrid(
					arange(1, size(spline.y_node)),
					arange(1, size(spline.x_node)),
					indexing="xy")
				curvature = 2*(
					spline.z_node[i, j] - spline.z_node[i, j - 1] -
					spline.z_node[i - 1, j] + spline.z_node[i - 1, j - 1]
				)/cell_size**2
				residual_vectors.append(torch.ravel(regularization_weight*curvature))
		return torch.concatenate(residual_vectors)

	# autodifferentiate it
	def residuals_gradient(state):
		inputs = torch.tensor(state, requires_grad=True)
		return jacobian(residuals_function, inputs).numpy()

	# pick a suitable value for the regularization weight
	regularization_weight = 0
	error_scale = (residuals_function(initial_state).numpy()**2).sum()
	curvature_scale = (size(y_node_warped) - 1)*(size(x_node_warped) - 1)/hypot(width, height)
	regularization_weight = REGULARIZATION_FACTOR*sqrt(error_scale/curvature_scale)

	# run the least squares algorithm
	optimization = optimize.least_squares(
		fun=lambda x: residuals_function(x).numpy(),
		jac=residuals_gradient,
		x0=initial_state,
		max_nfev=NUM_OPTIMIZATION_ITERATIONS,
		verbose=2,
	)
	optimal_state = optimization.x

	# don't forget to convert from Tensor back to Numpy array before returning
	x_spline, y_spline = unpack_state(optimal_state)
	x_spline.z_node = x_spline.z_node.numpy()
	y_spline.z_node = y_spline.z_node.numpy()
	return x_spline, y_spline


def fit_line(x: Tensor, y: Tensor, parameters: Line) -> Tuple[Tensor, Tensor]:
	if parameters.angle is not None:
		angle = torch.as_tensor(parameters.angle, dtype=torch.float64)
	else:  # if the user didn't specify an angle, find the least squares angle
		Δx = x - torch.mean(x)
		Δy = y - torch.mean(y)
		angle = torch.arctan2(torch.mean(-2*Δx*Δy), torch.mean(Δx**2 - Δy**2))/2
	sin_angle = torch.sin(angle)
	cos_angle = torch.cos(angle)
	actual_offsets = x*sin_angle + y*cos_angle
	if parameters.offset is not None:
		offset = torch.as_tensor(parameters.offset, dtype=torch.float64)
	else:  # if the user didn't specify an offset, find the least squares offset
		offset = torch.mean(actual_offsets)
	return actual_offsets - offset, torch.stack([sin_angle, cos_angle]).expand(x.shape + (-1,))


def fit_arc(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
	sx = torch.mean(x)
	sy = torch.mean(y)
	sxx = torch.mean(x**2)
	sxy = torch.mean(x*y)
	syy = torch.mean(y**2)
	sxxx = torch.mean(x**3)
	sxxy = torch.mean(x**2*y)
	sxyy = torch.mean(x*y**2)
	syyy = torch.mean(y**3)
	a1 = 2*(sx**2 - sxx)
	a2 = b1 = 2*(sx*sy - sxy)
	b2 = 2*(sy**2 - syy)
	c1 = (sxx*sx - sxxx + sx*syy - sxyy)
	c2 = (sxx*sy - sxxy + sy*syy - syyy)
	det = a1*b2 - a2*b1
	x_center = (c1*b2 - c2*b1)/det
	y_center = (a1*c2 - a2*c1)/det
	target_radius2 = sxx - 2*sx*x_center + x_center**2 + syy - 2*sy*y_center + y_center**2
	r = torch.stack([x - x_center, y - y_center], dim=-1)
	r2 = r[..., 0]**2 + r[..., 1]**2
	error_magnitude = (r2 - target_radius2)/2/torch.sqrt(target_radius2)
	return error_magnitude, r/torch.sqrt(r2[..., newaxis])


def apply_inverse_splines(x_desired: NDArray, y_desired: NDArray, x_spline: Spline, y_spline: Spline) -> Tuple[NDArray, NDArray]:
	states = stack([x_desired, y_desired], axis=-1)  # for the initial gess, assume the spline is the identity transform
	targets = stack([x_desired, y_desired], axis=-1)

	for i in range(NUM_INVERSION_ITERATIONS):
		# compute the error in each inverse point
		results = stack([
			apply_spline(states[..., 0], states[..., 1], x_spline).numpy(),
			apply_spline(states[..., 0], states[..., 1], y_spline).numpy(),
		], axis=-1)
		residuals = results - targets
		# compute the jacobian of that error
		jacobians = stack([
			spline_gradient(states[..., 0], states[..., 1], x_spline).numpy(),
			spline_gradient(states[..., 0], states[..., 1], y_spline).numpy(),
		], axis=-2)
		# take a Newton-Raphson step
		try:
			steps = (-linalg.inv(jacobians)@residuals[..., newaxis])[..., 0]
		except LinAlgError:
			print("The inversion failed due to a point with no gradient!  That shouldn't happen…")
			return states[..., 0], states[..., 1]
		states += steps

	return states[..., 0], states[..., 1]


def apply_spline(x_input: NDArray, y_input: NDArray, spline: Spline) -> Tensor:
	""" x_input and y_input must be evenly spaced! """
	assert shape(x_input) == shape(y_input)

	# find out in what cell each input point is
	i_node, di_input = digitize(y_input, spline.y_node)
	j_node, dj_input = digitize(x_input, spline.x_node)

	# apply the 4×4 convolution kernel
	result = torch.zeros(shape(x_input) + shape(spline.z_node)[2:], dtype=torch.float64)
	row_weits = {Δi: bicubic_function(di_input - Δi, -Δi) for Δi in range(-2, 2)}
	col_weits = {Δj: bicubic_function(dj_input - Δj, -Δj) for Δj in range(-2, 2)}
	for Δi in range(-2, 2):
		for Δj in range(-2, 2):
			weight = torch.as_tensor(row_weits[Δi]*col_weits[Δj])
			result += weight*spline.z_node[i_node + Δi, j_node + Δj, ...]
	return result


def spline_gradient(x_input: NDArray, y_input: NDArray, spline: Spline) -> Tensor:
	# find out in what cell each input point is
	i_node, di_input = digitize(y_input, spline.y_node)
	j_node, dj_input = digitize(x_input, spline.x_node)

	# apply the 4×4 differentiated convolution kernel
	x_gradients = torch.zeros(shape(x_input) + shape(spline.z_node)[2:], dtype=torch.float64)
	y_gradients = torch.zeros(shape(x_input) + shape(spline.z_node)[2:], dtype=torch.float64)
	row_weits = {Δi: bicubic_function(di_input - Δi, -Δi) for Δi in range(-2, 2)}
	col_weits = {Δj: bicubic_function(dj_input - Δj, -Δj) for Δj in range(-2, 2)}
	row_slopes = {Δi: bicubic_function_derivative(di_input - Δi, -Δi) for Δi in range(-2, 2)}
	col_slopes = {Δj: bicubic_function_derivative(dj_input - Δj, -Δj) for Δj in range(-2, 2)}
	for Δi in range(-2, 2):
		for Δj in range(-2, 2):
			x_weit = torch.as_tensor(row_weits[Δi]*col_slopes[Δj])
			y_weit = torch.as_tensor(row_slopes[Δi]*col_weits[Δj])
			x_gradients += x_weit*spline.z_node[i_node + Δi, j_node + Δj, ...]
			y_gradients += y_weit*spline.z_node[i_node + Δi, j_node + Δj, ...]
	# don't forget to scale to correct for the change of coordinates earlier in this function
	x_gradients /= (spline.x_node[1] - spline.x_node[0])
	y_gradients /= (spline.y_node[1] - spline.y_node[0])
	return torch.stack([x_gradients, y_gradients], dim=-1)


def bicubic_function(x: NDArray, section: int) -> NDArray:
	if section == -1:
		return 0.5*x**3 + 2.5*x**2 + 4*x + 2
	elif section == 0:
		return -1.5*x**3 - 2.5*x**2 + 1
	elif section == 1:
		return 1.5*x**3 - 2.5*x**2 + 1
	elif section == 2:
		return -0.5*x**3 + 2.5*x**2 - 4*x + 2
	else:
		return zeros_like(x)


def bicubic_function_derivative(x: NDArray, section: int) -> NDArray:
	if section == -1:
		return 1.5*x**2 + 5*x + 4
	elif section == 0:
		return -4.5*x**2 - 5*x
	elif section == 1:
		return 4.5*x**2 - 5*x
	elif section == 2:
		return -1.5*x**2 + 5*x - 4
	else:
		return zeros_like(x)


def digitize(x, bins) -> Tuple[NDArray, NDArray]:
	"""
	fit the given value into a bin that can be used to spline interpolate it
	:return: the index of one of the bin nodes and the distance between this point's true index and that node's index
	"""
	i = (x - bins[0])/(bins[1] - bins[0])
	i_bin = clip(ceil(i).astype(int), 2, size(bins) - 2)
	return i_bin, i - i_bin


class Spline:
	def __init__(self, x_node: NDArray, y_node: NDArray, z_node: Union[NDArray, Tensor]):
		# add random garbage to the outer edges to make the edges behave better
		x_node = concatenate([
			array([2*x_node[0] - x_node[1]]),
			x_node,
			array([2*x_node[-1] - x_node[-2]]),
		])
		y_node = concatenate([
			array([2*y_node[0] - y_node[1]]),
			y_node,
			array([2*y_node[-1] - y_node[-2]]),
		])
		z_node = torch.as_tensor(z_node)
		z_node = torch.concatenate([
			(2*z_node[0, :] - z_node[1, :])[newaxis, :],
			z_node,
			(2*z_node[-1, :] - z_node[-2, :])[newaxis, :],
		], dim=0)
		z_node = torch.concatenate([
			(2*z_node[:, 0] - z_node[:, 1])[:, newaxis],
			z_node,
			(2*z_node[:, -1] - z_node[:, -2])[:, newaxis],
		], dim=1)

		self.x_node = x_node
		self.y_node = y_node
		self.z_node = z_node


class PointSet:
	def __init__(self, target: Shape, points: Union[NDArray, Tensor]):
		self.target = target
		self.points = points

class Line:
	def __init__(self, angle: Optional[float], offset: Optional[float]):
		self.angle = angle
		self.offset = offset

class Arc:
	def __init__(self):
		pass  # in the future I may allow the user to specify radius and/or center coordinates, but not now.

Shape = Union[Line, Arc]
