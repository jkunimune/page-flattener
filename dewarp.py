"""
This work by Justin Kunimune is marked with CC0 1.0 Universal.
To view a copy of this license, visit <https://creativecommons.org/publicdomain/zero/1.0>
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from PIL import Image
from numpy import shape, linspace, sqrt, meshgrid, stack, arange, transpose, concatenate, array, \
	ravel, size, newaxis, linalg, clip, ceil, where
from numpy.typing import NDArray
from scipy import optimize
from scipy.interpolate import RegularGridInterpolator
from torch import Tensor
from torch.autograd.functional import jacobian


NUM_OPTIMIZATION_ITERATIONS = 10
NUM_INVERSION_ITERATIONS = 10


def dewarp(image_warped: NDArray, point_sets_warped: List[PointSet]) -> Tuple[NDArray, List[PointSet]]:
	num_y, num_x, num_channels = shape(image_warped)

	# do the optimization to define the flattening spline
	print("Solving for the optimal transformation...")
	x_spline, y_spline = optimize_spline_nodes(num_x, num_y, point_sets_warped)

	# recover the straightened point sets
	point_sets_flattened = []
	for point_set in point_sets_warped:
		points_flattened = stack([
			apply_spline(point_set.points[:, 0], point_set.points[:, 1], x_spline).numpy(),
			apply_spline(point_set.points[:, 0], point_set.points[:, 1], y_spline).numpy(),
		], axis=1)
		point_sets_flattened.append(PointSet(point_set.angle, point_set.offset, points_flattened))

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

	print("Got it!")
	return image_flattened, point_sets_flattened


def optimize_spline_nodes(width: int, height: int, point_sets: List[PointSet]) -> Tuple[Spline, Spline]:
	# define the mesh grid that will be used to define the transformation
	cell_size = sqrt(width*height)/10
	x_node_warped = linspace(0, width, round(width/cell_size) + 1)
	y_node_warped = linspace(0, height, round(height/cell_size) + 1)

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
			if point_set.angle is not None:
				angle = torch.as_tensor(point_set.angle)
			else:  # if the user didn't specify an angle, find the least squares angle
				Δx = x - torch.mean(x)
				Δy = y - torch.mean(y)
				angle = torch.arctan2((2*Δx*Δy).sum(), (Δx**2 - Δy**2).sum())/2
			values = x*torch.sin(angle) + y*torch.cos(angle)
			offset = point_set.offset
			if point_set.offset is not None:
				offset = torch.as_tensor(offset)
			else:  # if the user didn't specify an offset, find the least squares offset
				offset = torch.mean(values)
			residual_vectors.append(values - offset)
		return torch.concatenate(residual_vectors)

	# autodifferentiate it
	def residuals_gradient(state):
		inputs = torch.tensor(state, requires_grad=True)
		return jacobian(residuals_function, inputs)

	# run the least squares algorithm
	optimization = optimize.least_squares(
		fun=lambda x: residuals_function(x).numpy(),
		jac=residuals_gradient,
		x0=initial_state,
		max_nfev=NUM_OPTIMIZATION_ITERATIONS,
	)
	print(optimization.message)

	# don't forget to convert from Tensor back to Numpy array before returning
	x_spline, y_spline = unpack_state(optimization.x)
	x_spline.z_node = x_spline.z_node.numpy()
	y_spline.z_node = y_spline.z_node.numpy()
	return x_spline, y_spline


def apply_inverse_splines(x_desired: NDArray, y_desired: NDArray, x_spline: Spline, y_spline: Spline) -> Tuple[NDArray, NDArray]:
	states = stack([x_desired, y_desired], axis=-1)  # for the initial gess, assume the spline is the identity transform
	targets = stack([x_desired, y_desired], axis=-1)

	for i in range(NUM_INVERSION_ITERATIONS):
		results = stack([
			apply_spline(states[..., 0], states[..., 1], x_spline).numpy(),
			apply_spline(states[..., 0], states[..., 1], y_spline).numpy(),
		], axis=-1)
		residuals = results - targets
		jacobians = stack([
			spline_gradient(states[..., 0], states[..., 1], x_spline).numpy(),
			spline_gradient(states[..., 0], states[..., 1], y_spline).numpy(),
		], axis=-2)
		steps = (-linalg.inv(jacobians)@residuals[..., newaxis])[..., 0]
		states += steps

	x_optimal = states[..., 0]
	y_optimal = states[..., 1]
	return x_optimal, y_optimal


def apply_spline(x_input: NDArray, y_input: NDArray, spline: Spline) -> Tensor:
	""" x_input and y_input must be evenly spaced! """
	assert shape(x_input) == shape(y_input)

	# find out in what cell each input point is
	i_node, di_input = digitize(y_input, spline.y_node)
	j_node, dj_input = digitize(x_input, spline.x_node)

	# apply the 4×4 convolution kernel
	result = torch.zeros(shape(x_input) + shape(spline.z_node)[2:])
	row_weits = {Δi: bicubic_function(di_input - Δi) for Δi in range(-2, 2)}
	col_weits = {Δj: bicubic_function(dj_input - Δj) for Δj in range(-2, 2)}
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
	x_gradients = torch.zeros(shape(x_input) + shape(spline.z_node)[2:])
	y_gradients = torch.zeros(shape(x_input) + shape(spline.z_node)[2:])
	row_weits = {Δi: bicubic_function(di_input - Δi) for Δi in range(-2, 2)}
	col_weits = {Δj: bicubic_function(dj_input - Δj) for Δj in range(-2, 2)}
	row_slopes = {Δi: bicubic_function_derivative(di_input - Δi) for Δi in range(-2, 2)}
	col_slopes = {Δj: bicubic_function_derivative(dj_input - Δj) for Δj in range(-2, 2)}
	for Δi in range(-2, 2):
		for Δj in range(-2, 2):
			x_gradients += row_weits[Δi]*col_slopes[Δj]*spline.z_node[i_node + Δi, j_node + Δj, ...]
			y_gradients += row_slopes[Δi]*col_weits[Δj]*spline.z_node[i_node + Δi, j_node + Δj, ...]
	return torch.stack([x_gradients, y_gradients], dim=-1)


def bicubic_function(Δi: NDArray, a=-0.5) -> NDArray:
	x = abs(Δi)
	return where(
		x <= 1,
		(a + 2)*x**3 - (a + 3)*x**2 + 1,
		where(
			x < 2,
			a*x**3 - 5*a*x**2 + 8*a*x - 4*a,
			0,
		),
	)


def bicubic_function_derivative(Δi, a=-0.5):
	x = abs(Δi)
	return where(
		x <= 1,
		3*(a + 2)*x**2 - 2*(a + 3)*x,
		where(
			x < 2,
			3*a*x**2 - 10*a*x + 8*a,
			0,
		),
	)


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
	def __init__(self, angle: Optional[float], offset: Optional[float], points: Union[NDArray, Tensor]):
		self.angle = angle
		self.offset = offset
		self.points = points
