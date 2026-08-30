"""
This work by Justin Kunimune is marked with CC0 1.0 Universal.
To view a copy of this license, visit <https://creativecommons.org/publicdomain/zero/1.0>
"""
from __future__ import annotations

from math import nan, pi
from time import time
from typing import List, Optional, Tuple, Union

from autograd import jacobian
from autograd.numpy import shape, sqrt, meshgrid, stack, transpose, concatenate, ravel, newaxis, linalg, clip, ceil, \
	hypot, where, reshape, arctan2, sin, cos, arccos, maximum, mean, sign
from numpy import zeros, linspace, arange, array, size, zeros_like, isnan, any, seterr
from numpy.linalg import LinAlgError
from numpy.typing import NDArray
from scipy import optimize
from scipy.interpolate import RegularGridInterpolator

seterr(divide="ignore")


NUM_OPTIMIZATION_ITERATIONS = 100
NUM_INVERSION_ITERATIONS = 10
REGULARIZATION_FACTOR = 1


def dewarp(
		image_warped: NDArray, point_sets_warped: List[PointSet], resolution: float,
) -> Tuple[NDArray, List[PointSet], Spline, Spline]:
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
	x_macropixel_flat = resample(x_pixel_flat, reduced_shape)
	y_macropixel_flat = resample(y_pixel_flat, reduced_shape)
	x_macropixel_warp, y_macropixel_warp = apply_inverse_splines(
		x_macropixel_flat, y_macropixel_flat, x_spline, y_spline)

	x_pixel_warp = resample(x_macropixel_warp, full_shape)
	y_pixel_warp = resample(y_macropixel_warp, full_shape)
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

	return image_flattened, point_sets_flattened, x_spline, y_spline


def optimize_spline_nodes(width: int, height: int, point_sets: List[PointSet], resolution: float) -> Tuple[Spline, Spline]:
	# define the mesh grid that will be used to define the transformation
	cell_size = sqrt(width*height)/resolution
	x_node_warped = linspace(0, width, max(1, round(width/cell_size)) + 1)
	y_node_warped = linspace(0, height, max(1, round(height/cell_size)) + 1)

	# compress the state into a vector
	x_node_initial, y_node_initial = meshgrid(x_node_warped, y_node_warped, indexing="xy")
	initial_state = ravel(stack([x_node_initial, y_node_initial], axis=0))

	def unpack_state(state) -> Tuple[Spline, Spline]:
		x_node, y_node = reshape(
			state, (2, size(y_node_warped), size(x_node_warped)))
		x_spline = Spline(x_node_warped, y_node_warped, x_node)
		y_spline = Spline(x_node_warped, y_node_warped, y_node)
		return x_spline, y_spline

	# define the residuals function
	def residuals_function(state):
		x_spline, y_spline = unpack_state(state)
		residual_sets = []
		# each point set has its own set of residuals
		for point_set in point_sets:
			x = apply_spline(point_set.points[:, 0], point_set.points[:, 1], x_spline)
			y = apply_spline(point_set.points[:, 0], point_set.points[:, 1], y_spline)
			if type(point_set.target) is Line:
				residual_signs, residual_vectors = fit_line(x, y, point_set.target)
			elif type(point_set.target) is Arc:
				residual_signs, residual_vectors = fit_arc(x, y)  # use squared radii so that we can solve it algebraicly
			else:
				raise ValueError(point_set.target)
			jacobians = stack([
				spline_gradient(point_set.points[:, 0], point_set.points[:, 1], x_spline),
				spline_gradient(point_set.points[:, 0], point_set.points[:, 1], y_spline),
			], axis=-2)
			abs_residual_vectors = lstsq(jacobians, residual_vectors)  # scale the residuals so that we're measuring in warped image units
			abs_residuals = residual_signs[:, newaxis]*vector_norm(abs_residual_vectors)
			residual_sets.append(ravel(abs_residuals))
		# the second derivatives at each point can also be treated as residuals for regularization purposes
		if regularization_weight != 0:
			x = linspace(x_spline.x_node[0], x_spline.x_node[1], (len(x_spline.x_node) - 1)*4 + 1)[1:-1:2]
			y = linspace(x_spline.y_node[0], x_spline.y_node[1], (len(x_spline.y_node) - 1)*4 + 1)[1:-1:2]
			x, y = meshgrid(x, y, indexing="ij")
			x_hessians = spline_hessian(x, y, x_spline)
			y_hessians = spline_hessian(x, y, y_spline)
			illegal_fraction = 1 - paperlike_fraction(x_hessians, y_hessians)
			residual_sets.append(ravel(regularization_weight*illegal_fraction[:, None, None]*x_hessians))
			residual_sets.append(ravel(regularization_weight*illegal_fraction[:, None, None]*y_hessians))
		return concatenate(residual_sets)

	# autodifferentiate it
	def residuals_gradient(state):
		return jacobian(residuals_function)(state)

	# pick a suitable value for the regularization weight
	regularization_weight = 0
	error_scale = (residuals_function(initial_state)**2).sum()
	curvature_scale = (size(y_node_warped) - 1)*(size(x_node_warped) - 1)/hypot(width, height)
	regularization_weight = REGULARIZATION_FACTOR*sqrt(error_scale/curvature_scale)

	# run the least squares algorithm
	optimization = optimize.least_squares(
		fun=residuals_function,
		jac=residuals_gradient,
		x0=initial_state,
		max_nfev=NUM_OPTIMIZATION_ITERATIONS,
		ftol=1e-5,
		verbose=2,
	)
	optimal_state = optimization.x

	# don't forget to convert from NDArray back to Numpy array before returning
	x_spline, y_spline = unpack_state(optimal_state)
	x_spline.z_node = x_spline.z_node.numpy()
	y_spline.z_node = y_spline.z_node.numpy()
	return x_spline, y_spline


def paperlike_fraction(x_hessians, y_hessians):
	"""
	given a bunch of 2×2 hessian matrices, calculate the fraction of each's Frobenius norm that can be
	accounted for as curvature along a single direction of displacement in a single direction.
	"""
	d2x_dx2 = x_hessians[:, 0, 0]
	d2x_dxy = x_hessians[:, 0, 1]  # assuming the hessians are symmetric, we can discard element 1,0
	d2x_dy2 = x_hessians[:, 1, 1]
	d2y_dx2 = y_hessians[:, 0, 0]
	d2y_dxy = y_hessians[:, 0, 1]  # assuming the hessians are symmetric, we can discard element 1,0
	d2y_dy2 = y_hessians[:, 1, 1]

	total_curvature = d2x_dx2**2 + 2*d2x_dxy**2 + d2x_dy2**2 + d2y_dx2**2 + 2*d2y_dxy**2 + d2y_dy2**2

	M = stack([
		stack([1/2*(d2x_dx2 - d2x_dy2), 1/2*(d2y_dx2 - d2y_dy2)]),
		stack([d2x_dxy, d2y_dxy]),
	])
	B = stack([1/2*(d2x_dx2 + d2x_dy2), 1/2*(d2y_dx2 + d2y_dy2)])

	err_hessian = 2*stack([
		stack([M[0, 0]**2 + M[0, 1]**2, M[0, 0]*M[1, 0] + M[0, 1]*M[1, 1]]),
		stack([M[0, 0]*M[1, 0] + M[0, 1]*M[1, 1], M[1, 0]**2 + M[1, 1]**2]),
	])
	err_gradient = 2*stack([
		M[0, 0]*B[0] + M[0, 1]*B[1],
		M[1, 0]*B[0] + M[1, 1]*B[1]])
	t_solve_principal = solve_quartic_equation(
		err_hessian[1, 0] - err_gradient[1],
		2*(err_hessian[0, 0] - err_hessian[1, 1] - err_gradient[0]),
		-6*err_hessian[0, 1],
		2*(err_hessian[1, 1] - err_hessian[0, 0] - err_gradient[0]),
		err_hessian[0, 1] + err_gradient[1],
	)
	t_solve_backup = solve_quartic_equation(
		err_hessian[1, 0] + err_gradient[1],
		2*(err_hessian[0, 0] - err_hessian[1, 1] + err_gradient[0]),
		-6*err_hessian[0, 1],
		2*(err_hessian[1, 1] - err_hessian[0, 0] + err_gradient[0]),
		err_hessian[0, 1] - err_gradient[1]
	)
	captured_curvature = []
	for t, polarity in [(t, +1) for t in t_solve_principal] + [(t, -1) for t in t_solve_backup] + [(0, +1)]:
		if polarity > 0:
			α = arctan2(2*t, 1 - t**2)/2
		else:
			α = arctan2(-2*t, t**2 - 1)/2
		va = stack([
			cos(2*α),
			sin(2*α)])
		vb = stack([
			M[0, 0]*va[0] + M[1, 0]*va[1] + B[0],
			M[0, 1]*va[0] + M[1, 1]*va[1] + B[1]])
		captured_curvature.append(vb[0]**2 + vb[1]**2)
	max_captured_curvature = nanmax(captured_curvature)

	return where(total_curvature != 0, max_captured_curvature/total_curvature, 1/4)  # in the event of a zero hessian, just guess 1/4 (it won't be autodifferentiable so hopefully that's okay)


def solve_quartic_equation(a4_raw, a3_raw, a2_raw, a1_raw, a0_raw) -> list[NDArray]:
	"""
	solve an equation of the form a4 x⁴ + a3 x³ + a2 x² + a1 x + a0 = 0, in a differentiable fashion.
	if there are fewer than four roots, any imaginary or infinite ones will return as nan.
	"""
	a3 = a3_raw/a4_raw
	a2 = a2_raw/a4_raw
	a1 = a1_raw/a4_raw
	a0 = a0_raw/a4_raw

	constant = a3/4
	b2 = a2 - 6*constant**2
	b1 = a1 - 2*a2*constant + 8*constant**3
	b0 = a0 - a1*constant + a2*constant**2 - 3*constant**4
	Σ = where(b1 > 0, 1., -1.)
	(x1, _), (x2, y2), (x3, y3) = solve_cubic_equation(1., b2/2, (b2**2 - 4*b0)/16, -b1**2/64)
	pivot1 = sqrt(maximum(0., x1))
	pivot2 = x2 + x3
	root2 = 2*Σ*sqrt(x2*x3 + y2**2)

	(x1_backup, y1_backup), (x2_backup, y2_backup), (x3_backup, y3_backup) = solve_cubic_equation(a3_raw, a2_raw, a1_raw, a0_raw)

	return [
		where(
			a4_raw != 0,
			pivot1 + sqrt(pivot2 - root2) - constant,
			where(y1_backup == 0, x1_backup, nan),
		),
		where(
			a4_raw != 0,
			pivot1 - sqrt(pivot2 - root2) - constant,
			where(y2_backup == 0, x2_backup, nan),
		),
		where(
			a4_raw != 0,
			-pivot1 + sqrt(pivot2 + root2) - constant,
			where(y3_backup == 0, x3_backup, nan),
		),
		where(
			a4_raw != 0,
			-pivot1 - sqrt(pivot2 + root2) - constant,
			nan,
		),
	]


def solve_cubic_equation(
		a3_raw: NDArray, a2_raw: NDArray, a1_raw: NDArray, a0_raw: NDArray,
) -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
	""" solve an equation of the form a3 x³ + a2 x² + a1 x + a0 = 0, in a differentiable fashion """
	a2 = a2_raw/a3_raw
	a1 = a1_raw/a3_raw
	a0 = a0_raw/a3_raw

	q = a1/3 - a2**2/9
	r = (a1*a2 - 3*a0)/6 - a2**3/27
	sign = where(r > 0, 1, -1)
	A = sign*(abs(r) + sqrt(r**2 + q**3))**(1/3)
	t1 = A - q/A
	θ = where(q == 0, 0, arccos(r/(-q)**(3/2)))
	φ1 = θ/3
	φ2 = φ1 - 2*pi/3
	φ3 = φ1 + 2*pi/3
	x1 = where(
		r**2 + q**3 > 0,
		t1 - a2/3,
		2*sqrt(-q)*cos(φ1) - a2/3,
	)
	x2 = where(
		r**2 + q**3 > 0,
		-t1/2 - a2/3,
		2*sqrt(-q)*cos(φ2) - a2/3,
	)
	y2 = where(
		r**2 + q**3 > 0,
		sqrt(3)/2*(A + q/A),
		0,
	)
	x3 = where(
		r**2 + q**3 > 0,
		x2,
		2*sqrt(-q)*cos(φ3) - a2/3,
	)
	y3 = where(
		r**2 + q**3 > 0,
		-y2,
		0,
	)

	(x1_backup, y1_backup), (x2_backup, y2_backup) = solve_quadratic_equation(a2_raw, a1_raw, a0_raw)

	return (
		(where(a3_raw != 0, x1, x1_backup), where(a3_raw != 0, 0, y1_backup)),
		(where(a3_raw != 0, x2, x2_backup), where(a3_raw != 0, y2, y2_backup)),
		(where(a3_raw != 0, x3, nan), where(a3_raw != 0, y3, nan)),
	)


def solve_quadratic_equation(
		a2: NDArray, a1: NDArray, a0: NDArray,
) -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
	""" solve an equation of the form a2 x² + a1 x + a0 = 0, in a differentiable fashion """
	constant = -a1/(2*a2)
	discriminant = (1/2*a1/a2)**2 - a0/a2

	x1 = where(
		discriminant >= 0,
		constant + sqrt(discriminant),
		constant,
	)
	y1 = where(
		discriminant >= 0,
		0,
		sqrt(-discriminant),
	)
	x2 = where(
		discriminant >= 0,
		constant - sqrt(discriminant),
		constant,
	)
	y2 = where(
		discriminant >= 0,
		0,
		-sqrt(-discriminant),
	)

	x1_backup = where(a1 != 0, -a0/a1, nan)
	y1_backup = where(a1 != 0, 0, nan)

	return (
		(where(a2 != 0, x1, x1_backup), where(a2 != 0, y1, y1_backup)),
		(where(a2 != 0, x2, nan), where(a2 != 0, y2, nan)),
	)


def lstsq(matrices: NDArray, vectors: NDArray) -> NDArray:
	assert matrices.shape[-2] == 2
	assert matrices.shape[-1] == 2
	assert vectors.shape[-1] == 2
	assert matrices.shape[:-2] == vectors.shape[:-1]
	det = matrices[:, 0, 0]*matrices[:, 1, 1] - matrices[:, 0, 1]*matrices[:, 1, 0]
	a = matrices[:, 1, 1]/det
	b = -matrices[:, 0, 1]/det
	c = -matrices[:, 1, 0]/det
	d = matrices[:, 0, 0]/det
	return stack([
		a*vectors[:, 0] + b*vectors[:, 1],
		c*vectors[:, 0] + d*vectors[:, 1],
	], axis=1)


def vector_norm(vectors: NDArray) -> NDArray:
	assert vectors.shape[-1] == 2
	return vectors/hypot(vectors[:, 0], vectors[:, 1])[:, newaxis]


def fit_line(x: NDArray, y: NDArray, parameters: Line) -> Tuple[NDArray, NDArray]:
	if parameters.angle is not None:
		angle = parameters.angle
	else:  # if the user didn't specify an angle, find the least squares angle
		Δx = x - mean(x)
		Δy = y - mean(y)
		angle = arctan2(mean(-2*Δx*Δy), mean(Δx**2 - Δy**2))/2
	sin_angle = sin(angle)
	cos_angle = cos(angle)
	actual_offsets = x*sin_angle + y*cos_angle
	if parameters.offset is not None:
		offset = parameters.offset
	else:  # if the user didn't specify an offset, find the least squares offset
		offset = mean(actual_offsets)
	error = actual_offsets - offset
	error_sign = sign(error)
	error_vector = error[..., newaxis]*stack([sin_angle, cos_angle])
	return error_sign, error_vector


def fit_arc(x: NDArray, y: NDArray) -> Tuple[NDArray, NDArray]:
	sx = mean(x)
	sy = mean(y)
	sxx = mean(x**2)
	sxy = mean(x*y)
	syy = mean(y**2)
	sxxx = mean(x**3)
	sxxy = mean(x**2*y)
	sxyy = mean(x*y**2)
	syyy = mean(y**3)
	a1 = 2*(sx**2 - sxx)
	a2 = b1 = 2*(sx*sy - sxy)
	b2 = 2*(sy**2 - syy)
	c1 = (sxx*sx - sxxx + sx*syy - sxyy)
	c2 = (sxx*sy - sxxy + sy*syy - syyy)
	det = a1*b2 - a2*b1
	x_center = (c1*b2 - c2*b1)/det
	y_center = (a1*c2 - a2*c1)/det
	target_radius2 = sxx - 2*sx*x_center + x_center**2 + syy - 2*sy*y_center + y_center**2
	r = stack([x - x_center, y - y_center], axis=-1)
	r2 = r[..., 0]**2 + r[..., 1]**2
	error_sign = sign(r2 - target_radius2)
	error_vector = r*((r2 - target_radius2)/2/r2)[..., newaxis]
	return error_sign, error_vector


def resample(image: NDArray, new_size: tuple[int, int]) -> NDArray:
	""" interpolate an image so that the edges are the same but the interior has a different sample density """
	x_old = arange(0, shape(image)[1])
	y_old = arange(0, shape(image)[0])
	x_new = linspace(0, shape(image)[1] - 1, new_size[0])
	y_new = linspace(0, shape(image)[0] - 1, new_size[1])
	return RegularGridInterpolator((x_old, y_old), image.T)(transpose(meshgrid(x_new, y_new, indexing="xy"), (1, 2, 0)))


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


def apply_spline(x_input: NDArray, y_input: NDArray, spline: Spline) -> NDArray:
	""" x_input and y_input must be evenly spaced! """
	assert shape(x_input) == shape(y_input)

	# find out in what cell each input point is
	i_node, di_input = digitize(y_input, spline.y_node)
	j_node, dj_input = digitize(x_input, spline.x_node)

	# apply the 4×4 convolution kernel
	result = zeros(shape(x_input) + shape(spline.z_node)[2:], dtype=float)
	row_weits = {Δi: bicubic_function(di_input - Δi, -Δi) for Δi in range(-2, 2)}
	col_weits = {Δj: bicubic_function(dj_input - Δj, -Δj) for Δj in range(-2, 2)}
	for Δi in range(-2, 2):
		for Δj in range(-2, 2):
			weight = row_weits[Δi]*col_weits[Δj]
			result = result + weight*spline.z_node[i_node + Δi, j_node + Δj, ...]
	return result


def spline_gradient(x_input: NDArray, y_input: NDArray, spline: Spline) -> NDArray:
	# find out in what cell each input point is
	i_node, di_input = digitize(y_input, spline.y_node)
	j_node, dj_input = digitize(x_input, spline.x_node)

	# apply the 4×4 differentiated convolution kernel
	x_gradients = zeros(shape(x_input) + shape(spline.z_node)[2:], dtype=float)
	y_gradients = zeros(shape(x_input) + shape(spline.z_node)[2:], dtype=float)
	row_weits = {Δi: bicubic_function(di_input - Δi, -Δi) for Δi in range(-2, 2)}
	col_weits = {Δj: bicubic_function(dj_input - Δj, -Δj) for Δj in range(-2, 2)}
	row_slopes = {Δi: bicubic_function_derivative(di_input - Δi, -Δi) for Δi in range(-2, 2)}
	col_slopes = {Δj: bicubic_function_derivative(dj_input - Δj, -Δj) for Δj in range(-2, 2)}
	for Δi in range(-2, 2):
		for Δj in range(-2, 2):
			x_weit = row_weits[Δi]*col_slopes[Δj]
			y_weit = row_slopes[Δi]*col_weits[Δj]
			x_gradients = x_gradients + x_weit*spline.z_node[i_node + Δi, j_node + Δj, ...]
			y_gradients = y_gradients + y_weit*spline.z_node[i_node + Δi, j_node + Δj, ...]
	# don't forget to scale to correct for the change of coordinates earlier in this function
	x_gradients /= (spline.x_node[1] - spline.x_node[0])
	y_gradients /= (spline.y_node[1] - spline.y_node[0])
	return stack([x_gradients, y_gradients], axis=-1)


def spline_hessian(x_input: NDArray, y_input: NDArray, spline: Spline) -> NDArray:
	# find out in what cell each input point is
	i_node, di_input = digitize(y_input, spline.y_node)
	j_node, dj_input = digitize(x_input, spline.x_node)

	# apply the 4×4 differentiated convolution kernel
	xx_term = zeros(shape(x_input) + shape(spline.z_node)[2:], dtype=float)
	xy_term = zeros(shape(x_input) + shape(spline.z_node)[2:], dtype=float)
	yy_term = zeros(shape(x_input) + shape(spline.z_node)[2:], dtype=float)
	row_weits = {Δi: bicubic_function(di_input - Δi, -Δi) for Δi in range(-2, 2)}
	col_weits = {Δj: bicubic_function(dj_input - Δj, -Δj) for Δj in range(-2, 2)}
	row_slopes = {Δi: bicubic_function_derivative(di_input - Δi, -Δi) for Δi in range(-2, 2)}
	col_slopes = {Δj: bicubic_function_derivative(dj_input - Δj, -Δj) for Δj in range(-2, 2)}
	row_curves = {Δi: bicubic_function_twoth_derivative(di_input - Δi, -Δi) for Δi in range(-2, 2)}
	col_curves = {Δj: bicubic_function_twoth_derivative(dj_input - Δj, -Δj) for Δj in range(-2, 2)}
	for Δi in range(-2, 2):
		for Δj in range(-2, 2):
			xx_weit = row_weits[Δi]*col_curves[Δj]
			xx_term = xx_term + xx_weit*spline.z_node[i_node + Δi, j_node + Δj, ...]
			xy_weit = row_slopes[Δi]*col_slopes[Δj]
			xy_term = xy_term + xy_weit*spline.z_node[i_node + Δi, j_node + Δj, ...]
			yy_weit = row_curves[Δi]*col_weits[Δj]
			yy_term = yy_term + yy_weit*spline.z_node[i_node + Δi, j_node + Δj, ...]
	# don't forget to scale to correct for the change of coordinates earlier in this function
	xx_term /= (spline.x_node[1] - spline.x_node[0])**2
	xy_term /= (spline.x_node[1] - spline.x_node[0])*(spline.y_node[1] - spline.y_node[0])
	yy_term /= (spline.y_node[1] - spline.y_node[0])**2
	return stack([xx_term, xy_term, xy_term, yy_term], axis=-1).reshape((-1, 2, 2))


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


def bicubic_function_twoth_derivative(x: NDArray, section: int) -> NDArray:
	if section == -1:
		return 3*x + 5
	elif section == 0:
		return -9*x - 5
	elif section == 1:
		return 9*x - 5
	elif section == 2:
		return -3*x + 5
	else:
		return zeros_like(x)


def midpoints(x: NDArray) -> NDArray:
	""" convert an array of interval edges to an array of interval centers """
	return (x[0:-1] + x[1:])/2


def digitize(x, bins) -> Tuple[NDArray, NDArray]:
	"""
	fit the given value into a bin that can be used to spline interpolate it
	:return: the index of one of the bin nodes and the distance between this point's true index and that node's index
	"""
	i = (x - bins[0])/(bins[1] - bins[0])
	i_bin = clip(ceil(i).astype(int), 2, size(bins) - 2)
	return i_bin, i - i_bin


def nanmax(arrays: list[NDArray]) -> NDArray:
	""" it's elementwise nan-avoiding max for a bunch of arrays """
	if len(arrays) == 0:
		raise ValueError("cannot take max of an empty list")
	result = arrays[0]
	for i in range(1, len(arrays)):
		result = where((result > arrays[i]) | isnan(arrays[i]), result, arrays[i])
	if any(isnan(result)):
		raise ValueError("some channels were all nan")
	return result


class Spline:
	def __init__(self, x_node: NDArray, y_node: NDArray, z_node: Union[NDArray, NDArray]):
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
		z_node = concatenate([
			(2*z_node[0, :] - z_node[1, :])[newaxis, :],
			z_node,
			(2*z_node[-1, :] - z_node[-2, :])[newaxis, :],
		], axis=0)
		z_node = concatenate([
			(2*z_node[:, 0] - z_node[:, 1])[:, newaxis],
			z_node,
			(2*z_node[:, -1] - z_node[:, -2])[:, newaxis],
		], axis=1)

		self.x_node = x_node
		self.y_node = y_node
		self.z_node = z_node


class PointSet:
	def __init__(self, target: Shape, points: Union[NDArray, NDArray]):
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
