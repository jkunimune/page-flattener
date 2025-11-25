"""
This work by Justin Kunimune is marked with CC0 1.0 Universal.
To view a copy of this license, visit <https://creativecommons.org/publicdomain/zero/1.0>
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from numpy import shape, linspace, sqrt, meshgrid, random, where, zeros, stack, arange, transpose, concatenate, array, \
	expand_dims, minimum, maximum, ceil
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator


def dewarp(image_warped: NDArray, point_sets_warped: List[PointSet]) -> Tuple[NDArray, List[PointSet]]:
	# define the mesh grid that will be used to define the transformation
	num_y, num_x, num_channels = shape(image_warped)
	cell_size = sqrt(num_x*num_y)/10
	x_node_warped = linspace(0, num_x, round(num_x/cell_size) + 1)
	y_node_warped = linspace(0, num_y, round(num_y/cell_size) + 1)

	# do the optimization to define the flattening spline
	print("solving for the optimal transformation...")
	x_node_flattened, y_node_flattened = meshgrid(x_node_warped, y_node_warped, indexing="xy")
	x_node_flattened += random.normal(0, 100, shape(x_node_flattened))
	y_node_flattened += random.normal(0, 100, shape(y_node_flattened))
	x_spline = Spline(x_node_warped, y_node_warped, x_node_flattened)
	y_spline = Spline(x_node_warped, y_node_warped, y_node_flattened)

	# recover the straightened point sets
	point_sets_flattened = []
	for point_set in point_sets_warped:
		points_flattened = stack([
			spline_interpolate(point_set.points[:, 0], point_set.points[:, 1], x_spline),
			spline_interpolate(point_set.points[:, 0], point_set.points[:, 1], y_spline)
		], axis=1)
		point_sets_flattened.append(PointSet(point_set.angle, point_set.offset, points_flattened))

	# recover the original image
	print("inverting the optimal transformation...")
	x_pixel_flat, y_pixel_flat = meshgrid(0.5 + arange(num_x), 0.5 + arange(num_y), indexing="xy")
	x_pixel_warp, y_pixel_warp = inverse_spline_interpolate(
		x_pixel_flat, y_pixel_flat, x_spline, y_spline)
	print("applying the inverse transformation to the image...")
	image_flattened = stack([
		RegularGridInterpolator(
			(arange(num_x), arange(num_y)), transpose(image_warped[:, :, k]),
			method="linear", bounds_error=False, fill_value=0,
		)((x_pixel_warp, y_pixel_warp))
		for k in range(num_channels)
	], axis=2).astype(image_warped.dtype, casting="unsafe")

	print("done!")
	return image_flattened, point_sets_flattened


def inverse_spline_interpolate(x_desired: NDArray, y_desired: NDArray, x_spline: Spline, y_spline: Spline) -> Tuple[NDArray, NDArray]:
	return x_desired, y_desired


def spline_interpolate(x_input: NDArray, y_input: NDArray, spline: Spline) -> NDArray:
	""" x_input and y_input must be evenly spaced! """
	assert shape(x_input) == shape(y_input)

	# add random garbage to the outer edges to make the edges behave better
	x_node = concatenate([
		array([2*spline.x_node[0] - spline.x_node[1]]),
		spline.x_node,
		array([2*spline.x_node[-1] - spline.x_node[-2]]),
	])
	y_node = concatenate([
		array([2*spline.y_node[0] - spline.y_node[1]]),
		spline.y_node,
		array([2*spline.y_node[-1] - spline.y_node[-2]]),
	])
	z_node = spline.z_node
	z_node = concatenate([
		expand_dims(z_node[0, :], axis=0),
		z_node,
		expand_dims(z_node[-1, :], axis=0),
	], axis=0)
	z_node = concatenate([
		expand_dims(z_node[:, 0], axis=1),
		z_node,
		expand_dims(z_node[:, -1], axis=1),
	], axis=1)

	# find out in what cell each input point is
	i_input = (y_input - y_node[0])/(y_node[1] - y_node[0])
	j_input = (x_input - x_node[0])/(x_node[1] - x_node[0])
	I_input = maximum(2, minimum(y_node.size - 2, ceil(i_input).astype(int)))
	J_input = maximum(2, minimum(x_node.size - 2, ceil(j_input).astype(int)))

	# apply the 4×4 convolution kernel
	result = zeros(shape(x_input) + shape(spline.z_node)[2:])
	row_weits = {Δi: bicubic_function(i_input - (I_input + Δi)) for Δi in range(-2, 2)}
	col_weits = {Δj: bicubic_function(j_input - (J_input + Δj)) for Δj in range(-2, 2)}
	for Δi in range(-2, 2):
		for Δj in range(-2, 2):
			result = result + row_weits[Δi]*col_weits[Δj]*z_node[I_input + Δi, J_input + Δj, ...]
	return result


def bicubic_function(Δi, a=-0.5):
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


class Spline:
	def __init__(self, input_x: NDArray, input_y: NDArray, z_node: NDArray):
		self.x_node = input_x
		self.y_node = input_y
		self.z_node = z_node


class PointSet:
	def __init__(self, angle: Optional[float], offset: Optional[float], points: NDArray):
		self.angle = angle
		self.offset = offset
		self.points = points
