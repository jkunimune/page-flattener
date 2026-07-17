from typing import Callable

import torch
from numpy import array, meshgrid, all, nan, sqrt, isfinite
from numpy.core.numeric import isclose

from dewarp import paperlike_fraction, apply_spline, Spline, spline_gradient, spline_hessian, solve_quartic_equation, \
	solve_cubic_equation, solve_quadratic_equation

flat_spline = Spline(
	array([-10, 0, 10]),
	array([-10, 0, 10]),
	array([
		[5, 7, 9],
		[3, 5, 7],
		[1, 3, 5],
	]),
)

bent_spline = Spline(
	array([-10, 0, 10]),
	array([-10, 0, 10]),
	array([
		[0, 1, 0],
		[1, 0, 1],
		[0, 1, 0],
	]),
)

test_X, test_Y = meshgrid(
	array([-10, -5, 0, 5, 10]),
	array([-10, -5, 0, 5, 10]), indexing="xy")


def test_spline():
	assert all(isclose(
		apply_spline(test_X, test_Y, flat_spline),
		array([
			[5, 6, 7, 8, 9],
			[4, 5, 6, 7, 8],
			[3, 4, 5, 6, 7],
			[2, 3, 4, 5, 6],
			[1, 2, 3, 4, 5],
		]),
	))

	assert all(isclose(
		apply_spline(test_X, test_Y, bent_spline),
		array([
			[0.    , 0.625 , 1.    , 0.625 , 0.    ],
			[0.625 , 0.4688, 0.375 , 0.4688, 0.625 ],
			[1.    , 0.375 , 0.    , 0.375 , 1.    ],
			[0.625 , 0.4688, 0.375 , 0.4688, 0.625 ],
			[0.    , 0.625 , 1.    , 0.625 , 0.    ],
		]),
		atol=1e-4,
	))


def test_spline_gradient():
	assert all(isclose(
		spline_gradient(test_X, test_Y, flat_spline),
		array([
			[[.2, -.2], [.2, -.2], [.2, -.2], [.2, -.2], [.2, -.2]],
			[[.2, -.2], [.2, -.2], [.2, -.2], [.2, -.2], [.2, -.2]],
			[[.2, -.2], [.2, -.2], [.2, -.2], [.2, -.2], [.2, -.2]],
			[[.2, -.2], [.2, -.2], [.2, -.2], [.2, -.2], [.2, -.2]],
			[[.2, -.2], [.2, -.2], [.2, -.2], [.2, -.2], [.2, -.2]],
		]),
	))

	assert all(isclose(
		spline_gradient(test_X, test_Y, bent_spline),
		array([
			[[ 0.1  ,  0.1  ], [ 0.125, -0.025], [ 0.   , -0.1  ], [-0.125, -0.025], [-0.1  ,  0.1  ]],
			[[-0.025,  0.125], [-0.031, -0.031], [ 0.   , -0.125], [ 0.031, -0.031], [ 0.025,  0.125]],
			[[-0.1  ,  0.   ], [-0.125,  0.   ], [ 0.   ,  0.   ], [ 0.125,  0.   ], [ 0.1  ,  0.   ]],
			[[-0.025, -0.125], [-0.031,  0.031], [ 0.   ,  0.125], [ 0.031,  0.031], [ 0.025, -0.125]],
			[[ 0.1  , -0.1  ], [ 0.125,  0.025], [ 0.   ,  0.1  ], [-0.125,  0.025], [-0.1  , -0.1  ]],
		]),
		atol=1e-3,
	))


def test_spline_hessian():
	assert all(isclose(
		spline_hessian(5, 5, flat_spline),
		array([
			[0, 0],
			[0, 0],
		]),
	))

	assert all(isclose(
		spline_hessian(5, 5, bent_spline),
		array([
			[ 0.0025 , -0.03125],
			[-0.03125,  0.0025 ],
		]),
	))


def test_quartic_solver():
	# four roots
	assert all(isclose(
		unvectorize(solve_quartic_equation)(10., -100., 350., -500., 240.),
		[4., 3., 2., 1.],
		atol=1e-3,
	))

	# two roots
	assert all(isclose(
		unvectorize(solve_quartic_equation)(10., -80., 240., -320., 150.),
		[3., 1., nan, nan],
		atol=1e-3, equal_nan=True,
	))

	# no roots
	assert all(isclose(
		unvectorize(solve_quartic_equation)(10., -80., 240., -320., 170.),
		[nan, nan, nan, nan],
		atol=1e-3, equal_nan=True,
	))

	# multiple-root
	assert all(isclose(
		unvectorize(solve_quartic_equation)(10., -80., 240., -320., 160.),
		[2., 2., 2., 2.],
		atol=1e-3,
	))

	# cubic
	assert all(isclose(
		unvectorize(solve_quartic_equation)(0., 10., -60., 110., -60.),
		[3., 2., 1., nan],
		atol=1e-4, equal_nan=True,
	))


def test_cubic_solver():
	# three roots
	assert all(isclose(
		unvectorize(solve_cubic_equation)(10., -60., 110., -60.),
		[(3., 0.), (2., 0.), (1., 0.)],
	))

	# one root
	assert all(isclose(
		unvectorize(solve_cubic_equation)(10., -60., 120., -90.),
		[(3., 0.), (1.5, sqrt(3)/2), (1.5, -sqrt(3)/2)],
	))

	# multiple-root
	assert all(isclose(
		unvectorize(solve_cubic_equation)(10., -60., 120., -80.),
		[(2., 0.), (2., 0.), (2., 0.)],
	))
	# quadratic
	assert all(isclose(
		unvectorize(solve_cubic_equation)(0., 10., -30., 20.),
		[(2., 0.), (1., 0.), (nan, nan)],
		equal_nan=True,
	))


def test_quadratic_solver():
	# two roots
	assert all(isclose(
		unvectorize(solve_quadratic_equation)(10., -40., 30.),
		[(3., 0.), (1., 0.)],
	))

	# no roots
	assert all(isclose(
		unvectorize(solve_quadratic_equation)(10., -40., 50.),
		[(2., 1.), (2., -1.)],
	))

	# multiple-root
	assert all(isclose(
		unvectorize(solve_quadratic_equation)(10., -40., 40.),
		[(2., 0.), (2., 0.)],
	))

	# linear
	assert all(isclose(
		unvectorize(solve_quadratic_equation)(0., 10., -10.),
		[(1., 0.), (nan, nan)],
		equal_nan=True,
	))

	# oops all zero
	assert all(isclose(
		unvectorize(solve_quadratic_equation)(0., 0., 0.),
		[(nan, nan), (nan, nan)],
		equal_nan=True,
	))


def test_paperlike_fraction():
	pure_x_hessian = torch.tensor([[
		[1, -2],
		[-2, 4],
	]])
	pure_y_hessian = torch.tensor([[
		[-3, 6],
		[6, -12],
	]])
	assert paperlike_fraction(pure_x_hessian, pure_y_hessian) == 1

	mixed_x_hessian = torch.tensor([[
		[1 + 12, -2 + 6],
		[-2 + 6, 4 + 3],
	]])
	mixed_y_hessian = torch.tensor([[
		[-3 + 4, 6 + 2],
		[6 + 2, -12 + 1],
	]])
	assert isclose(paperlike_fraction(mixed_x_hessian, mixed_y_hessian), 1/2)

	rando_x_hessian = torch.tensor([[
		[1, 2],
		[2, -3],
	]])
	rando_y_hessian = torch.tensor([[
		[6, -5],
		[-5, -4],
	]])
	assert isclose(paperlike_fraction(rando_x_hessian, rando_y_hessian), 0.552543, atol=1e-6)

	# there's some shenanigans that happens when one of the principal components is [[0, 1], [1, 0]]
	degenerate_x_hessian = torch.tensor([[
		[0, -1],
		[-1, 0],
	]])
	degenerate_y_hessian = torch.tensor([[
		[2, 0],
		[0, 0],
	]])
	assert isclose(paperlike_fraction(degenerate_x_hessian, degenerate_y_hessian), 2/3)

	# oops all zero
	flat_x_hessian = torch.tensor([[
		[0, 0],
		[0, 0],
	]])
	flat_y_hessian = torch.tensor([[
		[0, 0],
		[0, 0],
	]])
	assert isfinite(paperlike_fraction(flat_x_hessian, flat_y_hessian))


def unvectorize(func: Callable) -> Callable:
	def call_func_on_floats(*args: float):
		tensor_result = func(*(torch.tensor(x) for x in args))
		return recursively_extract_tensors(tensor_result)
	return call_func_on_floats


def recursively_extract_tensors(structure):
	if type(structure) is torch.Tensor:
		return structure.numpy()
	else:
		return [recursively_extract_tensors(component) for component in structure]
