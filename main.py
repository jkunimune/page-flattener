import sys
from os import path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.image as img
from numpy.typing import NDArray


class PointSet:
	def __init__(self, kind: str, args: List[float], points: List[Tuple[float, float]]):
		self.kind = kind
		self.args = args
		self.points = points


def main(filename: str):
	warped_image = img.imread(filename)
	point_sets: List[PointSet] = load_point_sets(filename)
	while True:
		command = input("Enter a command:\n"
		                "  add – draw a new line on the warped image\n"
		                "  remove – delete one of the lines previusly drawn on the warped image\n"
		                "  dewarp – apply the dewarping algorithm\n"
		                "  exit – give up like a baby loser\n"
		                "> ")
		if command.startswith("a") or command == "+":
			point_sets = add_points(warped_image, point_sets)
			save_point_sets(filename, point_sets)
			print("Successfully added a line!")
		elif command.startswith("r") or command == "-":
			point_sets = remove_points(warped_image, point_sets)
			save_point_sets(filename, point_sets)
			print("Successfully removed a line!")
		elif command.startswith("d") or command == "w":
			flat_image = dewarp(warped_image, point_sets)
			save_and_show_final_image(filename, flat_image)
			print("Saved the dewarped image!")
		elif command.startswith("e") or command == "x":
			return
		else:
			print("I don't recognize that command.  use one of the commands from the list I just gave you, please.")


def add_points(warped_image: NDArray, point_sets: List[PointSet]) -> List[PointSet]:
	# query the user for the type of point set
	# listen for the user right clicking on the image
	# plot the image and current points
	show_current_state(warped_image, point_sets, title="Right-click to add points")
	# add the new point set to the list
	return point_sets


def remove_points(warped_image: NDArray, point_sets: List[PointSet]) -> List[PointSet]:
	# listen for the user right clicking on the image
	# plot the image and current points
	show_current_state(warped_image, point_sets, title="Right-click to select the transgressor")
	# remove the selected points
	return point_sets


def show_current_state(warped_image: NDArray, point_sets: List[PointSet], title: str):
	faded_warped_image = 128 + warped_image//2
	plt.figure()
	plt.imshow(faded_warped_image)
	plt.title(title)
	plt.tight_layout()
	plt.show()


def load_point_sets(filename: str) -> List[PointSet]:
	return []


def save_point_sets(filename: str, point_sets: List[PointSet]):
	pass


def dewarp(warped_image: NDArray, point_sets: List[PointSet]) -> NDArray:
	return warped_image[::-1, ::-1, :]


def save_and_show_final_image(filename: str, flat_image: NDArray):
	name, extension = path.splitext(filename)
	img.imsave(name + " - flat" + path.extsep + extension, flat_image)
	plt.figure()
	plt.imshow(flat_image)
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("you need to pass the warped image's filename as an argument.")
		sys.exit(1)
	else:
		main(sys.argv[1])
		sys.exit(0)
