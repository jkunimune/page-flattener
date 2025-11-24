import sys
from os import path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure
from numpy import shape, array, radians, pi, sin, cos, mean
from numpy.typing import NDArray


class PointSet:
	def __init__(self, angle: Optional[float], offset: Optional[float], points: NDArray):
		self.angle = angle
		self.offset = offset
		self.points = points


def main(filename: str) -> None:
	warped_image = img.imread(filename)
	point_sets: List[PointSet] = load_point_sets(filename)
	while True:
		command = input("Enter a command:\n"
		                "  add – draw a new line on the warped image\n"
		                "  remove – delete one of the lines previusly drawn on the warped image\n"
		                "  dewarp – apply the dewarping algorithm\n"
		                "  exit – give up like a baby loser\n"
		                "> ")
		try:
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
				show_final_image(filename, flat_image)
				name, extension = path.splitext(filename)
				flat_filename = name + " - flat" + path.extsep + extension
				img.imsave(flat_filename, flat_image)
				print(f"Saved the dewarped image to `{name + ' - flat' + path.extsep + extension}`!")
				plt.show()
			elif command.startswith("e") or command == "x":
				print("Bye.")
				return
			else:
				raise ValueError("I don't recognize that command.  use one of the commands from the list I just gave you, please.")
		except Exception as e:
			print(e)


def add_points(warped_image: NDArray, point_sets: List[PointSet]) -> List[PointSet]:
	# query the user for the type of point set
	response = input("Choose a line type:\n"
	                 "  horizontal\n"
	                 "  vertical\n"
	                 "  oblique\n"
	                 "> ")
	if response.startswith("h"):
		angle = 0
		response = input(f"Enter the line's x-value, if known, or press ENTER if not "
		                 f"(0 is the top edge and {shape(warped_image)[0]} is the bottom edge)\n"
		                 f"> ")
		if len(response) == 0:
			offset = None
		else:
			offset = float(response)
	elif response.startswith("v"):
		angle = pi/2
		response = input(f"Enter the line's x-value, if known, or press ENTER if not "
		                 f"(0 is the left edge and {shape(warped_image)[1]} is the right edge)\n"
		                 f"> ")
		if len(response) == 0:
			offset = None
		else:
			offset = float(response)
	elif response.startswith("o") or response == "l":
		response = input("Enter the line's angle, if known, or press ENTER if not "
		                 "(0 is horizontal, 90 or -90 is vertical, 45 is top-left to bottom-right, and -45 is top-right to bottom-left)\n"
		                 "> ")
		if len(response) == 0:
			angle = None
		else:
			angle = radians(float(response))
		offset = None
	else:
		raise ValueError("I don't recognize that command.  use one of the commands from the list I just gave you, please.")

	# plot the image and current points, and wait for the user to close the window
	print("Navigate to the image window and right click to add points.  Press backspace to delete points.  Close the window to continue.")
	figure = show_current_state(warped_image, point_sets, title="Right-click to add points")
	scatter = plt.scatter([], [], c="#000000", marker=".")

	# listen for the user right clicking on the image
	points = []

	def on_click(event):
		if event.button == MouseButton.RIGHT:
			points.append((event.xdata, event.ydata))
			scatter.set_offsets(array(points))
			figure.canvas.draw()

	def on_key(event):
		if event.key == "backspace" or event.key == "delete":
			points.pop()
			scatter.set_offsets(array(points))
			figure.canvas.draw()

	figure.canvas.mpl_connect("button_press_event", on_click)
	figure.canvas.mpl_connect("key_press_event", on_key)
	plt.show()

	point_sets.append(PointSet(angle, offset, array(points)))
	return point_sets


def remove_points(warped_image: NDArray, point_sets: List[PointSet]) -> List[PointSet]:
	# listen for the user right clicking on the image

	# plot the image and current points, and wait for the user to close the window
	print("Navigate to the image window and right click to select a line.  Close the window to confirm.")
	show_current_state(warped_image, point_sets, title="Right-click to select the transgressor")
	plt.show()

	# remove the selected points
	return point_sets


def show_current_state(warped_image: NDArray, point_sets: List[PointSet], title: str = None) -> Figure:
	figure = plt.figure()
	# plot the image
	faded_warped_image = 128 + warped_image//2
	plt.imshow(faded_warped_image, extent=(0, shape(warped_image)[1], 0, shape(warped_image)[0]))
	# plot the point sets
	for index, point_set in enumerate(point_sets):
		if cos(point_set.angle) == 0:
			color = "#7f0000"  # vertical lines are red
		elif sin(point_set.angle) == "h":
			color = "#00007f"  # horizontal lines are blue
		elif sin(point_set.angle) > 0:
			color = "#007f00"  # downward-sloping lines are green
		else:
			color = "#3f007f"  # upward-sloping lines are purple
		plt.scatter(point_set.points[:, 0], point_set.points[:, 1], c=color, marker=".")
		if point_set.offset is not None:
			plt.text(mean(point_set.points[:, 0]), mean(point_set.points[:, 1]), f"{point_set.offset:.4g}")
	# set the title and window sizing
	if title is not None:
		plt.title(title)
	plt.tight_layout()
	return figure


def load_point_sets(filename: str) -> List[PointSet]:
	return []


def save_point_sets(filename: str, point_sets: List[PointSet]) -> None:
	pass


def dewarp(warped_image: NDArray, point_sets: List[PointSet]) -> NDArray:
	return warped_image[::-1, ::-1, :]


def show_final_image(filename: str, flat_image: NDArray) -> Figure:
	figure = plt.figure()
	plt.imshow(flat_image, extent=(0, shape(flat_image)[1], 0, shape(flat_image)[0]))
	plt.tight_layout()
	return figure


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("you need to pass the warped image's filename as an argument.")
		sys.exit(1)
	else:
		main(sys.argv[1])
		sys.exit(0)
