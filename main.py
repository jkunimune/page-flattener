import json
import sys
from os import path
from typing import List, Optional, Tuple, Any, Dict

import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure
from numpy import shape, array, radians, pi, sin, cos, mean, hypot, inf, empty
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
		                "  horizontal – draw a new horizontal line on the warped image\n"
		                "  vertical – draw a new vertical line on the warped image\n"
		                "  oblique – draw a new oblique line on the warped image\n"
		                "  remove – delete one of the lines previusly drawn on the warped image\n"
		                "  dewarp – apply the dewarping algorithm\n"
		                "  exit – give up like a baby loser\n"
		                "> ")
		try:
			if command.startswith("h") or command.startswith("v") or command.startswith("o") or command == "l":
				point_sets = add_points(command, warped_image, point_sets)
				save_point_sets(filename, point_sets)
			elif command.startswith("r") or command == "-":
				point_sets = remove_points(warped_image, point_sets)
				save_point_sets(filename, point_sets)
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


def add_points(kind: str, warped_image: NDArray, point_sets: List[PointSet]) -> List[PointSet]:
	# query the user for the type of point set
	if kind.startswith("h"):
		angle = 0
		response = input(f"Enter the line's y-value, if known, or press ENTER if not "
		                 f"(0 is the top edge and {shape(warped_image)[0]} is the bottom edge)\n"
		                 f"> ")
		if len(response) == 0:
			offset = None
		else:
			offset = float(response)
	elif kind.startswith("v"):
		angle = pi/2
		response = input(f"Enter the line's x-value, if known, or press ENTER if not "
		                 f"(0 is the left edge and {shape(warped_image)[1]} is the right edge)\n"
		                 f"> ")
		if len(response) == 0:
			offset = None
		else:
			offset = float(response)
	elif kind.startswith("o") or kind == "l":
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

	# plot the image and current points
	print("Navigate to the image window and right click to add points.  Press backspace to delete points.  Close the window to continue.")
	figure = show_current_state(warped_image, point_sets, title="Right-click to add points")
	scatter = plt.scatter([], [], c="#000000", marker=".")

	# listen for the user right clicking on the image
	points: List[Tuple[float, float]] = []

	def on_click(event):
		if event.button == MouseButton.RIGHT:
			points.append((event.xdata, event.ydata))
			scatter.set_offsets(array(points))
			figure.canvas.draw()

	def on_key(event):
		if event.key == "backspace" or event.key == "delete":
			if len(points) > 0:
				points.pop()
			scatter.set_offsets(array(points))
			figure.canvas.draw()

	figure.canvas.mpl_connect("button_press_event", on_click)
	figure.canvas.mpl_connect("key_press_event", on_key)

	# wait for the user to close the window
	plt.show()

	# add the new points to the main list
	if len(points) > 0:
		point_sets.append(PointSet(angle, offset, array(points)))
		print("Successfully added a line!")
	else:
		print("Declined to add a line.")
	return point_sets


def remove_points(warped_image: NDArray, point_sets: List[PointSet]) -> List[PointSet]:
	# plot the image and current points, and wait for the user to close the window
	print("Navigate to the image window and right click to select a line.  Press escape to deselect.  Close the window to continue.")
	figure = show_current_state(warped_image, point_sets, title="Right-click to select the transgressor")
	scatter = plt.scatter([], [], c="#000000", marker="x")

	# listen for the user right clicking on the image
	selected_index: Optional[int] = None

	def on_click(event):
		nonlocal selected_index
		if event.button == MouseButton.RIGHT:
			selected_index = None
			nearest_distance = inf
			for index, point_set in enumerate(point_sets):
				distance = hypot(
					event.xdata - point_set.points[:, 0],
					event.ydata - point_set.points[:, 1]).min()
				if distance < nearest_distance:
					nearest_distance = distance
					selected_index = index
			if selected_index is not None:
				scatter.set_offsets(point_sets[selected_index].points)
			figure.canvas.draw()

	def on_key(event):
		nonlocal selected_index
		if event.key == "escape":
			selected_index = None
			scatter.set_offsets(empty((0, 2)))
			figure.canvas.draw()

	figure.canvas.mpl_connect("button_press_event", on_click)
	figure.canvas.mpl_connect("key_press_event", on_key)

	# wait for the user to close the window
	plt.show()

	# remove the selected points from the main list
	if selected_index is not None:
		point_sets.pop(selected_index)
		print("Successfully removed a line!")
	else:
		print("Declined to remove a line.")
	return point_sets


def show_current_state(warped_image: NDArray, point_sets: List[PointSet], title: str = None) -> Figure:
	figure = plt.figure()
	# plot the image
	faded_warped_image = 128 + warped_image//2
	plt.imshow(faded_warped_image, extent=(0, shape(warped_image)[1], 0, shape(warped_image)[0]))
	# plot the point sets
	for index, point_set in enumerate(point_sets):
		if point_set.angle is None:
			color = "#3f3f3f"  # ambiguus lines are gray
		elif abs(cos(point_set.angle)) < 1e-15:
			color = "#7f0000"  # vertical lines are red
		elif abs(sin(point_set.angle)) < 1e-15:
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
	try:
		with open(filename + " reference points.json", "r") as file:
			data = json.load(file)
	except FileNotFoundError:
		return []
	point_sets = []
	for datum in data:
		point_sets.append(PointSet(datum["angle"], datum["offset"], array(datum["points"])))
	return point_sets


def save_point_sets(filename: str, point_sets: List[PointSet]) -> None:
	data: List[Dict[str, Any]] = []
	for point_set in point_sets:
		data.append({"angle": point_set.angle, "offset": point_set.offset, "points": point_set.points.tolist()})
	with open(filename + " reference points.json", "w") as file:
		json.dump(data, file, indent="\t")


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
