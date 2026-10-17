# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from cv2 import IMREAD_COLOR, addWeighted as add_weighted, \
		fillPoly, imread, imwrite, polylines, rectangle

__all__ = [
	'draw_load',
	'draw_blend',
	'draw_rectangle',
	'draw_polyline',
	'draw_polylines',
	'draw_fill_polygon',
	'draw_dump',
]

draw_polylines = polylines
draw_fill_polygon = fillPoly

def draw_load(path):
	return imread(path, IMREAD_COLOR)

def draw_blend(image, overlay, overlay_alpha):
	return add_weighted(image, 1 - overlay_alpha, overlay,
			    overlay_alpha, 0)

def draw_rectangle(image, coords, color, width):
	rectangle(image, coords[0], coords[1], color, width)

def draw_polyline(image, coord, color, width):
	polylines(image, [coord], False, color, width)

def draw_dump(image, path):
	imwrite(path, image)
