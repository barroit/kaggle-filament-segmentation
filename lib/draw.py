# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from cv2 import FONT_HERSHEY_SIMPLEX, IMREAD_COLOR, LINE_AA, \
		addWeighted as add_weighted, fillPoly as fill_poly, imread, \
		imwrite, polylines, putText as put_text, rectangle

from lib.kconfig import *

__all__ = [
	'draw_load',
	'draw_blend',
	'draw_rectangle',
	'draw_polyline',
	'draw_polylines',
	'draw_fill_polygon',
	'draw_text',
	'draw_text_border',
	'draw_dump',
]

draw_polylines = polylines
draw_fill_polygon = fill_poly

def draw_load(path):
	return imread(path, IMREAD_COLOR)

def draw_blend(image, overlay, overlay_alpha):
	return add_weighted(image, 1 - overlay_alpha, overlay,
			    overlay_alpha, 0)

def draw_rectangle(image, coords, color, width):
	rectangle(image, coords[0], coords[1], color, width)

def draw_polyline(image, coord, color, width):
	polylines(image, [coord], False, color, width)

def __draw_text(image, xy, text, color, thickness):
	pos = (xy[0], max(xy[1] - 5, 20))

	put_text(image, text, pos, FONT_HERSHEY_SIMPLEX,
		 CONFIG_DRAW_FONT_SCALE / 10, color, thickness, LINE_AA)

def draw_text(image, xy, text, color):
	__draw_text(image, xy, text, color, CONFIG_DRAW_FONT_THICKNESS)

def draw_text_border(image, xy, text, color):
	__draw_text(image, xy, text, color,
		    CONFIG_DRAW_FONT_THICKNESS + CONFIG_DRAW_TEXT_BORDER_WIDTH)

def draw_dump(image, path):
	imwrite(path, image)
