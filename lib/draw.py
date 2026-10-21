# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from cv2 import FONT_HERSHEY_SIMPLEX, IMREAD_COLOR, LINE_AA, \
		addWeighted as add_weighted, fillPoly as fill_poly, imread, \
		imwrite, polylines, putText as put_text, rectangle

from lib.color import *
from lib.kconfig import *

__all__ = [
	'draw_load',
	'draw_dump',
	'draw_blend',

	'draw_rectangle',
	'draw_polyline',
	'draw_polylines',
	'draw_fill_polygon',
	'draw_label',
	'draw_label_border',

	'draw_blend_bbox',
	'draw_blend_spine',
	'draw_blend_border',
	'draw_blend_label',
	'draw_blend_area',
]

draw_polylines = polylines
draw_fill_polygon = fill_poly

bbox_color = color_rgba_to_bgra(CONFIG_DRAW_BBOX_RGBA)
bbox_alpha = color_alpha_percent(bbox_color)

spine_color = color_rgba_to_bgra(CONFIG_DRAW_SPINE_RGBA)
spine_alpha = color_alpha_percent(spine_color)

border_color = color_rgba_to_bgra(CONFIG_DRAW_BORDER_RGBA)
border_alpha = color_alpha_percent(border_color)

label_color = color_rgba_to_bgra(CONFIG_DRAW_LABEL_RGBA)
label_alpha = color_alpha_percent(label_color)

label_border_color = color_rgba_to_bgra(CONFIG_DRAW_LABEL_BORDER_RGBA)
label_border_alpha = color_alpha_percent(label_border_color)

fill_color = color_rgba_to_bgra(CONFIG_DRAW_FILL_RGBA)
fill_alpha = color_alpha_percent(fill_color)

font_scale = CONFIG_DRAW_FONT_SCALE / 10

def draw_load(path):
	return imread(path, IMREAD_COLOR)

def draw_blend(image, overlay, overlay_alpha):
	return add_weighted(image, 1 - overlay_alpha, overlay,
			    overlay_alpha, 0)

def draw_dump(image, path):
	imwrite(path, image)

def draw_rectangle(image, coords, color, width):
	rectangle(image, coords[0], coords[1], color, width)

def draw_polyline(image, coord, color, width):
	polylines(image, [coord], False, color, width)

def __draw_text(image, xy, text, color, thickness):
	pos = (xy[0], max(xy[1] - 5, 20))

	put_text(image, text, pos, FONT_HERSHEY_SIMPLEX, font_scale, color,
		 thickness, LINE_AA)

def draw_label(image, xy, text, color):
	__draw_text(image, xy, text, color, CONFIG_DRAW_FONT_THICKNESS)

def draw_label_border(image, xy, text, color):
	__draw_text(image, xy, text, color,
		    CONFIG_DRAW_FONT_THICKNESS + CONFIG_DRAW_LABEL_BORDER_WIDTH)

def draw_blend_bbox(image, coords):
	overlay = image.copy()

	draw_rectangle(overlay, coords, bbox_color, CONFIG_DRAW_BBOX_WIDTH)
	return draw_blend(image, overlay, bbox_alpha)

def draw_blend_spine(image, coords):
	overlay = image.copy()

	draw_polyline(overlay, coords, spine_color, CONFIG_DRAW_SPINE_WIDTH)
	return draw_blend(image, overlay, spine_alpha)

def draw_blend_border(image, coords):
	overlay = image.copy()

	draw_polyline(overlay, coords, border_color, CONFIG_DRAW_BORDER_WIDTH)
	return draw_blend(image, overlay, border_alpha)

def draw_blend_label(image, xy, label):
	overlay = image.copy()

	draw_label(overlay, xy, label, label_color)
	draw_label_border(overlay, xy, label, label_border_color)
	return draw_blend(image, overlay, label_alpha)

def draw_blend_area(image, coords):
	overlay = image.copy()

	draw_fill_polygon(overlay, [coords], fill_color)
	return draw_blend(image, overlay, fill_alpha)
