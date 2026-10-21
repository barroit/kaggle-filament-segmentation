# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from math import ceil, floor

from cv2 import boundingRect as bounding_rect
from numpy import array, diff, float64, hypot, int32, rint

from lib.kconfig import *

__all__ = [
	'coord_load_bbox_coco',
	'coord_load_label_yolo',
	'coord_pack_int32',
	'coord_pack_float64',
	'coord_calc_bbox',
	'coord_snap_bbox',
	'coord_calc_bbox_intersection',
	'coord_calc_spine_length',
]

def coord_pack_float64(flatten):
	return array(flatten, dtype=float64).reshape(-1, 2)

def coord_pack_int32(flatten):
	coords_float = coord_pack_float64(flatten)

	return rint(coords_float).astype(int32)

def coord_load_bbox_coco(flatten):
	x, y, width, height = flatten

	return coord_pack_int32([x, y, x + width, y + height])

def coord_load_label_yolo(line):
	cols = line.split()
	scale = coord_pack_float64(cols[1:])

	return rint(scale * CONFIG_IMAGE_DIMENSION).astype(int32)

def coord_calc_bbox(coords):
	x, y, width, height = bounding_rect(coords)

	return (x, y, x + width, y + height)

def coord_snap_bbox(bbox):
	x1, y1, x2, y2 = bbox

	x1 = floor(x1)
	y1 = floor(y1)
	x2 = ceil(x2)
	y2 = ceil(y2)

	return (x1, y1, x2, y2)

def coord_calc_bbox_intersection(bbox_a, bbox_b):
	ax1, ay1, ax2, ay2 = bbox_a
	bx1, by1, bx2, by2 = bbox_b

	width = max(0, min(ax2, bx2) - max(ax1, bx1))
	height = max(0, min(ay2, by2) - max(ay1, by1))
	intersection = width * height

	return intersection

def coord_calc_spine_length(coords):
	delta = diff(coords, axis=0)

	return hypot(delta[:, 0], delta[:, 1]).sum()
