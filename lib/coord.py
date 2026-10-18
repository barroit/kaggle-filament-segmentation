# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from cv2 import boundingRect as bounding_rect
from numpy import array, float64, int32, rint

from lib.kconfig import *

__all__ = [
	'coord_pack',
	'coord_pack_float64',
	'coord_load_bbox_coco',
	'coord_load_label_yolo',
	'coord_calc_bbox_yolo',
]

def coord_pack_float64(flatten):
	return array(flatten, dtype=float64).reshape(-1, 2)

def coord_pack(flatten):
	coords_float = coord_pack_float64(flatten)

	return rint(coords_float).astype(int32)

def coord_load_bbox_coco(flatten):
	x, y, width, height = flatten

	return coord_pack([x, y, x + width, y + height])

def coord_load_label_yolo(line):
	cols = line.split()
	scale = coord_pack_float64(cols[1:])

	return rint(scale * ([CONFIG_IMAGE_DIMENSION] * 2)).astype(int32)

def coord_calc_bbox_yolo(coords):
	x, y, width, height = bounding_rect(coords)

	return (x, y, x + width, y + height)
