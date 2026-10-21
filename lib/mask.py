# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from cv2 import fillPoly as fill_poly
from numpy import count_nonzero, uint8, zeros

from lib.kconfig import *

__all__ = [
	'mask_from_coords',
	'mask_calc_iou',
	'mask_calc_coverage',
]

def mask_from_coords(coords):
	mask = zeros(CONFIG_IMAGE_DIMENSION[::-1], dtype=uint8)

	fill_poly(mask, [coords], 1)
	return mask

def mask_calc_iou(mask_a, mask_b):
	intersection = count_nonzero(mask_a & mask_b)
	union = count_nonzero(mask_a | mask_b)
	iou = intersection / union

	return iou

def mask_calc_coverage(mask_a, mask_b):
	intersection = count_nonzero(mask_a & mask_b)
	area_a = count_nonzero(mask_a)
	area_b = count_nonzero(mask_b)
	coverage = intersection / min(area_a, area_b)

	return coverage
