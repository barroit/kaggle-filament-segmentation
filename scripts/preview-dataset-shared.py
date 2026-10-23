#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from itertools import combinations
from json import dump as json_dump, load as json_load
from multiprocessing import Pool as alloc_pool, cpu_count
from os import makedirs

from lib.coord import *
from lib.draw import *
from lib.kconfig import *
from lib.mask import *
from lib.path import *

prefix = 'preview/shared'
overlap_prefix = 'preview/shared/overlap'

def draw(task):
	name, annos = task

	__image = draw_load(f"{DATASET_SHARED_IMAGE_DIR}/{name}.png")
	image = __image.copy()
	drawn = set()

	for __idx, anno in enumerate(annos):
		bbox = coord_load_bbox_coco(anno['bbox'])
		border = coord_pack_int32(anno['polygon'])
		spine = coord_pack_int32(anno['spine'])
		idx = str(__idx + 1)

		image = draw_blend_spine(image, spine)
		image = draw_blend_area(image, border)
		image = draw_blend_border(image, border)
		image = draw_blend_bbox(image, bbox)
		image = draw_blend_label(image, bbox[0], idx)

	draw_dump(image, f"{prefix}/{name}.png")
	image = None

	for anno_a, anno_b in combinations(annos, 2):
		coords_a = coord_pack_int32(anno_a['polygon'])
		coords_b = coord_pack_int32(anno_b['polygon'])

		mask_a = mask_from_coords(coords_a)
		mask_b = mask_from_coords(coords_b)
		iou = mask_calc_iou(mask_a, mask_b)

		if not iou:
			continue

		if image is None:
			image = __image

		for anno in (anno_a, anno_b):
			id = anno['id']
			bbox = coord_load_bbox_coco(anno['bbox'])
			border = coord_pack_int32(anno['polygon'])
			spine = coord_pack_int32(anno['spine'])

			if id in drawn:
				continue

			image = draw_blend_spine(image, spine)
			image = draw_blend_area(image, border)
			image = draw_blend_border(image, border)
			image = draw_blend_label(image, bbox[0], id)

			drawn.add(id)

	if image is not None:
		draw_dump(image, f"{overlap_prefix}/{name}.png")

if __name__ == '__main__':
	cpus = cpu_count()
	pool = alloc_pool(cpus)

	src = open(DATASET_SHARED_ANNOTATIONS)
	annotations = json_load(src)
	tasks = annotations.items()

	makedirs(prefix)
	makedirs(overlap_prefix)
	pool.map(draw, tasks)
