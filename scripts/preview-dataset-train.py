#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from multiprocessing import Pool, cpu_count
from os import makedirs

from pycocotools.coco import COCO

from lib.coord import *
from lib.draw import *
from lib.path import *

prefix = f"{DATASET_PREVIEW_DIR}/train"

def draw(task):
	src, dst, annos = task
	image = draw_load(src)

	for __idx, anno in enumerate(annos):
		bbox = coord_load_bbox_coco(anno['bbox'])
		spine = coord_pack_int32(anno['spine'])
		border = coord_pack_int32(anno['segmentation'][0])
		idx = str(__idx + 1)

		image = draw_blend_bbox(image, bbox)
		image = draw_blend_spine(image, spine)
		image = draw_blend_border(image, border)
		image = draw_blend_label(image, bbox[0], idx)

	draw_dump(image, dst)

if __name__ == '__main__':
	cpus = cpu_count()
	pool = Pool(cpus)

	coco = COCO(MAGFILO_METADATA_FILE)
	ids = coco.getImgIds()
	tasks = []

	for id in ids:
		annotator, name = id.split('-')
		filename = coco.imgs[id]['file_name']

		src = f"{MAGFILO_IMAGE_DIR}/{filename}"
		dst = f"{prefix}/{name}-{annotator}.png"
		annos = coco.imgToAnns[id]

		tasks.append([src, dst, annos])

	makedirs(prefix)
	pool.map(draw, tasks)
