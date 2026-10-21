#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from multiprocessing import Pool, cpu_count
from os import makedirs

from pycocotools.coco import COCO

from lib.color import *
from lib.coord import *
from lib.draw import *
from lib.kconfig import *
from lib.path import *

prefix = f"{DATASET_PREVIEW_DIR}/train"

bbox_color = color_rgba_to_bgra(CONFIG_DRAW_BBOX_RGBA)
bbox_alpha = color_alpha_percent(bbox_color)

spine_color = color_rgba_to_bgra(CONFIG_DRAW_SPINE_RGBA)
spine_alpha = color_alpha_percent(spine_color)

border_color = color_rgba_to_bgra(CONFIG_DRAW_BORDER_RGBA)
border_alpha = color_alpha_percent(border_color)

label_color = color_rgba_to_bgra(CONFIG_DRAW_TEXT_RGBA)
label_alpha = color_alpha_percent(label_color)

label_border_color = color_rgba_to_bgra(CONFIG_DRAW_TEXT_BORDER_RGBA)
label_border_alpha = color_alpha_percent(label_border_color)

def draw(task):
	src, dst, annos = task
	image = draw_load(src)

	for __idx, anno in enumerate(annos):
		bbox = coord_load_bbox_coco(anno['bbox'])
		spine = coord_pack_int32(anno['spine'])
		border = coord_pack_int32(anno['segmentation'][0])
		idx = str(__idx + 1)
		overlay = image.copy()

		draw_rectangle(overlay, bbox, bbox_color,
			       CONFIG_DRAW_BBOX_WIDTH)
		image = draw_blend(image, overlay, bbox_alpha)

		overlay = image.copy()
		draw_polyline(overlay, spine, spine_color,
			      CONFIG_DRAW_SPINE_WIDTH)
		image = draw_blend(image, overlay, spine_alpha)

		overlay = image.copy()
		draw_polyline(overlay, border, border_color,
			      CONFIG_DRAW_BORDER_WIDTH)
		image = draw_blend(image, overlay, border_alpha)

		overlay = image.copy()
		draw_text(overlay, bbox[0], idx, label_color)
		draw_text_border(overlay, bbox[0], idx, label_border_color)
		image = draw_blend(image, overlay, label_border_alpha)

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
