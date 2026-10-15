#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from multiprocessing import Pool, cpu_count
from os import makedirs

from PIL import Image, ImageDraw
from pycocotools.coco import COCO

from lib.kconfig import *

output = f"{CONFIG_DATASET_PREFIX}/preview/train"
coco = COCO(f"{CONFIG_DATASET_PREFIX}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")

def flat_to_points(coord):
	x_set = coord[::2]
	y_set = coord[1::2]

	coord_set = zip(x_set, y_set)
	points = list(coord_set)

	return points

def draw_image(id):
	info = coco.loadImgs([id])[0]
	anno_ids = coco.getAnnIds(imgIds=[id])
	annos = coco.loadAnns(anno_ids)

	image_raw = Image.open(f"{CONFIG_DATASET_PREFIX}/train/train_images/{info['file_name']}")
	image = image_raw.convert('RGB')
	draw = ImageDraw.Draw(image, 'RGBA')

	for anno in annos:
		x, y, width, height = anno['bbox']
		coord = (x, y, x + width, y + height)
		spine = flat_to_points(anno['spine'])

		draw.rectangle(coord, outline=CONFIG_PREVIEW_BBOX_RGBA,
			       width=CONFIG_PREVIEW_BBOX_WIDTH)

		for segs in anno['segmentation']:
			polygon = flat_to_points(segs)

			draw.line(polygon + polygon[:1],
				  fill=CONFIG_PREVIEW_BORDER_RGBA,
				  width=CONFIG_PREVIEW_BORDER_WIDTH)

		draw.line(spine, fill=CONFIG_PREVIEW_SPINE_RGBA,
			  width=CONFIG_PREVIEW_SPINE_WIDTH)

	image.save(f"{output}/{id}.png")

if __name__ == '__main__':
	cpus = cpu_count()
	pool = Pool(cpus)
	ids = coco.getImgIds()

	makedirs(output, exist_ok=True)
	pool.map(draw_image, ids)
