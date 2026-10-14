#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from multiprocessing import Pool, cpu_count
from os import makedirs

from PIL import Image, ImageDraw
from pycocotools.coco import COCO

dataset = 'dataset'
output_prefix = f"{dataset}/preview"
coco = COCO(f"{dataset}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")

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

	filename = f"{dataset}/train/train_images/{info['file_name']}"
	image_raw = Image.open(filename)
	image = image_raw.convert('RGB')
	draw = ImageDraw.Draw(image, 'RGBA')

	for anno in annos:
		x, y, width, height = anno['bbox']
		coord = (x, y, x + width, y + height)
		spine = flat_to_points(anno['spine'])

		draw.rectangle(coord, outline=(255, 255, 0, 128), width=1)

		for segs in anno['segmentation']:
			polygon = flat_to_points(segs)

			draw.line(polygon + polygon[:1],
				  fill=(0, 255, 255, 30), width=1)

		draw.line(spine, fill=(255, 0, 0, 30), width=1)

	image.save(f"{output_prefix}/{id}.png")

if __name__ == '__main__':
	cpus = cpu_count()
	pool = Pool(cpus)
	ids = coco.getImgIds()

	makedirs(output_prefix, exist_ok=True)
	pool.map(draw_image, ids)
