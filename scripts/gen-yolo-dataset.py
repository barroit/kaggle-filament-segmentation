#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#
# This only emits the first annotator's mark in the annotation set.

from os import link, makedirs
from yaml import dump as yaml_dump

from pycocotools.coco import COCO

from lib.coord import *
from lib.path import *

images = {}

coco = COCO(MAGFILO_METADATA_FILE)
metadata = None

makedirs(DATASET_YOLO_TRAIN_IMAGE_DIR)
makedirs(DATASET_YOLO_TRAIN_LABEL_DIR)
makedirs(DATASET_YOLO_VERIFY_IMAGE_DIR)
makedirs(DATASET_YOLO_VERIFY_LABEL_DIR)

train_dir = DATASET_YOLO_TRAIN_IMAGE_DIR.removeprefix(f"{DATASET_YOLO_DIR}/")
verify_dir = DATASET_YOLO_VERIFY_IMAGE_DIR.removeprefix(f"{DATASET_YOLO_DIR}/")

metadata = open(DATASET_YOLO_METADATA, 'w')
yaml_dump({
	'train': train_dir,
	'val': verify_dir,
	'names': {
		0: 'filament',
	},
}, metadata, indent=3)

for id, meta in coco.imgs.items():
	name = id.split('-')[1]

	if name not in images:
		images[name] = meta

for name, info in images.items():
	filename = info['file_name']
	image_prefix = DATASET_YOLO_TRAIN_IMAGE_DIR
	label_prefix = DATASET_YOLO_TRAIN_LABEL_DIR

	label = None
	annos = coco.imgToAnns[info['id']]

	link(f"{MAGFILO_IMAGE_DIR}/{filename}", f"{image_prefix}/{filename}")
	label = open(f"{label_prefix}/{name}.txt", 'w')

	for anno in annos:
		coords = coord_pack_float64(anno['segmentation'][0])

		label.write('0')
		for coord in coords:
			x = coord[0] / info['width']
			y = coord[1] / info['height']

			label.write(f" {x:.12f} {y:.12f}")

		label.write('\n')

	label.close()
