#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#
# This only emits the first annotator's mark in the annotation set.

from os import link, makedirs
from random import Random
from yaml import dump as yaml_dump

from pycocotools.coco import COCO

from lib.coord import *
from lib.kconfig import *
from lib.path import *

images = {}
train_days = set()
verify_days = set()

coco = COCO(TRAIN_METADATA)
rand = Random(CONFIG_FOLD_SPLIT_SEED)
metadata = None

makedirs(YOLO_TRAIN_IMAGE_DIR)
makedirs(YOLO_TRAIN_LABEL_DIR)
makedirs(YOLO_VERIFY_IMAGE_DIR)
makedirs(YOLO_VERIFY_LABEL_DIR)

metadata = open(YOLO_METADATA, 'w')
yaml_dump({
	'train': YOLO_TRAIN_IMAGE_DIR.removeprefix(YOLO_PREFIX),
	'val': YOLO_VERIFY_IMAGE_DIR.removeprefix(YOLO_PREFIX),
	'names': {
		0: 'filament',
	},
}, metadata, indent=3)

for id, meta in coco.imgs.items():
	name = id.split('-')[1]

	if name not in images:
		images[name] = meta
		train_days.add(name[:CONFIG_FOLD_SPLIT_GRANULARITY])

train_days = sorted(train_days)
rand.shuffle(train_days)

for day in train_days[:len(train_days) * CONFIG_FOLD_VERIFY_RATIO // 10]:
	verify_days.add(day)

for name, info in images.items():
	filename = info['file_name']
	image_prefix = YOLO_TRAIN_IMAGE_DIR
	label_prefix = YOLO_TRAIN_LABEL_DIR

	label = None
	annos = coco.imgToAnns[info['id']]

	if name[:CONFIG_FOLD_SPLIT_GRANULARITY] in verify_days:
		image_prefix = YOLO_VERIFY_IMAGE_DIR
		label_prefix = YOLO_VERIFY_LABEL_DIR

	link(f"{TRAIN_IMAGE_DIR}/{filename}", f"{image_prefix}/{filename}")
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
