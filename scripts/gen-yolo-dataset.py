#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#
# This only emits the first annotator's mark in the annotation set.

from glob import glob
from os import link, makedirs
from os.path import abspath, basename, splitext
from random import Random
from yaml import dump as yaml_dump

from pycocotools.coco import COCO

from lib.path import *

images = {}
selected = {}

days = set()
verify_days = set()

coco = COCO(TRAIN_METADATA)
rand = Random(39)

makedirs(YOLO_TRAIN_IMAGE_DIR)
makedirs(YOLO_TRAIN_LABEL_DIR)
makedirs(YOLO_VERIFY_IMAGE_DIR)
makedirs(YOLO_VERIFY_LABEL_DIR)

with open(YOLO_METADATA, 'w') as file:
	data = {
		'train': YOLO_TRAIN_IMAGE_DIR.removeprefix(YOLO_PREFIX),
		'val': YOLO_VERIFY_IMAGE_DIR.removeprefix(YOLO_PREFIX),
		'names': {
			0: 'filament',
		},
	}

	yaml_dump(data, file, indent=3)

for path in glob(GLOB_TRAIN_IMAGES):
	stem = path_resolve_stem(path)

	images[stem] = path

for info in coco.imgs.values():
	stem = path_resolve_stem(info['file_name'])

	if stem not in selected:
		selected[stem] = info

for stem in selected:
	days.add(stem[:8])

days = sorted(days)
rand.shuffle(days)

for day in days[:len(days) // 5]:
	verify_days.add(day)

for stem, info in selected.items():
	filename = abspath(images[stem])
	name = basename(images[stem])
	type = 'verify' if stem[:8] in verify_days else 'train'

	width = info['width']
	height = info['height']
	lines = []

	image = f"{YOLO_PREFIX}/{type}/images/{name}"
	label = f"{YOLO_PREFIX}/{type}/labels/{stem}.txt"

	anno_ids = coco.getAnnIds(imgIds=[info['id']])
	annos = coco.loadAnns(anno_ids)

	link(filename, image)

	for anno in annos:
		polygon = anno['segmentation'][0]
		coords = []

		for idx, val in enumerate(polygon):
			scale = width

			if idx % 2 != 0:
				scale = height

			coords.append(f"{val / scale:.12f}")

		lines.append('0 ' + ' '.join(coords))

	with open(label, 'w') as file:
		file.write('\n'.join(lines) + '\n')
