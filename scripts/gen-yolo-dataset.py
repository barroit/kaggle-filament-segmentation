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
from yaml import dump

from pycocotools.coco import COCO

dataset = 'dataset'
output = f"{dataset}/yolo"

paths = {}
selected = {}

days = set()
verify_days = set()

coco = COCO(f"{dataset}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
rand = Random(39)

makedirs(f"{output}/train/images")
makedirs(f"{output}/train/labels")
makedirs(f"{output}/verify/images")
makedirs(f"{output}/verify/labels")

with open(f"{output}/dataset.yaml", 'w') as file:
	dump({
		'train': 'train/images',
		'val': 'verify/images',
		'names': { 0: 'filament' },
	}, file, indent=3)

for path in glob(f"{dataset}/train/train_images/*.jpeg"):
	name = basename(path)
	stem = splitext(name)[0]
	paths[stem] = path

for info in coco.imgs.values():
	name = basename(info['file_name'])
	stem = splitext(name)[0]

	if stem not in selected:
		selected[stem] = info

for stem in selected:
	days.add(stem[:8])

days = sorted(days)
rand.shuffle(days)

for day in days[:len(days) // 5]:
	verify_days.add(day)

for stem, info in selected.items():
	src_abs = abspath(paths[stem])
	src_base = basename(paths[stem])
	type = 'verify' if stem[:8] in verify_days else 'train'

	width = info['width']
	height = info['height']
	lines = []

	image = f"{output}/{type}/images/{src_base}"
	label = f"{output}/{type}/labels/{stem}.txt"

	anno_ids = coco.getAnnIds(imgIds=[info['id']])
	annos = coco.loadAnns(anno_ids)

	link(src_abs, image)

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
