#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from json import load as json_load
from os import link, makedirs
from os.path import abspath
from yaml import dump as yaml_dump

from lib.coord import *
from lib.kconfig import *
from lib.path import *

periods_src = open(DATASET_SHARED_PERIODS)
periods = json_load(periods_src)

annotations_src = open(DATASET_SHARED_ANNOTATIONS)
annotations = json_load(annotations_src)

metadata_src = None
metadata = {
	'train': 'train/images',
	'val': 'validate/images',
	'test': 'test/images',
	'names': {
		0: 'filament',
	},
}

width, height = CONFIG_IMAGE_DIMENSION

for set in ('train', 'validate', 'test'):
	makedirs(f"{DATASET_YOLO_DIR}/{set}/images")
	makedirs(f"{DATASET_YOLO_DIR}/{set}/labels")

for period in periods:
	set = period['set']

	for name in period['images']:
		label = open(f"{DATASET_YOLO_DIR}/{set}/labels/{name}.txt", 'w')

		for anno in annotations[name]:
			coords = coord_pack_float64(anno['polygon'])

			label.write('0')
			for coord in coords:
				x = coord[0] / CONFIG_IMAGE_DIMENSION[0]
				y = coord[1] / CONFIG_IMAGE_DIMENSION[1]

				label.write(f" {x:.12f} {y:.12f}")

			label.write('\n')

		link(f"{DATASET_SHARED_IMAGE_DIR}/{name}.png",
		     f"{DATASET_YOLO_DIR}/{set}/images/{name}.png")

metadata_src = open(f"{DATASET_YOLO_DIR}/dataset.yaml", 'w')
yaml_dump({
	'train': 'train/images',
	'val': 'validate/images',
	'test': 'test/images',
	'names': {
		0: 'filament',
	},
}, metadata_src, indent=3, sort_keys=False)
