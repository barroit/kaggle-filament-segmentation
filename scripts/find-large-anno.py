#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from json import load as json_load
from sys import argv

from lib.path import *

anno_src = open(DATASET_SHARED_ANNOTATIONS)
annos = json_load(anno_src)

period_src = open(DATASET_SHARED_PERIODS)
periods = json_load(period_src)
period_map = {}

max_dim = float(argv[1])
found = 0

for period in periods:
	for image in period['images']:
		period_map[image] = period['idx']

for image, items in annos.items():
	for anno in items:
		*_, width, height = anno['bbox']
		dim = None
		val = None

		if width > max_dim:
			dim = 'width'
			val = width

		if height > max_dim:
			dim = 'height'
			val = height

		if dim:
			print(f"{anno['id']}  ({image})")
			print(f"\t{dim}\t{val:.3f}")
			print(f"\tperiod\t{period_map[image]}")
			print()

			found += 1

print(f"found {found} annotation{'s' if found > 1 else ''}")
