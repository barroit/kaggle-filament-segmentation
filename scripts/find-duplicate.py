#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from collections import defaultdict

from pycocotools.coco import COCO

from lib.path import *

coco = COCO(MAGFILO_METADATA_FILE)
duplicates = 0

for id, annos in coco.imgToAnns.items():
	spines = defaultdict(list)

	for idx, anno in enumerate(annos):
		spine = tuple(anno['spine'])

		spines[spine].append((idx, anno))

	for matches in spines.values():
		if len(matches) < 2:
			continue

		print(f"{id}:")

		for idx, anno in matches:
			print(f"\tGT={idx + 1}: id={anno['id']}")

		duplicates += 1

print(f"Duplicate spine groups: {duplicates}")
