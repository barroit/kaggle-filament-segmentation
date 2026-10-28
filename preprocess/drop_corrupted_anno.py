# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#
# Corrupted annotations were identified using scripts/find-duplicate.py.
#
# 010403-20160810234034Lh:
# 	GT=1:	id=d38060a1-d7c7-459a-8048-abeb4d6fa7c4	(bad)
# 	GT=2:	id=066551fe-8fe4-4678-bbeb-06782e5b3416
# 010403-20150228082414Th:
# 	GT=9:	id=1ad99d93-01a2-4717-8f94-a938a22080eb
# 	GT=11:	id=bfed9608-8ed0-4bdb-850b-fce9f8ae5a84	(bad)
# 050202-20120523163014Mh:
# 	GT=5:	id=21fc0657-5d90-40ac-b955-38bb51c54038
# 	GT=6:	id=f1167b5f-4c20-4a10-a64f-58b3c25eeebe	(bad)
#

from json import dump as json_dump, load as json_load

from lib.path import *

bad = (
	'd38060a1-d7c7-459a-8048-abeb4d6fa7c4',
	'bfed9608-8ed0-4bdb-850b-fce9f8ae5a84',
	'f1167b5f-4c20-4a10-a64f-58b3c25eeebe',

	# There are also some poor quality annotations with overly aggressive
	# polygons.
	'f8e08c38-d36a-4d75-aa83-e328a2f4e873',
	'2309b8f3-ea38-4a0e-931c-a657433c391a',
)

src = open(DATASET_SHARED_ANNOTATIONS)
annotations = json_load(src)

periods_src = open(DATASET_SHARED_PERIODS)
periods = json_load(periods_src)
periods_dst = open(DATASET_SHARED_PERIODS, 'w+')

for period in periods:
	images = len(period['images'])

	for idx in range(images - 1, -1, -1):
		name = period['images'][idx]
		annos = annotations[name]

		for anno in annos[:]:
			if anno['id'] in bad:
				annos.remove(anno)

		period['annos'][idx] = len(annos)

		if not period['annos'][idx]:
			period['images'].pop(idx)
			period['annos'].pop(idx)

json_dump(periods, periods_dst, separators=(',', ':'))
