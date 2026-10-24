# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from json import dump as json_dump, load as json_load

from lib.date import *
from lib.kconfig import *
from lib.path import *

src = open(DATASET_SHARED_ANNOTATIONS)
annotations = json_load(src)
dst = open(DATASET_SHARED_PERIODS, 'w+')

days = list()
periods = list()
start = 0

for name in annotations:
	days.append((name[:8], name))

days.sort(key=lambda item: item[0])
start = date_from_str(days[0][0])

periods.append({
	'idx': 0,
	'annos': [],
	'images': [],
})

for date, name in days:
	current = date_from_str(date)
	span = current - start

	idx = span.days // CONFIG_OBSERVE_PERIOD
	annos = len(annotations[name])

	if periods[-1]['idx'] != idx:
		periods.append({
			'idx': idx,
			'annos': [],
			'images': [],
		})

	periods[-1]['annos'].append(annos)
	periods[-1]['images'].append(name)

json_dump(periods, dst, separators=(',', ':'))
