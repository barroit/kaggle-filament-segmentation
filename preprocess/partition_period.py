# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from datetime import datetime
from json import dump as json_dump, load as json_load

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
start = datetime.strptime(days[0][0], '%Y%m%d')

periods.append({
	'idx': 0,
	'annos': 0,
	'images': [],
})

for date, name in days:
	current = datetime.strptime(date, '%Y%m%d')
	span = current - start
	idx = span.days // CONFIG_OBSERVE_PERIOD

	if periods[-1]['idx'] != idx:
		periods.append({
			'idx': idx,
			'annos': 0,
			'images': [],
		})

	periods[-1]['annos'] += len(annotations[name])
	periods[-1]['images'].append(name)

json_dump(periods, dst, separators=(',', ':'))
