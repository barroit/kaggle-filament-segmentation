# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from json import dump as json_dump, load as json_load

from lib.date import *
from lib.kconfig import *
from lib.path import *

src = open(DATASET_SHARED_PERIODS)
periods = json_load(src)
dst = open(DATASET_SHARED_PERIODS, 'w+')

days = list()
scores = list()
parents = list()

selected = set()
excluded = dict()

idx_next = -1
score_best = 1

days.append({
	'date': date_from_str(periods[0]['images'][0]),
	'set': periods[0]['set'],
	'annos': 0,
})

for period in periods:
	for name, count in zip(period['images'], period['annos']):
		date = date_from_str(name[:8])

		if days[-1]['date'] == date:
			days[-1]['annos'] += count
		else:
			days.append({
				'date': date,
				'set': period['set'],
				'annos': count,
			})

for idx_next, day in enumerate(days):
	best = day['annos']
	parent = -1

	for idx_prev in range(idx_next):
		prev = days[idx_prev]
		span = day['date'] - prev['date']
		gap = span.days - 1

		if prev['set'] == day['set'] or \
		   gap >= CONFIG_SPLIT_TEMPORAL_GAP:
			score = scores[idx_prev] + day['annos']

			if score > best:
				best = score
				parent = idx_prev

	scores.append(best)
	parents.append(parent)

score_best = max(scores)
idx_next = scores.index(score_best)

while idx_next != -1:
	selected.add(days[idx_next]['date'])
	idx_next = parents[idx_next]

for period in periods[:]:
	images = len(period['images'])

	for idx in range(images - 1, -1, -1):
		name = period['images'][idx]
		date = date_from_str(name[:8])

		if date not in selected:
			period['images'].pop(idx)
			excluded[name] = period['annos'].pop(idx)

	if not period['images']:
		periods.remove(period)

json_dump(periods, dst, separators=(',', ':'))
