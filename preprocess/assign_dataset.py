# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from functools import reduce
from json import dump as json_dump, load as json_load
from random import Random as alloc_prng

from lib.kconfig import *
from lib.path import *

def accum_annos(accum, period):
	annos = sum(period['annos'])

	return accum + annos

src = open(DATASET_SHARED_PERIODS)
periods = json_load(src)
annos = reduce(accum_annos, periods, 0)
dst = open(DATASET_SHARED_PERIODS, 'w+')

train_annos = annos * CONFIG_TRAIN_SET_PROPORTION // 100
validate_annos = annos * CONFIG_VALIDATE_SET_PROPORTION // 100
test_annos = annos * CONFIG_TEST_SET_PROPORTION // 100
annos_left = annos - (train_annos + validate_annos + test_annos)

train_count = 0
validate_count = 0
test_count = 0

prng = alloc_prng(CONFIG_SPLIT_PRNG_SEED)

train_annos += annos_left
prng.shuffle(periods)

for period in periods:
	train_left = train_annos - train_count
	validate_left = validate_annos - validate_count
	test_left = test_annos - test_count
	next = max(train_left, validate_left, test_left)

	if next == train_left:
		period['set'] = 'train'
		train_count += sum(period['annos'])

	elif next == validate_left:
		period['set'] = 'validate'
		validate_count += sum(period['annos'])

	else:
		period['set'] = 'test'
		test_count += sum(period['annos'])

periods.sort(key=lambda periods: periods['idx'])
json_dump(periods, dst, separators=(',', ':'))
