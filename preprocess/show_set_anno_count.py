# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from json import load as json_load

from lib.path import *

src = open(DATASET_SHARED_PERIODS)
periods = json_load(src)

train_count = 0
validate_count = 0
test_count = 0
total = 0

for period in periods:
	count = sum(period['annos'])

	if period['set'] == 'train':
		train_count += count
	elif period['set'] == 'validate':
		validate_count += count
	else:
		test_count += count

total = train_count + validate_count + test_count

print(f"    Train\t{train_count}\t({train_count / total * 100:.2f}%)")
print(f"    Validate\t{validate_count}\t({validate_count / total * 100:.2f}%)")
print(f"    Test\t{test_count}\t({test_count / total * 100:.2f}%)")
