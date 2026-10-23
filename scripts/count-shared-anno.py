#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from functools import reduce
from json import load as json_load

from lib.path import *

src = open(DATASET_SHARED_ANNOTATIONS)
metadata = json_load(src)

annos_list = metadata.values()
count = reduce(lambda accume, annos: accume + len(annos), annos_list, 0)

print(count)
