# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from datetime import datetime

__all__ = [
	'date_from_str',
]

def date_from_str(str):
	return datetime.strptime(str[:8], '%Y%m%d')
