# SPDX-License-Identifier: GPL-3.0-or-later

.PHONY: menuconfig

menuconfig:
	MENUCONFIG_STYLE=aquatic KCONFIG_FUNCTIONS=kconfig \
	PYTHONPATH=$(CURDIR)/scripts PYTHONDONTWRITEBYTECODE=1 \
	./kconfiglib/menuconfig.py
