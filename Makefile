# SPDX-License-Identifier: GPL-3.0-or-later

.PHONY: menuconfig.preprocess menuconfig.evaluation menuconfig.runtime \
	menuconfig.preview alldefconfig

KCONFIG_ENV := MENUCONFIG_STYLE=aquatic KCONFIG_FUNCTIONS=kconfig \
	       PYTHONPATH=$(CURDIR)/scripts PYTHONDONTWRITEBYTECODE=1

ALLDEFCONFIG := $(KCONFIG_ENV) ./kconfiglib/alldefconfig.py
MENUCONFIG := $(KCONFIG_ENV) ./kconfiglib/menuconfig.py

alldefconfig: kconfig/gen/preprocess kconfig/gen/evaluation \
	      kconfig/gen/runtime kconfig/gen/preview

menuconfig.preprocess \
menuconfig.evaluation \
menuconfig.runtime \
menuconfig.preview: menuconfig.%: kconfig/gen/%
	KCONFIG_CONFIG=$< $(MENUCONFIG) $(subst gen/,,$<)

kconfig/gen/%:
	mkdir -p $(@D)
	KCONFIG_CONFIG=$@ $(ALLDEFCONFIG) $(subst gen/,,$@)
