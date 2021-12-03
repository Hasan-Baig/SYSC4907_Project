#!/usr/bin/env python
# coding=utf-8

from lifxlan import LifxLAN
from lightsOn import turnOn
from lightsOn import isOn
from rainbow import run_rainbow
from lightsOff import turnOff

led = turnOn()
rainbow = run_rainbow()
led = turnOff()
