#!/usr/bin/env python
# coding=utf-8

from lifxlan import LifxLAN

status = "Off"

def turnOn():
	lifxlan = LifxLAN()

	lifxlan.set_power_all_lights("on", rapid=True)
	status = "On"
	
def isOn():
	if status == "On":
		return True 

