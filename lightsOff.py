#!/usr/bin/env python
# coding=utf-8

from lifxlan import LifxLAN

status = "Off"

def turnOff():
	lifxlan = LifxLAN()

	lifxlan.set_power_all_lights("off", rapid=True)
	status = "Off"
def isOn():
	if status == "On":
		return True 
	elif status == "Off":
		return False 
