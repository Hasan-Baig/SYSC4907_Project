from sense_emu import SenseHat
import time

def sense_temp():
	while True:
		sense = SenseHat()
		temp = sense.get_temperature() #get teperature reading from senshat 
		print("Temp: %s C" % temp)               

		sense.set_rotation(180)        #move LEDS 

		sense.show_message("%.1f C" % temp, scroll_speed=0.10, text_colour=[0, 255, 0])
		time.sleep(10)