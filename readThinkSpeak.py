
import urllib.request
import json
import threading
import requests
from tempSensor import*

def read_data_thingspeak():
    URL= 'https://api.thingspeak.com/channels/1559793/feeds.json?results=1' #Only want 1 result per instruction

    KEY='SXGGNBP0XIU0WENQ '
    HEADER='&results=1'
    NEW_URL=URL+KEY+HEADER
    print(NEW_URL)

    get_data=requests.get(NEW_URL).json()
    #print(get_data)
    channel_id=get_data['channel']['id']

    feild_1=get_data['feeds']
    #print(feild_1)

    i=[]
    for x in feild_1:
        #print(x['field1'])
        i.append(x['field1'])
    print(i)
    if x['field1'] == '2':
        	print("Turn on temperature sensor")
        	sense_temp()
        	
        
if __name__ == '__main__':
    read_data_thingspeak()
