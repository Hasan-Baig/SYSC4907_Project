#
# Copyright 2021 HiveMQ GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import paho.mqtt.client as paho
from paho import mqtt
import mqtt_constants

class hive_mq_client:
    
    def __init__(self, get_gestures):
        self.get_gestures = get_gestures
        
        # using MQTT version 5 here, for 3.1.1: MQTTv311, 3.1: MQTTv31
        # userdata is user defined data of any type, updated by user_data_set()
        # client_id is the given name of the client
        self.client = paho.Client(client_id="", userdata=None, protocol=paho.MQTTv5)
        self.client.on_connect = self.on_connect

        # enable TLS for secure connection
        self.client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
        # set username and password
        self.client.username_pw_set(mqtt_constants.mqtt_username,
                            mqtt_constants.mqtt_password)
        # connect to HiveMQ Cloud on port 8883 (default for MQTT)
        self.client.connect(mqtt_constants.mqtt_host, mqtt_constants.mqtt_port)

        # setting callbacks, use separate functions like above for better visibility
        self.client.on_subscribe = self.on_subscribe
        self.client.on_message = self.on_message
        self.client.on_publish = self.on_publish

        # Start a background loop to handle incoming messages and reconnections
        self.client.loop_start()
    
    # setting callbacks for different events to see if it works, print the message etc.
    def on_connect(self, client, userdata, flags, rc, properties=None):
        print("CONNACK received with code %s." % rc)

    # with this callback you can see if your publish was successful
    def on_publish(self, client, userdata, mid, properties=None):
        print("mid: " + str(mid))

    # print which topic was subscribed to
    def on_subscribe(self, client, userdata, mid, granted_qos, properties=None):
        print("Subscribed: " + str(mid) + " " + str(granted_qos))

    # print message, useful for checking if it was successful
    def on_message(self, client, userdata, msg: paho.MQTTMessage):
        print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload, "utf-8"))
        print("Type: " + str(type(msg.payload)))

        # Analyze the message
        if msg.payload.isdigit():
            self.analyze_msg(int(msg.payload))

    # Stub that prints out different gesture actions
    def analyze_msg(self, message):
        gestures = self.get_gestures()
        
        if len(gestures) > message:
            print("Activating " + gestures[message]['name'])
            print("Connecting to... " + gestures[message]["ip"])


    def subscribe(self):
        # subscribe to all topics of encyclopedia by using the wildcard "#"
        self.client.subscribe(mqtt_constants.mqtt_topic_subscribe, qos=2)


    def stop(self):
        self.client.loop_stop()
