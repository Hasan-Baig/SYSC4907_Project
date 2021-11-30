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

# setting callbacks for different events to see if it works, print the message etc.
def on_connect(client, userdata, flags, rc, properties=None):
    print("CONNACK received with code %s." % rc)

# with this callback you can see if your publish was successful
def on_publish(client, userdata, mid, properties=None):
    print("mid: " + str(mid))

# print which topic was subscribed to
def on_subscribe(client, userdata, mid, granted_qos, properties=None):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))

# print message, useful for checking if it was successful
def on_message(client, userdata, msg: paho.MQTTMessage):
    print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload, "utf-8"))
    print("Type: " + str(type(msg.payload)))

    # Analyze the message
    if msg.payload.isdigit():
        analyze_msg(int(msg.payload))

# Stub that prints out different gesture actions
def analyze_msg(message):
    if (message == 1):
        print("Turn on the lights...")
    elif (message == 2):
        print("Unlock the door...")
    elif (message == 3):
        print("Roll down the blinds...")
    elif (message == 4):
        print("Start the vacuum...")
    elif (message == 5):
        print("Make some coffee...")
    else:
        print("No action assigned.")


def setup() -> paho.Client:
    # using MQTT version 5 here, for 3.1.1: MQTTv311, 3.1: MQTTv31
    # userdata is user defined data of any type, updated by user_data_set()
    # client_id is the given name of the client
    client = paho.Client(client_id="", userdata=None, protocol=paho.MQTTv5)
    client.on_connect = on_connect

    # enable TLS for secure connection
    client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
    # set username and password
    client.username_pw_set(mqtt_constants.mqtt_username,
                           mqtt_constants.mqtt_password)
    # connect to HiveMQ Cloud on port 8883 (default for MQTT)
    client.connect(mqtt_constants.mqtt_host, mqtt_constants.mqtt_port)

    # setting callbacks, use separate functions like above for better visibility
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.on_publish = on_publish

    # Start a background loop to handle incoming messages and reconnections
    client.loop_start()

    return client


def subscribe(client: paho.Client):
    # subscribe to all topics of encyclopedia by using the wildcard "#"
    client.subscribe(mqtt_constants.mqtt_topic_subscribe, qos=2)


def stop(client: paho.Client):
    client.loop_stop()
