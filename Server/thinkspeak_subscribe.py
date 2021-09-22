import constants
import paho.mqtt.client as mqtt
import time

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(constants.mqtt_topic)
    print("Subscribed to topic")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, message):
    print("message received " ,str(message.payload.decode("utf-8")))
    print("message topic=",message.topic)
    print("message qos=",message.qos)
    print("message retain flag=",message.retain)

# The client itself
client = mqtt.Client(client_id=constants.mqtt_clientId)
#client.on_connect = on_connect
client.on_message = on_message

client.username_pw_set(
	username=constants.mqtt_username,
	password=constants.mqtt_password
)
client.connect(constants.mqtt_host, 1883, 60)

client.loop_start()
client.subscribe(constants.mqtt_topic)
client.publish(constants.mqtt_topic, 15)
time.sleep(10)
client.loop_stop()
