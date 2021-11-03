from time import sleep
import paho.mqtt.client as mqtt

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client: mqtt.Client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(topic = "channels/1501637/subscribe/fields/+", qos=1)

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))


client = mqtt.Client(client_id="NR0UNC46ARMQMAI8LgU5HQA")
client.on_connect = on_connect
client.on_message = on_message

client.username_pw_set(
    username="NR0UNC46ARMQMAI8LgU5HQA",
    password="LETPZ3cjLhIu5tHmydm9MNux"
)

client.connect("mqtt3.thingspeak.com", 1883, 60)

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
client.loop_forever()

# i = 1
# while True:
#     print("Next iteration " + str(i))
#     client.loop()
#     sleep(1)
#     i += 1
