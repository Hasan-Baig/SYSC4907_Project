import constants
import paho.mqtt.client as mqtt
import time


def on_connect(client: mqtt.Client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    response = client.subscribe(constants.mqtt_topic_subscribe)
    print("Subscribed to topic ")
    print(response)


def on_subscribe(client, userdata, mid, granted_qos, properties=None):
    print("Subscribed")
    print(properties)


def on_message(client, userdata, message):
    print("Got something")
    # The callback for when a PUBLISH message is received from the server.
    print("message received ", str(message.payload.decode("utf-8")))
    print("message topic=", message.topic)
    print("message qos=", message.qos)
    print("message retain flag=", message.retain)

    # Analyze the message
    analyze_msg(message.payload)


def on_disconnect(userdata, rc, properties):
    print("Disconnected...")


def analyze_msg(message):
    if (message == "1"):
        do_something_1()
    elif (message == "2"):
        do_something_2()
    else:
        print("No action assigned.")


def do_something_1():
    # These are all stubs
    print("Lights on")


def do_something_2():
    print("Blinds roll down...")


# The client itself
client = mqtt.Client(client_id=constants.mqtt_clientId)
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect
client.on_subscribe = on_subscribe

client.username_pw_set(
    username=constants.mqtt_username,
    password=constants.mqtt_password
)
client.connect(constants.mqtt_host, 1883, 60)

# client.loop_start()
client.loop_forever()
# Uncomment if you want to have the loop end by itself
# time.sleep(10)
# client.loop_stop()
