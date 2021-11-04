import constants
import paho.mqtt.client as mqtt

# Get the gesture from the user. This is what will be sent over mqtt.
payload = input("What gesture to transmit (1-5)?: ")

# Set up the client
client = mqtt.Client(client_id=constants.mqtt_clientId)
client.username_pw_set(
    username=constants.mqtt_username,
    password=constants.mqtt_password
)
client.tls_set(tls_version=mqtt.ssl.PROTOCOL_TLS)
client.connect(constants.mqtt_host, 8883, 60)

# Publish the message
message_info = client.publish(
    topic=constants.mqtt_topic_publish,
    payload=payload,
    qos=2
)

# Wait until the message is published
while not message_info.is_published():
    client.loop(1)

print("Is the message published?: " + str(message_info.is_published()))
