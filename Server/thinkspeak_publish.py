import constants
import paho.mqtt.client as mqtt

# payload = input("What do you want to transmit?: ")

client = mqtt.Client(client_id=constants.mqtt_clientId)
client.username_pw_set(
    username=constants.mqtt_username,
    password=constants.mqtt_password
)
client.connect(constants.mqtt_host, 1883, 60)
message_info = client.publish(
    topic=constants.mqtt_topic_publish,
    payload=12,
    qos=0
)
print(message_info.is_published())
