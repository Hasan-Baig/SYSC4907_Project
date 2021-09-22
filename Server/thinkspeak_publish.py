import constants
import paho.mqtt.client as mqtt

client = mqtt.Client(client_id=constants.mqtt_clientId)
client.username_pw_set(
	username=constants.mqtt_username,
	password=constants.mqtt_password
)
client.connect(constants.mqtt_host, 1883, 60)
client.publish(
	topic=constants.mqtt_topic,
	payload=12
) 
