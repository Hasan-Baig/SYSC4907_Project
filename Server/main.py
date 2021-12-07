import hivemq_subscribe as mqtt
import firebase

client = None
session = None
gestures = None


def main():
    # Display the default message
    intro_message()

    # Setup code
    setup()

    # Run the event loop
    start()

    # Shutdown everything
    shutdown()


def intro_message():
    print("----------------------------------------------------")
    print("This is the hand gesture detection server program.")
    print("This program was designed by __names__here__...")
    print("----------------------------------------------------\n\n")


def setup():
    # Connect to firebase
    global session
    session = firebase.setup_firebase()

    # Read the realtime database
    global gestures
    gestures = firebase.read_gestures_from_db(session)
    print(gestures)

    # Setup the mqtt subscribe stuff
    global client
    client = mqtt.setup()
    mqtt.subscribe(client)


def start():
    # Start the main loop
    input("Enter anything to have the program exit...\n")


def shutdown():
    print("Shutting down.... goodbye")

    global client
    mqtt.stop(client)


if __name__ == '__main__':
    main()
