import hivemq_subscribe as mqtt

client = None


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
