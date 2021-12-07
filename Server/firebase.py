import requests
import time
from getpass import getpass

API_KEY = "AIzaSyAx-h8OoQMpsBRplnKzmM0gwCpZpmcl2pk"


class Session:

    def __init__(self, email, id_token, refresh_token, expires_in):
        self.email = email
        self.id_token = id_token
        self.refresh_token = refresh_token
        self.expiration_time = time.time() + expires_in

    def is_token_expired(self) -> bool:
        return time.time() > self.expiration_time


def signin(email: str, password: str) -> Session:
    url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=" + API_KEY

    body = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }

    response = requests.post(
        url=url,
        data=body
    )

    if (response.status_code != 200):
        error_msg = "Error code " + str(response.status_code)

        # Check if the response is json to add a more detailed message
        if response.headers.get('content-type').__contains__('application/json'):
            error_msg += ", message: " + response.json()['error']['message']
        else:
            error_msg += ", message: " + response.text

        # Some error occured
        raise ValueError(error_msg)

    jsonResponse = response.json()
    session = Session(
        email=email,
        id_token=jsonResponse['idToken'],
        refresh_token=jsonResponse['refreshToken'],
        expires_in=int(jsonResponse['expiresIn'])
    )

    return session


def refresh_token(session: Session):
    url = "https://securetoken.googleapis.com/v1/token?key=" + API_KEY

    body = {
        "grant_type": "refresh_token",
        "refresh_token": session.refresh_token
    }

    response = requests.post(url=url, data=body)
    jsonResponse = response.json()

    session.id_token = jsonResponse['id_token']


def make_request(url, session: Session, method="GET", body={}, try_again=True):
    if session.is_token_expired():
        refresh_token(session)

    response = requests.request(
        method=method,
        url=url,
        data=body
    )

    return response


def setup_firebase():
    session = None

    while session == None:
        # Get input from user
        email = input("Enter the email: ")
        password = getpass()

        # Create the session
        try:
            session = signin(email, password)
            print("Logged into firebase account.")

            return session
        except BaseException as e:
            print("Log in failed, try again: \"" + e.args[0] + "\"")


def read_gestures_from_db(session: Session):
    url = "https://smarthome-4feea-default-rtdb.firebaseio.com/gestures/user:" + \
        session.email.replace(".", "") + ".json"
    print(url)

    response = make_request(url, session)
    return response.json()
