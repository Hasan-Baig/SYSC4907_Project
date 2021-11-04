import http.client
import urllib.request
import time
key = "NCS6Z397AZ7QPCXJ"  # Put your API Key here
def publish():
    while True:
        
        instruct = 2
        params = urllib.parse.urlencode({'field1': instruct, 'key':key }) 
        headers = {"Content-typZZe": "application/x-www-form-urlencoded","Accept": "text/plain"}
        conn = http.client.HTTPConnection("api.thingspeak.com:80")
        try:
            conn.request("POST", "/update", params, headers)
            response = conn.getresponse()
            print (instruct)
            print (response.status, response.reason)
            data = response.read()
            conn.close()
        except:
            print("connection failed")
        break
if __name__ == "__main__":
        while True:
                publish()
                #time.wait(5)
