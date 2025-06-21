import requests

SERVER_IP_ADDRESS = "167.235.58.55"
url = f"http://{SERVER_IP_ADDRESS}:8080/events"

data = {
    "from_email": "sarah.smith@example.com",
    "to_email": "support@techgear.com",
    "sender": "Sarah Smith",
    "subject": "Compatibility question",
    "body": "Hi TechGear support, I'm considering buying your new TechGear SmartHome Hub, but I'm not sure if it's compatible with my existing smart devices. I have Philips Hue lights, a Nest thermostat, and Amazon Echo devices. Can you confirm if the SmartHome Hub will work with these? Thanks in advance for your help!",
}


def send_event():
    """Send event to the API endpoint for processing."""

    response = requests.post(url, json=data)

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")

    assert response.status_code == 202


if __name__ == "__main__":
    send_event()
