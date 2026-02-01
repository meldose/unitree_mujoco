import os
from dotenv import load_dotenv
from twilio_client import client, FROM_NUMBER, TO_NUMBER

load_dotenv()
VOICE_URL = os.getenv("TWILIO_VOICE_URL")

if not VOICE_URL:
    raise RuntimeError("TWILIO_VOICE_URL not set")

call = client.calls.create(
    to=TO_NUMBER,
    from_=FROM_NUMBER,
    url=VOICE_URL
)

print("Call SID:", call.sid)

