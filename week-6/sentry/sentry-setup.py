import os
import sentry_sdk
from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn

"""
Sentry Error Tracking & Performance Monitoring Setup

First, go to https://sentry.io and create a free account to get your Sentry DSN.

1. Create a new project in Sentry (choose Python)
2. Copy your DSN from the project settings
3. Add to your .env file:

SENTRY_DSN=https://your-dsn-key@your-org.ingest.sentry.io/your-project-id
"""

load_dotenv()

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    send_default_pii=True,
)

app = FastAPI()


@app.get("/sentry-debug")
async def trigger_error():
    division_by_zero = 1 / 0
    return division_by_zero


@app.get("/key-error")
async def trigger_key_error():
    data = {"name": "John", "age": 30}
    missing_key = data["email"]
    return missing_key


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
When you open http://localhost:8000/sentry-debug/ or http://localhost:8000/key-error/ with your browser, a transaction in the Performance section of Sentry will be created.
Additionally, an error event will be sent to Sentry and will be connected to the transaction.
It takes a couple of moments for the data to appear in Sentry.
"""
