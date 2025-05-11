from typing import List, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI()

"""
https://platform.openai.com/docs/guides/structured-outputs/
"""

# --------------------------------------------------------------
# Simple Example
# --------------------------------------------------------------


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


response = client.responses.parse(
    model="gpt-4o-mini",
    input=[
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        },
    ],
    text_format=CalendarEvent,
)

event = response.output_parsed
print(event.model_dump_json(indent=2))

# --------------------------------------------------------------
# Example 2: Complex Example
# --------------------------------------------------------------


class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    key_phrases: List[str]
    escalate: bool = Field(description="Whether to escalate the ticket to a manager.")


response = client.responses.parse(
    model="gpt-4o-mini",
    input=[
        {"role": "system", "content": "Analyze the sentiment of the following text."},
        {"role": "user", "content": "I'm very happy with the service."},
    ],
    text_format=SentimentAnalysis,
)

sentiment = response.output_parsed
print(sentiment.model_dump_json(indent=2))
