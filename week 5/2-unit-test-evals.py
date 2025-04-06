import os
import instructor
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv

load_dotenv("../.env")


# --------------------------------------------------------------
# Initialize the OpenAI client with Instructor
# --------------------------------------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = instructor.patch(client)


# --------------------------------------------------------------
# Define the CustomerInquiry model
# --------------------------------------------------------------


class CustomerInquiry(BaseModel):
    category: Literal["question", "complaint", "feature_request", "billing", "other"]
    response: str


# --------------------------------------------------------------
# Define the process_customer_message function
# --------------------------------------------------------------


def process_customer_message(message: str) -> CustomerInquiry:
    inquiry = client.chat.completions.create(
        model="gpt-4o",
        response_model=CustomerInquiry,
        messages=[
            {
                "role": "system",
                "content": "You are a customer service AI that analyzes customer inquiries.",
            },
            {"role": "user", "content": message},
        ],
    )

    return inquiry


# --------------------------------------------------------------
# Define the test messages
# --------------------------------------------------------------


billing_message = "I've been charged twice for my subscription this month. Please help!"
feature_request_message = (
    "When will you add dark mode to the mobile app? It's really needed."
)
question_message = "How do I reset my password? I've been locked out of my account."


# --------------------------------------------------------------
# Run the tests to check all categories
# --------------------------------------------------------------


def test_process_customer_message():
    assert process_customer_message(billing_message).category == "billing"
    print("Passed billing test")
    assert (
        process_customer_message(feature_request_message).category == "feature_request"
    )
    print("Passed feature request test")
    assert process_customer_message(question_message).category == "question"
    print("Passed question test")


if __name__ == "__main__":
    test_process_customer_message()
