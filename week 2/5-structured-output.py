import os
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, Optional
from dotenv import load_dotenv

load_dotenv("../.env")

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Patch the client with Instructor
client = instructor.patch(client)


class CustomerInquiry(BaseModel):
    """Model representing a classified customer inquiry."""

    inquiry_type: Literal[
        "question", "complaint", "feature_request", "billing", "other"
    ] = Field(description="The category of the customer's inquiry")

    product: Optional[str] = Field(
        default=None, description="The product or service mentioned in the inquiry"
    )

    confidence: float = Field(
        description="Confidence score for the classification (0.0-1.0)", ge=0.0, le=1.0
    )

    priority: Literal["low", "medium", "high", "urgent"] = Field(
        description="The priority level of this inquiry"
    )

    summary: str = Field(description="A brief summary of the customer's inquiry")

    def get_handler(self):
        """Return the appropriate handler based on inquiry type and priority."""
        if self.inquiry_type == "complaint" and self.priority in ["high", "urgent"]:
            return "escalation_team"
        elif self.inquiry_type == "billing":
            return "billing_department"
        elif self.inquiry_type == "feature_request":
            return "product_team"
        elif self.confidence < 0.7:
            return "human_review"
        else:
            return "standard_support"


def process_customer_message(message: str) -> CustomerInquiry:
    """Process a customer message and return structured information.

    Args:
        message: The customer's message text

    Returns:
        A structured CustomerInquiry object
    """
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


def route_inquiry(inquiry: CustomerInquiry):
    """Route the inquiry to the appropriate handler.

    Args:
        inquiry: The structured customer inquiry
    """
    handler = inquiry.get_handler()
    print(f"Routing inquiry to: {handler}")
    print(f"Inquiry type: {inquiry.inquiry_type}")
    print(f"Priority: {inquiry.priority}")
    print(f"Summary: {inquiry.summary}")
    print(f"Confidence: {inquiry.confidence:.2f}")

    # In a real application, you would call different functions based on the handler
    # handlers = {
    #     "escalation_team": handle_escalation,
    #     "billing_department": handle_billing,
    #     # ...
    # }
    # handlers[handler](inquiry)


# Example usage
if __name__ == "__main__":
    # Example customer messages
    messages = [
        "I've been charged twice for my subscription this month. Please help!",
        "When will you add dark mode to the mobile app? It's really needed.",
        "How do I reset my password? I've been locked out of my account.",
    ]

    # Process each message
    for message in messages:
        print("\n" + "=" * 50)
        print(f"Customer message: {message}")
        inquiry = process_customer_message(message)
        route_inquiry(inquiry)
