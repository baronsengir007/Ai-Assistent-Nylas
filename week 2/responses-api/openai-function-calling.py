from dotenv import load_dotenv
from openai import OpenAI
import json


load_dotenv()

client = OpenAI()

"""
https://platform.openai.com/docs/guides/function-calling
"""

# --------------------------------------------------------------
# Define the function(s)
# --------------------------------------------------------------


def send_email(to, subject, body):
    # In a real application, this would send an actual email
    print(f"Email sent to: {to}")
    print(f"Subject: {subject}")
    print(f"Body: {body}")
    return f"Email sent to {to} successfully!"


def call_function(name, args):
    if name == "send_email":
        return send_email(**args)
    # Add other functions as needed
    return "Function not found"


# --------------------------------------------------------------
# Create a tool specification for the function(s)
# --------------------------------------------------------------

tools = [
    {
        "type": "function",
        "name": "send_email",
        "description": "Send an email to a given recipient with a subject and message.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "The recipient email address."},
                "subject": {"type": "string", "description": "Email subject line."},
                "body": {"type": "string", "description": "Body of the email message."},
            },
            "required": ["to", "subject", "body"],
            "additionalProperties": False,
        },
    }
]

# --------------------------------------------------------------
# Initial prompt
# --------------------------------------------------------------


messages = [
    {
        "role": "user",
        "content": "Can you send an email to ilan@example.com and katia@example.com saying hi?",
    }
]

response = client.responses.create(
    model="gpt-4.1",
    input=messages,
    tools=tools,
)

print("Initial response:")
print(response.output)

# --------------------------------------------------------------
# Process tool calls
# --------------------------------------------------------------


# Process tool calls
for tool_call in response.output:
    if tool_call.type != "function_call":
        continue

    name = tool_call.name
    args = json.loads(tool_call.arguments)

    # Add the tool call to the messages
    messages.append(tool_call)

    # Call the function
    result = call_function(name, args)

    # Add the result to the messages
    messages.append(
        {
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(result),
        }
    )

# --------------------------------------------------------------
# Get the final response
# --------------------------------------------------------------

response_2 = client.responses.create(
    model="gpt-4.1",
    input=messages,
    tools=tools,
)

print("\nFinal response:")
print(response_2.output_text)
