from langfuse.decorators import observe
from langfuse.openai import openai
from dotenv import load_dotenv


"""
Make sure you have set the environment variables in the .env file:
LANGFUSE_SECRET_KEY=your-langfuse-api-secret-here
LANGFUSE_PUBLIC_KEY=your-langfuse-api-key-here
LANGFUSE_HOST=your-langfuse-host-here
"""

load_dotenv()

# --------------------------------------------------------------
# Using the observe decorator to log the LLM calls
# --------------------------------------------------------------


@observe()
def story():
    return (
        openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a great storyteller."},
                {
                    "role": "user",
                    "content": "Once upon a time in a galaxy far, far away...",
                },
            ],
        )
        .choices[0]
        .message.content
    )


# --------------------------------------------------------------
# Run the main function
# --------------------------------------------------------------


@observe()
def main():
    return story()


main()

# --------------------------------------------------------------
# Check the Langfuse dashboard to see the logged LLM calls
# --------------------------------------------------------------

"""
You can see the logged LLM calls in the Langfuse dashboard:
https://us.cloud.langfuse.com/ or https://eu.cloud.langfuse.com/
"""
