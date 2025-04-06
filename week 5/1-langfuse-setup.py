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


@observe()
def main():
    return story()


main()