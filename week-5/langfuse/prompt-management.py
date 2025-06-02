from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.openai import openai

load_dotenv()

# Initialize LangFuse
langfuse = Langfuse()

# 1. Register a prompt
langfuse.create_prompt(
    name="joke-prompt",
    prompt="Tell me a funny joke about {{topic}}",
    config={"model": "gpt-4.1", "temperature": 0.8},
    labels=["production"],
)

# 2. Load the prompt
prompt = langfuse.get_prompt("joke-prompt")


# 3. Use the prompt with OpenAI responses API
def generate_joke(topic: str) -> str:
    compiled_prompt = prompt.compile(topic=topic)

    response = openai.responses.create(
        model=prompt.config["model"],
        input=[{"role": "user", "content": compiled_prompt}],
        temperature=prompt.config["temperature"],
        langfuse_prompt=prompt,
    )

    return response.output_text


# Example usage
if __name__ == "__main__":
    joke = generate_joke("artificial intelligence")
    print(f"Generated joke: {joke}")
