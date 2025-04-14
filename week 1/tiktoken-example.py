import tiktoken

"""
This script demonstrates how to count tokens in text using the tiktoken library.
"""

# --------------------------------------------------------------
# Define a function to count tokens
# --------------------------------------------------------------


def count_tokens(text, model="gpt-4o"):  # Get the encoding for the specified model
    encoding = tiktoken.encoding_for_model(model)

    # Count tokens
    token_count = len(encoding.encode(text))

    print(f"Token count: {token_count}")

    return token_count


# --------------------------------------------------------------
# Count tokens for a simple text
# --------------------------------------------------------------

text = "Hello, this is a simple example of counting tokens with tiktoken!"

count_tokens(text)

# --------------------------------------------------------------
# Count tokens for the longer text
# --------------------------------------------------------------

longer_text = """
Tiktoken is a fast BPE tokenizer for use with OpenAI's models.
It allows you to count tokens in text without making API calls.
This is useful for estimating costs and staying within token limits.
"""

count_tokens(longer_text)
