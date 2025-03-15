"""
Chain of Responsibility Pattern Implementation for LLM Text Processing

This module demonstrates the Chain of Responsibility design pattern
in the context of processing text for an LLM application.
"""

from abc import ABC, abstractmethod
import re


class TextRequest:
    """
    Represents a text processing request that passes through the chain.

    Attributes:
        original_text (str): The original input text
        processed_text (str): The text after processing
        prompt (str): The constructed prompt for the LLM
        token_count (int): The number of tokens in the prompt
        response (str): The response from the LLM
    """

    def __init__(self, text: str):
        """
        Initialize a new text request.

        Args:
            text: The original input text
        """
        self.original_text = text
        self.processed_text = text
        self.prompt = ""
        self.token_count = 0
        self.response = ""
        self.errors = []


class Handler(ABC):
    """
    Abstract base class for handlers in the chain of responsibility.
    """

    def __init__(self):
        """Initialize the handler with no successor."""
        self._next_handler = None

    def set_next(self, handler):
        """
        Set the next handler in the chain.

        Args:
            handler: The next handler in the chain

        Returns:
            The next handler, allowing for method chaining
        """
        self._next_handler = handler
        return handler

    def handle(self, request: TextRequest) -> TextRequest:
        """
        Process the request and pass it to the next handler if one exists.

        Args:
            request: The text request to process

        Returns:
            The processed text request
        """
        if self._process(request) and self._next_handler:
            return self._next_handler.handle(request)
        return request

    @abstractmethod
    def _process(self, request: TextRequest) -> bool:
        """
        Process the request. To be implemented by concrete handlers.

        Args:
            request: The text request to process

        Returns:
            True if the chain should continue, False otherwise
        """
        pass


class TextNormalizationHandler(Handler):
    """Handler that normalizes text by removing extra whitespace and fixing common issues."""

    def _process(self, request: TextRequest) -> bool:
        """
        Normalize the text in the request.

        Args:
            request: The text request to process

        Returns:
            True to continue the chain
        """
        print("TextNormalizationHandler: Normalizing text...")

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", request.processed_text)

        # Fix common punctuation issues
        text = re.sub(r"\s([.,;:!?])", r"\1", text)

        # Trim whitespace
        text = text.strip()

        request.processed_text = text
        return True


class PromptConstructionHandler(Handler):
    """Handler that constructs a prompt for the LLM based on the processed text."""

    def __init__(self, template: str = "Answer the following question: {text}"):
        """
        Initialize with a prompt template.

        Args:
            template: The prompt template to use
        """
        super().__init__()
        self.template = template

    def _process(self, request: TextRequest) -> bool:
        """
        Construct a prompt using the template and processed text.

        Args:
            request: The text request to process

        Returns:
            True to continue the chain
        """
        print("PromptConstructionHandler: Building prompt...")
        request.prompt = self.template.format(text=request.processed_text)
        return True


class LLMProcessingHandler(Handler):
    """Handler that sends the prompt to an LLM and gets a response."""

    def __init__(self, api_key: str = None):
        """
        Initialize with an API key.

        Args:
            api_key: The API key for the LLM service
        """
        super().__init__()
        self.api_key = api_key

    def _process(self, request: TextRequest) -> bool:
        """
        Process the request by sending it to an LLM.

        Args:
            request: The text request to process

        Returns:
            True after processing
        """
        print("LLMProcessingHandler: Sending to LLM...")

        # In a real implementation, this would call the OpenAI API or similar
        # For this example, we'll simulate a response
        request.response = f"This is a simulated response to: {request.prompt[:30]}..."

        print("LLMProcessingHandler: Received response")
        return True


def main():
    """Example usage of the Chain of Responsibility pattern."""

    # Create the chain of handlers
    normalizer = TextNormalizationHandler()
    prompt_builder = PromptConstructionHandler()
    llm_processor = LLMProcessingHandler()

    # Set up the chain
    normalizer.set_next(prompt_builder).set_next(llm_processor)

    # Process a request
    request = TextRequest("What is the capital   of France?  ")
    result = normalizer.handle(request)

    # Print the results
    print("\nResults:")
    print(f"Original text: '{result.original_text}'")
    print(f"Processed text: '{result.processed_text}'")
    print(f"Prompt: '{result.prompt}'")
    print(f"Response: '{result.response}'")

    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()
