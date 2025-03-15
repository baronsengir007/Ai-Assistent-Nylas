#!/usr/bin/env python3
"""
Strategy Pattern Example for Text Processing

This module demonstrates the Strategy design pattern in the context
of text processing, which is a common task when working with LLMs.
"""

from abc import ABC, abstractmethod
from typing import List


class TextProcessingStrategy(ABC):
    """Abstract base class defining the interface for all text processing strategies."""

    @abstractmethod
    def process(self, text: str) -> str:
        """
        Process the input text according to the strategy.

        Args:
            text: The input text to process

        Returns:
            The processed text
        """
        pass


class SummarizationStrategy(TextProcessingStrategy):
    """Strategy for summarizing text by keeping only the first and last sentences."""

    def process(self, text: str) -> str:
        """
        Summarize text by extracting first and last sentences.

        Args:
            text: The input text to summarize

        Returns:
            A summary containing first and last sentences
        """
        sentences = text.split(". ")
        if len(sentences) <= 2:
            return text

        return f"{sentences[0]}. ... {sentences[-1]}"


class KeywordExtractionStrategy(TextProcessingStrategy):
    """Strategy for extracting keywords from text."""

    def __init__(self, stop_words: List[str] = None):
        """
        Initialize with optional stop words to ignore.

        Args:
            stop_words: List of common words to ignore
        """
        self.stop_words = stop_words or [
            "the",
            "and",
            "is",
            "in",
            "to",
            "of",
            "a",
            "for",
            "with",
        ]

    def process(self, text: str) -> str:
        """
        Extract keywords from text by removing stop words and sorting by frequency.

        Args:
            text: The input text to process

        Returns:
            A string of extracted keywords
        """
        # Convert to lowercase and split into words
        words = text.lower().replace(".", "").replace(",", "").split()

        # Remove stop words
        filtered_words = [word for word in words if word not in self.stop_words]

        # Count word frequency
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency (descending)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # Return top 5 keywords with their counts
        top_keywords = sorted_words[:5]
        return ", ".join([f"{word}({count})" for word, count in top_keywords])


class SentimentAnalysisStrategy(TextProcessingStrategy):
    """Strategy for basic sentiment analysis of text."""

    def __init__(self):
        """Initialize with positive and negative word lists."""
        self.positive_words = [
            "good",
            "great",
            "excellent",
            "positive",
            "wonderful",
            "best",
            "love",
        ]
        self.negative_words = [
            "bad",
            "terrible",
            "negative",
            "worst",
            "hate",
            "awful",
            "poor",
        ]

    def process(self, text: str) -> str:
        """
        Analyze sentiment of text based on positive and negative word counts.

        Args:
            text: The input text to analyze

        Returns:
            A string indicating the sentiment analysis result
        """
        text_lower = text.lower()

        # Count positive and negative words
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)

        # Determine sentiment
        if positive_count > negative_count:
            return f"Positive sentiment (score: +{positive_count - negative_count})"
        elif negative_count > positive_count:
            return f"Negative sentiment (score: -{negative_count - positive_count})"
        else:
            return "Neutral sentiment"


class TextProcessor:
    """Context class that uses a text processing strategy."""

    def __init__(self, strategy: TextProcessingStrategy):
        """
        Initialize with a strategy.

        Args:
            strategy: The text processing strategy to use
        """
        self._strategy = strategy

    def set_strategy(self, strategy: TextProcessingStrategy):
        """
        Change the strategy at runtime.

        Args:
            strategy: The new text processing strategy to use
        """
        self._strategy = strategy

    def process_text(self, text: str) -> str:
        """
        Process text using the current strategy.

        Args:
            text: The input text to process

        Returns:
            The processed text
        """
        return self._strategy.process(text)


def main():
    """Example usage of the Strategy pattern for text processing."""

    # Sample text to process
    sample_text = """
    Natural language processing is a subfield of artificial intelligence. 
    It focuses on the interaction between computers and human language.
    NLP technologies are used in many applications including chatbots, 
    translation services, and sentiment analysis tools. 
    The field has seen tremendous progress with the advent of large language models.
    """

    # Create strategies
    summarizer = SummarizationStrategy()
    keyword_extractor = KeywordExtractionStrategy()
    sentiment_analyzer = SentimentAnalysisStrategy()

    # Create processor with initial strategy
    processor = TextProcessor(summarizer)

    # Process text with summarization strategy
    print("\n=== Using Summarization Strategy ===")
    result = processor.process_text(sample_text)
    print(result)

    # Switch to keyword extraction strategy
    processor.set_strategy(keyword_extractor)
    print("\n=== Using Keyword Extraction Strategy ===")
    result = processor.process_text(sample_text)
    print(result)

    # Switch to sentiment analysis strategy
    processor.set_strategy(sentiment_analyzer)
    print("\n=== Using Sentiment Analysis Strategy ===")
    result = processor.process_text(sample_text)
    print(result)

    # Process a different text with the current strategy
    print("\n=== Processing Different Text with Current Strategy ===")
    new_text = "I love how NLP models have improved. They're excellent at understanding context now."
    result = processor.process_text(new_text)
    print(result)


if __name__ == "__main__":
    main()
