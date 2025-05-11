from abc import abstractmethod
from pydantic import BaseModel


class TaskContext(BaseModel):
    text: str


class Handler:
    def __init__(self, name):
        self.name = name
        self.next_handler = None

    def set_next(self, handler):
        self.next_handler = handler
        return handler

    @abstractmethod
    def process(self, task_context: TaskContext):
        # Must be implemented by concrete handlers
        pass


class TextPreprocessor(Handler):
    def process(self, task_context: TaskContext):
        print(f"{self.name}: Preprocessing text...")
        task_context.text = task_context.text.lower().strip()

        if self.next_handler:
            return self.next_handler.process(task_context)
        return task_context


class EntityExtractor(Handler):
    def process(self, task_context: TaskContext):
        print(f"{self.name}: Extracting entities...")
        # Simulate entity extraction

        if self.next_handler:
            return self.next_handler.process(task_context)
        return task_context


class SentimentAnalyzer(Handler):
    def process(self, task_context: TaskContext):
        print(f"{self.name}: Analyzing sentiment...")
        task_context.text += " [SENTIMENT: POSITIVE]"

        if self.next_handler:
            return self.next_handler.process(task_context)
        return task_context


# Usage example
if __name__ == "__main__":
    # Create handlers
    preprocessor = TextPreprocessor(name="Preprocessor")
    extractor = EntityExtractor(name="Entity Extractor")
    analyzer = SentimentAnalyzer(name="Sentiment Analyzer")

    # Build the chain
    preprocessor.set_next(extractor).set_next(analyzer)

    # Create task context and process it through the chain
    task = TaskContext(text="  Hello World!  ")
    result = preprocessor.process(task)

    print("\nFinal result:")
    print(result)
