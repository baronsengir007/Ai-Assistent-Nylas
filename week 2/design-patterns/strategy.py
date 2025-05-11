from abc import ABC, abstractmethod
from pydantic import BaseModel


class TaskContext(BaseModel):
    text: str


# Strategy interface
class ProcessingStrategy(ABC):
    @abstractmethod
    def process(self, task_context: TaskContext) -> TaskContext:
        pass


# Concrete strategies
class SummarizationStrategy(ProcessingStrategy):
    def process(self, task_context: TaskContext) -> TaskContext:
        print("Applying summarization strategy...")
        result = TaskContext(text=f"Summary: {task_context.text[:50]}...")
        return result


class QuestionAnsweringStrategy(ProcessingStrategy):
    def process(self, task_context: TaskContext) -> TaskContext:
        print("Applying Q&A strategy...")
        result = TaskContext(text=f"Answer: {task_context.text}")
        return result


class TranslationStrategy(ProcessingStrategy):
    def process(self, task_context: TaskContext) -> TaskContext:
        print("Applying translation strategy...")
        result = TaskContext(text=f"Translation: {task_context.text}")
        return result


# Context router that chooses the appropriate strategy
class LLMRouter:
    def __init__(self):
        self.strategies = {}

    def register_strategy(self, task_type, strategy):
        self.strategies[task_type] = strategy

    def process(self, task_context: TaskContext, task_type: str = None) -> TaskContext:
        # Determine the task type if not provided
        if not task_type:
            task_type = self._determine_task_type(task_context.text)

        # Get the appropriate strategy
        if task_type in self.strategies:
            return self.strategies[task_type].process(task_context)
        else:
            print(f"No strategy found for task type: {task_type}")
            return task_context

    def _determine_task_type(self, text):
        """Simple heuristic to determine task type from text"""
        text = text.lower()
        if "summarize" in text or "summary" in text:
            return "summarize"
        elif "?" in text:
            return "question"
        elif "translate" in text:
            return "translate"
        return "unknown"


# Usage example
if __name__ == "__main__":
    # Create strategies
    summarize_strategy = SummarizationStrategy()
    qa_strategy = QuestionAnsweringStrategy()
    translate_strategy = TranslationStrategy()

    # Create router and register strategies
    router = LLMRouter()
    router.register_strategy("summarize", summarize_strategy)
    router.register_strategy("question", qa_strategy)
    router.register_strategy("translate", translate_strategy)

    # Create tasks and process them
    tasks = [
        (
            TaskContext(
                text="Summarize the following article: Lorem ipsum dolor sit amet..."
            ),
            "summarize",
        ),
        (TaskContext(text="What is the capital of France?"), "question"),
        (TaskContext(text="Translate this text to Spanish: Hello world"), "translate"),
        (TaskContext(text="This is an unknown task type"), "unknown"),
    ]

    for task, task_type in tasks:
        processed_task = router.process(task, task_type)
        print(f"\nInput: {task.text}")
        print(f"Output: {processed_task.text}")
