import os

# Set Redis connection defaults for local development
# Uses localhost if env vars aren't set, otherwise uses existing env vars
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

from tasks import add, process_data

if __name__ == "__main__":
    # Example 1: Adding numbers
    # Send task to worker (non-blocking)
    result = add.delay(x=4, y=4)
    print(f"Task sent! Task ID: {result.id}")

    # Check task status (non-blocking)
    print(f"Is the task ready? {result.ready()}")  # Likely False initially

    # Wait for result (blocking)
    print("Waiting for task to complete...")
    task_result = result.get()
    print(f"Result of add(4,4): {task_result}")
    print(f"Is the task ready now? {result.ready()}")  # Will be True

    # Example 2: Processing data
    print("\nSending another task...")
    data_task = process_data.delay("sample data")
    print(f"Processing task sent! Task ID: {data_task.id}")

    # Wait for second task result
    print("Waiting for second task to complete...")
    processed_data_result = data_task.get()
    print(f"Result of process_data('sample data'): {processed_data_result}")

    print("\nClient finished.")
