import os

# Set Redis connection defaults for local development
# Uses localhost if env vars aren't set, otherwise uses existing env vars
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Import both the tasks and the celery app
from tasks import add
from celery_app import celery_app

if __name__ == "__main__":
    # Example 1: Adding numbers using task.delay()
    print("\n--- Example 1: Using task.delay() ---")
    result = add.delay(x=4, y=4)
    print(f"Task sent! Task ID: {result.id}")

    # Check task status (non-blocking)
    print(f"Is the task ready? {result.ready()}")  # Likely False initially

    # Wait for result (blocking)
    print("Waiting for task to complete...")
    task_result = result.get()
    print(f"Result of add(4,4): {task_result}")
    print(f"Is the task ready now? {result.ready()}")  # Will be True

    # Example 2: Processing data using send_task()
    print("\n--- Example 2: Using app.send_task() ---")
    # Note: We use the task name as a string, no need to import the function
    data_task = celery_app.send_task("tasks.process_data", args=["sample data"])
    print(f"Processing task sent! Task ID: {data_task.id}")

    # Wait for second task result
    print("Waiting for second task to complete...")
    processed_data_result = data_task.get()
    print(f"Result of process_data('sample data'): {processed_data_result}")

    # Example 3: Another send_task example with add
    print("\n--- Example 3: Using send_task() with add ---")
    add_task = celery_app.send_task("tasks.add", args=[5, 5])
    print(f"Add task sent! Task ID: {add_task.id}")
    print(f"Result of add(5,5): {add_task.get()}")

    print("\nClient finished.")
