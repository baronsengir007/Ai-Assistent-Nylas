from celery_app import celery_app
import time


@celery_app.task
def add(x, y):
    # Simulate a time-consuming tasks
    time.sleep(5)
    return x + y


@celery_app.task
def process_data(data):
    # Another example task
    time.sleep(3)
    return f"Processed: {data}"
