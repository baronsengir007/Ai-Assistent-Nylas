"""
PostgreSQL Database Setup with SQLAlchemy

This example demonstrates how to:
1. Set up PostgreSQL connection with SQLAlchemy
2. Define database models for our AI application
3. Create tables and basic operations

Make sure the database is running:
docker/docker compose up -d (check README in /docker)
"""

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    JSON,
    DateTime,
    Enum as SQLEnum,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import enum

# Import our settings
from e1_pydantic_settings import get_settings

settings = get_settings()

# Initialize SQLAlchemy
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Define status enum
class ProcessingStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AIEvent(Base):
    """Model to store AI processing events"""

    __tablename__ = "ai_events"

    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(String)
    max_tokens = Column(Integer)
    status = Column(SQLEnum(ProcessingStatus), default=ProcessingStatus.PENDING)
    result = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)


def init_db():
    """Initialize the database"""
    try:
        # Create tables
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully!")

        # Test connection
        with SessionLocal() as db:
            # Create a test event
            test_event = AIEvent(input_text="Test message", max_tokens=100)
            db.add(test_event)
            db.commit()
            print("Test event created successfully!")

            # Query the event back
            event = db.query(AIEvent).first()
            print("\nRetrieved test event:")
            print(f"ID: {event.id}")
            print(f"Status: {event.status}")
            print(f"Created at: {event.created_at}")

    except Exception as e:
        print(f"Error initializing database: {e}")


def main():
    print("\nInitializing database...")
    print(f"Database URL: {settings.database_url}")
    init_db()
    print("\nTip: Make sure Docker is running and the database container is started!")
    print("Next: Update b_fastapi_quickstart.py to store events in the database")


if __name__ == "__main__":
    main()
