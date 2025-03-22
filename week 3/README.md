# Week 3: Event-Driven Architecture & Containerization

This week provides hands-on experience with essential infrastructure components for AI applications. We'll explore each component individually to understand its role in building production-ready AI systems.

## Lab Exercises

### 1. Configuration Management with Pydantic
**File**: `e1_pydantic_settings.py`
- Learn how to manage application settings
- Handle environment variables securely
- Validate configuration with Pydantic
- Set up OpenAI API credentials

### 2. Building AI Endpoints with FastAPI
**File**: `e2_fastapi_quickstart.py` and `e3_test_fastapi.py`
- Create a FastAPI application
- Implement structured LLM outputs with Instructor
- Define type-safe request/response models
- Test API endpoints

### 3. Database Setup with Docker
**Directory**: `docker/`
- Set up PostgreSQL in Docker
- Understand container configuration
- Manage persistent data
- Connect to the database

### 4. Database Operations
**File**: `e4_database_setup.py`
- Define SQLAlchemy models
- Implement basic CRUD operations
- Store AI processing events
- Query the database

## Running the Exercises

1. **Configuration Setup**:
```bash
# Copy example env file
cp .env.example .env
# Add your OpenAI API key to .env
```

2. **Start Database**:
```bash
cd docker
docker compose up -d
```

3. **Run Exercises in Order**:
```bash
# Exercise 1: Test configuration
python e1_pydantic_settings.py

# Exercise 2: Start API server
python e2_fastapi_quickstart.py

# Exercise 3: Test the API service
python e3_test_fastapi.py

# Exercise 4: Start docker container (see README in /docker)
docker compose up -d

# Exercise 5: Test database operations
# Connect to database with a database GUI

# Exercise 6: Test database operations
python e4_database_setup.py
```

## What You'll Learn

- How to manage configuration in AI applications
- Building type-safe APIs with structured LLM outputs
- Working with databases in Docker containers
- Storing and retrieving AI processing results

## Next Steps

These exercises provide the foundation for understanding the individual components used in the GenAI Launchpad framework. In the coming weeks, we'll see how these components work together in a production-ready AI system.

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Docker Documentation](https://docs.docker.com/)
- [Instructor Documentation](https://python.useinstructor.com/)

## Tips

- Focus on understanding each component individually
- Check the Docker README for database connection help
- Use the FastAPI interactive docs at http://localhost:8000/docs
- Experiment with different LLM prompts and structured outputs 