# Docker Setup for AI Application

This directory contains the Docker configuration for our AI application's infrastructure. Currently, it sets up a PostgreSQL database for storing AI events and results. Make sure Docker is installed and running.

## Docker Compose Configuration

Here's our `docker-compose.yml` explained:

```yaml
services:
  db:  # Service name
    image: postgres:16  # Using PostgreSQL version 16
    ports:
      - "5432:5432"  # Map container port to host port
    environment:
      - POSTGRES_USER=admin      # Database user
      - POSTGRES_PASSWORD=admin  # Database password
      - POSTGRES_DB=aidb        # Database name
    volumes:
      - postgres_data:/var/lib/postgresql/data  # Persist data

volumes:
  postgres_data:  # Named volume for data persistence
```

## Key Components Explained

1. **Service Definition**:
   - `db`: Names our PostgreSQL service
   - `image: postgres:17`: Uses official PostgreSQL 17 image

2. **Port Mapping**:
   - `"5432:5432"`: Maps container's port 5432 to host's port 5432
   - Format is `"HOST_PORT:CONTAINER_PORT"`

3. **Environment Variables**:
   - Sets up initial database credentials
   - These should match your application's `.env` configuration

4. **Data Persistence**:
   - Uses named volume `postgres_data`
   - Stores data in `/var/lib/postgresql/data` inside container
   - Data survives container restarts/removals

## Usage Instructions

1. **Start the Database**:
   ```bash
   cd week3/docker
   docker compose up -d
   ```
   The `-d` flag runs in detached mode (background)

2. **Check Status**:
   ```bash
   docker compose ps
   ```

3. **View Logs**:
   ```bash
   docker compose logs db
   ```

4. **Stop the Database**:
   ```bash
   docker compose down
   ```
   Note: Data persists even after stopping

5. **Remove Everything** (including data):
   ```bash
   docker compose down -v
   ```
   The `-v` flag removes volumes

## Connecting to the Database

- **Host**: localhost
- **Port**: 5432
- **User**: admin
- **Password**: admin
- **Database**: aidb

Connection URL format: `postgresql://admin:admin@localhost:5432/aidb`

### Using GUI Tools to connect to your database
You can connect using your preferred database GUI:

- **DBeaver** (Free, all platforms): https://dbeaver.io/
- **pgAdmin** (Free, PostgreSQL specific): https://www.pgadmin.org/
- **DataGrip** (JetBrains paid, all platforms): https://www.jetbrains.com/datagrip/
- **TablePlus** (Freemium, macOS/Windows): https://tableplus.com/

## Data Persistence Explained

- Docker containers are ephemeral (temporary)
- Named volumes (`postgres_data`) persist data
- Data stored in volume survives:
  - Container restarts
  - Container removal
  - Docker Compose down
- Only removed when explicitly using `docker compose down -v`

## Common Issues

1. **Port Conflict**:
   - Error: "port 5432 already in use"
   - Solution: Stop other PostgreSQL instances or change port mapping

2. **Permission Issues**:
   - Error: "permission denied"
   - Solution: Use `sudo` or add user to docker group

3. **Connection Refused**:
   - Wait a few seconds after starting
   - Database needs time to initialize

## Next Steps

After setting up Docker:

1. Update your `.env` file with database credentials
2. Run database initialization script
3. Start developing with persistent data storage

For more details on Docker and PostgreSQL, refer to:
- [Docker Documentation](https://docs.docker.com/)
- [PostgreSQL Docker Hub](https://hub.docker.com/_/postgres)