# Deploying the GenAI Launchpad to Server

## Introduction

In this guide, you'll learn how to deploy the GenAI Launchpad to a remote server, making your application accessible from anywhere on the internet. We'll be using the quickstart branch of the GenAI Launchpad repository, which provides a streamlined setup process. 

To ensure you have full control over your deployment, you'll first need to create a copy of the quickstart branch and publish it to a new GitHub repository that you fully own. This approach gives you the flexibility to customize the code and manage your deployment independently.

## Step 1: Prepare Your Project Locally

First, you need to get your project ready for deployment by cloning the quickstart branch and setting up your own repository.

**Clone the quickstart branch:**
```bash
git clone --branch quickstart --single-branch https://github.com/datalumina/genai-launchpad.git
```

**Create your own repository:**
Navigate to your GitHub account and create a new repository called `genai-launchpad-quickstart`. Make sure to set this repository to private to keep your configuration secure.

**Push the code to your new repository:**
After creating your repository, push all the cloned code to your new repository so you have complete ownership and control over the codebase.

**Test your setup locally:**
Before deploying to the server, configure your environment variables in the `.env` file, create a virtual environment with `uv venv`, install the dependencies with `uv sync`, start the Docker containers, and run the `send_event` file to verify that everything is working correctly on your local machine.

## Step 2: Deploy Your Project to the Server

Now that your project is prepared and tested locally, it's time to deploy it to your remote server.

**Follow the official deployment guide:**
Navigate to the deployment documentation at: https://launchpad-docs.datalumina.com/tutorials/latest/deploying-on-vps

Use the `genai-accelerator-demo` server that you created in the previous tutorial for this deployment.

**Note about HTTPS configuration:**
You can skip the HTTPS configuration for now, as we'll be covering SSL certificate setup and secure connections in the next tutorial.

**Update environment variables on your server:**
Once you've cloned your repository on the server, you'll need to configure the environment variables in the Docker folder. If you've never worked with environment files on a server before, here's how to do it:

1. **Navigate to the Docker directory:**
   ```bash
   cd /opt/genai-launchpad-quickstart/docker
   ```

2. **Create the environment file:**
   ```bash
   cp .env.example .env
   ```
   This copies the example environment file to create your actual environment file.

3. **Edit the environment file:**
   ```bash
   nano .env
   ```
   This opens the nano text editor. If you prefer vim, you can use `vim .env` instead.

4. **Update your OpenAI API key:**
   You only need to update your OpenAI API key in the file. Find the line that looks like `OPENAI_API_KEY=` and replace it with your actual API key. You can leave all other settings at their default values.

5. **Save and exit:**
   - In nano: Press `Ctrl + X`, then `Y` to confirm, then `Enter` to save
   - In vim: Press `Esc`, type `:wq`, then press `Enter`

6. **Verify the file was created:**
   ```bash
   ls -la .env
   ```
   This should show your `.env` file with the correct permissions.

## Step 3: Exposing the API and Supabase Ports

By default, Hetzner servers have restrictive firewall settings that block most ports for security. You need to create additional firewall rules to allow access to your application ports while maintaining security by restricting access to your IP address only.

**Create firewall rules for your application ports:**

1. **Navigate to your Hetzner Cloud Console** and go to your server's firewall settings

2. **Add a rule for Supabase (Port 8000):**
   - **Direction:** Inbound
   - **Protocol:** TCP
   - **Port:** 8000
   - **Source:** Your IP address (find it at [whatismyipaddress.com](https://whatismyipaddress.com))
   - **Description:** Supabase access

3. **Add a rule for API (Port 8080):**
   - **Direction:** Inbound  
   - **Protocol:** TCP
   - **Port:** 8080
   - **Source:** Your IP address (same as above)
   - **Description:** API access

**Security recommendation:**
Just like with SSH (port 22), restricting access to your IP address only prevents unauthorized access to your application services. This is especially important since these ports will expose your database interface and API endpoints.

**Configure Docker to allow external API access:**

Since we're not using a reverse proxy in this setup, you also need to modify the Docker Compose configuration to make your API accessible from external connections (not just localhost):

4. **Edit the Docker Compose file:**
   ```bash
   nano /opt/genai-launchpad-quickstart/docker/docker-compose.launchpad.yml
   ```

5. **Find the API service section** and locate the ports configuration:
   ```yaml
   api:
     build:
       context: ..
       dockerfile: docker/Dockerfile.api
     container_name: "${PROJECT_NAME}_api"
     depends_on:
       - db
       - redis
     ports:
       - "127.0.0.1:8080:8080"  # This line needs to be changed
   ```

6. **Update the ports binding:**
   Change the line from:
   ```yaml
   - "127.0.0.1:8080:8080"
   ```
   to:
   ```yaml
   - "8080:8080"
   ```

**Understanding the change:**
- **`127.0.0.1:8080:8080`** only allows connections from localhost (the server itself)
- **`8080:8080`** allows connections from any IP address, making your API accessible from external clients
- This change is necessary because the firewall rules you created above will only work if the service itself accepts external connections

## Step 4: Start the Application

Navigate to the docker/ directory:

```bash
cd /opt/genai-launchpad-quickstart/docker
```

Run the startup script:

```bash
./start.sh
```

Verify everything is running correctly by running:

```bash
./logs.sh
```

## Step 5: Run the Database Migrations

Create and apply database migrations using Alembic to set up your database schema. Since Alembic and other required tools are not installed directly on the server, we need to run these commands from within the Docker container where all dependencies are available.

1. **Navigate to the app directory:**
   ```bash
   cd /opt/genai-launchpad-quickstart/app
   ```
   This is where the Alembic configuration and migration files are located in your project structure.

2. **Create a new migration:**
   ```bash
   docker exec -it launchpad_api ./makemigration.sh
   ```
   This script will prompt you for a message to describe the migration. You can enter something like: `init db`

3. **Apply the migration:**
   ```bash
   docker exec -it launchpad_api ./migrate.sh
   ```
   This will apply the migration that was just created, setting up your database tables and structure.

**Understanding the Docker commands:**
- **`docker exec`**: Executes a command inside a running Docker container
- **`-it`**: Interactive terminal flags that allow you to interact with the command (needed for the migration message prompt)
- **`launchpad_api`**: The name of the Docker container running your API service
- **`./makemigration.sh` and `./migrate.sh`**: The migration scripts that run Alembic commands

We use these Docker commands because Alembic and the Python environment with all dependencies are contained within the Docker container, not installed directly on the server. This approach ensures consistency and avoids potential conflicts with different Python versions or missing dependencies on the host system.

## Step 6: Test Your API Access

Once your deployment is complete and your API is properly exposed, you need to verify that everything is working correctly.

**Access Supabase Studio:**
First, verify that Supabase is running and your database is properly set up:

1. **Open your browser** and navigate to:
   ```
   http://YOUR_SERVER_IP:8000
   ```
   Replace `YOUR_SERVER_IP` with your actual server's IP address.

2. **Authenticate with Supabase:**
   - **Username:** `supabase`
   - **Password:** `supabase`

3. **Verify the database setup:**
   Once logged in to Supabase Studio, navigate to the Tables section and confirm that the `events` table exists and is properly configured.

**Run the test script:**
After verifying Supabase is accessible, test your API endpoint by executing the `week-6/deployment/request.py` script. Make sure to update the script with your server's public IP address before running it. This script will help you verify that your API is responding correctly and processing requests as expected.

**Monitor your deployment:**
While running the test script, monitor the logs on your server through the terminal to observe real-time activity and ensure there are no errors in the processing pipeline.

**Verify data persistence:**
Check your Supabase events table to confirm that the test events are being properly stored in your database. This final verification step ensures that your entire data pipeline is functioning correctly from API reception to database storage.