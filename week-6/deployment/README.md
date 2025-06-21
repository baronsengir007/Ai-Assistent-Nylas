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

## Step 3: Exposing the API Endpoint (Without Reverse Proxy)

By default, the API service is configured to only accept connections from localhost, which prevents external access. To make your API accessible from the internet, you need to modify the port binding configuration.

**Required configuration change:**

Navigate to your Docker Compose file and locate the API service configuration in `docker/docker-compose.launchpad.yml`. You'll need to update the ports section as follows:

```yaml
# Change this restrictive binding:
ports:
  - "127.0.0.1:8080:8080"

# To this open binding:
ports:
  - "0.0.0.0:8080:8080"
```

**Understanding the change:**
- `127.0.0.1:8080:8080` only allows connections from localhost (the server itself)
- `0.0.0.0:8080:8080` allows connections from any IP address, making your API publicly accessible

**After making this change:**
- Your API will be accessible at: `http://YOUR_SERVER_IP:8080`
- Supabase services will be accessible at: `http://YOUR_SERVER_IP:8000` (or your configured KONG_HTTP_PORT)

**Important security consideration:**
This configuration exposes your API directly to the internet without additional security layers. For production deployments, you should consider implementing a reverse proxy solution like Caddy   with SSL certificates to provide better security and encrypted connections.

## Step 4: Test Your API Access

Once your deployment is complete and your API is properly exposed, you need to verify that everything is working correctly.

**Run the test script:**
Execute the `week-6/deployment/request.py` script to send test events (update with your server public IP) to your deployed endpoint. This script will help you verify that your API is responding correctly and processing requests as expected.

**Monitor your deployment:**
While running the test script, monitor the logs on your server through the terminal to observe real-time activity and ensure there are no errors in the processing pipeline.

**Verify data persistence:**
Check your Supabase events table to confirm that the test events are being properly stored in your database. This final verification step ensures that your entire data pipeline is functioning correctly from API reception to database storage.