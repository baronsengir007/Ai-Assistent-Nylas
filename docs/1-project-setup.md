# AI Executive Assistant - Project Setup

## This Week's Goals

1. Introduction to the project
2. Understanding the project architecture
3. Seting up and configuring Nylas
4. Setting up your webhook server
5. Receiving your first webhook
6. Understanding the webhook data model

## Additional Resources

- [Pydantic](https://docs.pydantic.dev/latest/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Pinggy](https://pinggy.io/)
- [Nylas](https://developer.nylas.com/docs/v3/getting-started/)

## API Quickstart

This tutorial will walk you through creating a functional app with Nylas by setting up user authentication using the Nylas API in just a few simple steps.

### 1. Prerequisites
- Python 3.12 or higher
- A Nylas Developer Account (free tier available)
- Git
- GitHub account

### 2. Repository Setup
1. First, create your own repository on GitHub:
   - Go to [GitHub](https://github.com)
   - Click "New repository"
   - Name it `ai-assistant`
   - Make sure it's set to private!
   - Don't initialize with README

2. Clone the template repository and set up your own:
   ```bash
   # Clone the template repository
   git clone https://github.com/datalumina/genai-accelerator-labs.git
   cd genai-accelerator-labs

   # Switch to the correct branch
   git checkout project/ai-assistant-week-1

   # Create a new git repository
   rm -rf .git
   git init

   # Add your repository as remote
   git remote add origin https://github.com/YOUR-USERNAME/ai-assistant.git

   # Push to your repository
   git add .
   git commit -m "Initial commit"
   git push -u origin main
   ```

### 3. Environment Setup
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

If you're using VS Code or Cursor I recommend to update your `.code-workspace` file:

```json
"settings": {
   "python.analysis.extraPaths": [
      "./app",
      "./playground"
   ]
}
```

### 4. Nylas Setup
1. Sign up for a Nylas Developer Account at [dashboard-v3.nylas.com/register](https://dashboard-v3.nylas.com/register)
2. Get your credentials from the Nylas Dashboard:
   - Client ID
   - API Key
   - Choose your API URI (US or EU)

### 5. Configuration
1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Update `.env` with your Nylas credentials:
   ```env
   NYLAS_CLIENT_ID=your_client_id
   NYLAS_API_KEY=your_api_key
   NYLAS_API_URI=https://api.us.nylas.com  # or https://api.eu.nylas.com
   EMAIL=youremail@domain.com
   ```

3. Set up OAuth callback:
   - Go to Nylas Dashboard → Hosted Authentication
   - Add callback URI: `http://localhost:5010/oauth/exchange`

### 6. Running the Application
```bash
# Start the server to configure auth
cd app/config
uv run config_auth.py
```

1. Visit `http://localhost:5010/nylas/auth` to begin the authentication process. 
2. Log in with your Gmail account to authenticate
3. Store the grant ID in your `.env` file in `NYLAS_GRANT_ID`
4. Visit `http://localhost:5010/nylas/recent-emails` to view your last emails
5. Visit `http://localhost:5010/nylas/send-email` to send a test email to your own account


Congrats! You've now successfully configured Nylas with your account.

## Webhook Quickstart

This guide walks you through the process of creating a sample web app that receives webhooks from Nylas whenever you get a new email. Webhooks are essential for real-time email processing as they allow Nylas to notify your application immediately when new emails arrive, rather than having to constantly poll for changes.

### 1. Configure a Tunnel

To receive webhooks from Nylas, your local development server needs to be accessible from the internet. This is where tunneling comes in - it creates a secure connection between your local machine and the internet, allowing Nylas to send webhook notifications to your local development environment.

We'll use [Pinggy](https://pinggy.io/) for tunneling because it offers several advantages over alternatives like ngrok:
- No account required
- Simple setup with a single command
- Secure HTTPS endpoints
- Real-time connection status

Run this command in your terminal:

```bash
ssh -p 443 -R0:localhost:8000 free.pinggy.io
```

After running the command, you'll receive a public URL (e.g., `https://rnzer-185-200-135-131.a.free.pinggy.link`). Copy this URL and add it to your `.env` file:

```env
SERVER_URL=https://your-pinggy-url
```

### 2. Start the Server

```bash
cd app
uv run main.py
```

Visit http://localhost:8000 to view the website. You should see "Nylas Webhooks" displayed, indicating that your local server is running correctly.

### 3. Create the Webhook

```bash
cd app/config
uv run config_webhook.py
```

This script will:
1. Create a webhook endpoint in your Nylas account
2. Generate a webhook secret for security
3. Configure the webhook to notify your application of new emails

Copy the webhook secret and add it to your `.env` file:

```env
WEBHOOK_SECRET=your_webhook_secret
```

### 4. Confirm Webhook in Dashboard

Go to the Nylas Dashboard and navigate to **Notifications**. You should see your webhook listed there. This confirms that Nylas is configured to send notifications to your application.

#### How Webhooks Work

Here's what happens when you receive a new email:

1. Nylas detects a new email in your connected account
2. Nylas sends a POST request to your webhook URL (the Pinggy URL)
3. Pinggy forwards this request to your local server
4. Your application processes the webhook data
5. The email content is displayed on your local server

The flow looks like this:
```
New Email → Nylas → Pinggy Tunnel → Your Local Server
```

This setup allows you to develop and test webhook functionality locally while still receiving real-time notifications from Nylas.

### 5. Test the Webhook

1. Send a test email to the address you specified in your `.env` file's `EMAIL` variable
2. Watch your terminal for the webhook request
3. Refresh http://localhost:8000 to see the incoming message

If everything is configured correctly, you should see:
- A 200 status code in your terminal
- The new email content displayed on the webpage

Congratulations! You've now set up a complete webhook system that can receive real-time email notifications.