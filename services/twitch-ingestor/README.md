# Twitch Chat Ingestor Service

A robust microservice that connects to Twitch chat, listens to messages in real-time, and publishes them to a RabbitMQ topic exchange. This service uses a long-term OAuth2 strategy with token refreshing, ensuring your bot can stay connected indefinitely.

## Features

- **Reliable Auth**: Implements Twitch's OAuth2 Authorization Code Flow with automatic token refreshing. No need to manually generate tokens again!
- **RabbitMQ Integration**: Publishes all chat messages to a RabbitMQ topic exchange, allowing other microservices to consume them easily.
- **Structured Logging**: Uses Winston for clear, structured logging.
- **Environment-based Configuration**: All sensitive keys and settings are loaded from a `.env` file.
- **Docker-Ready**: Fully containerized with Docker and Docker Compose for easy and consistent deployment.

## Prerequisites

- **Node.js**: v20 or later
- **Docker & Docker Compose**: For running the service and its RabbitMQ dependency.
- **A Twitch Account**: To register your application.

---

## 🚀 Getting Started (5-Minute Guide)

Follow these steps to get the ingestor running.

### Step 1: Configure Your Environment

First, you need to provide your Twitch application credentials.

1.  **Create a `.env` file** in this directory (`services/twitch-ingestor/`) by copying the example:
    ```bash
    cp .env.example .env
    ```

2.  **Register Your Application on Twitch:**
    *   Go to the [Twitch Developer Console](https://dev.twitch.tv/console) and register a new application.
    *   Choose **"Chat Bot"** as the category.
    *   Under **"OAuth Redirect URLs"**, add the following URL **exactly**: `http://localhost:3000/auth/callback`
    *   Save your application. You will get a **Client ID** and can generate a **Client Secret**.

3.  **Edit your `.env` file** and fill in the values from the Twitch Developer console:

    ```env
    # .env

    # Credentials from your app on https://dev.twitch.tv/console
    TWITCH_CLIENT_ID=your_client_id_here
    TWITCH_CLIENT_SECRET=your_client_secret_here

    # This MUST match the redirect URL in your Twitch app settings
    TWITCH_REDIRECT_URI=http://localhost:3000/auth/callback

    # The channel(s) your bot should join, separated by commas
    TWITCH_CHANNELS=your_favorite_streamer

    # The permissions your bot needs. 'chat:read' and 'chat:edit' are common.
    TWITCH_SCOPES="chat:read chat:edit"
    ```

### Step 2: Authenticate with Twitch (One-Time Setup)

Next, you need to authorize your application to get your initial tokens. This process only needs to be done once.

1.  **Start the Authentication Server:** In a terminal, run the following command from the project's root directory (`/path/to/consensus`):
    ```bash
    npx ts-node services/twitch-ingestor/src/twitchAuth.ts
    ```

2.  **Authorize in Your Browser:**
    *   While the server is running, open your web browser and go to:
        **[http://localhost:3000/auth](http://localhost:3000/auth)**
    *   This will redirect you to Twitch. Click "Authorize" to approve the connection.
    *   You will be redirected back and see a message: `Authentication successful! You can close this window.`

3.  **Stop the Auth Server:** A `tokenStore.json` file has now been created in `src/`. You can stop the auth server by pressing `Ctrl+C` in its terminal.

### Step 3: Run the Service with Docker

Now you can start the main application and RabbitMQ.

1.  **Make sure Docker is running** on your machine.

2.  **Start the services** from this directory (`services/twitch-ingestor/`) using Docker Compose:
    ```bash
    docker-compose up --build -d
    ```
    This command builds the `twitch-ingestor` image and starts both it and the RabbitMQ container in the background.

3.  **Check the logs** to see it in action:
    ```bash
    docker-compose logs -f twitch-ingestor
    ```
    You should see logs confirming connections to RabbitMQ and Twitch.

---

## ✅ Verifying the Output

The easiest way to confirm that messages are flowing is by using the RabbitMQ Management UI.

1.  **Open the UI:** In your browser, navigate to **[http://localhost:15672](http://localhost:15672)**.
2.  **Log In:** Use the default credentials: `guest` / `guest`.
3.  **Create a Test Queue:**
    *   Go to the **"Queues"** tab.
    *   Under "Add a new queue", enter a name like `chat_test_consumer` and click "Add queue".
4.  **Bind the Queue:**
    *   Click on your new queue's name in the list.
    *   In the **"Bindings"** section, use the `twitch_chat` exchange and a routing key of `chat.#` to capture all chat messages.
    *   Click "Bind".
5.  **Get Messages:**
    *   On the same page, find the **"Get messages"** panel.
    *   Click the **"Get Message(s)"** button.
    *   As messages are sent in the Twitch channel, they will appear here, confirming the entire pipeline is working!

## Message Format

Messages are published to the `twitch_chat` exchange with a routing key of `chat.<channel_name>`. The message payload is a JSON object:

```json
{
  "username": "<twitch_username>",
  "message": "<chat_message>",
  "channel": "<streamer_channel>",
  "timestamp": "<ISO_timestamp>"
}
```

## Local Development (Without Docker)

If you prefer to run the service locally without containerizing it:

1.  Ensure you have a RabbitMQ instance running and accessible. You can start one quickly with Docker: `docker-compose up rabbitmq -d`
2.  Install local dependencies: `npm install`
3.  Run the development server: `npm run dev`


## Screenshots of my run

![image](https://github.com/user-attachments/assets/cb862136-195e-482d-8b0f-36d45822c5db)



## License

MIT 
