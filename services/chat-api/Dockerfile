FROM node:20-alpine
WORKDIR /app

# Copy the entire project
COPY . .

# Install dependencies and nx globally
RUN npm install -g nx && npm install

# Build the application
RUN nx build chat-api --skip-nx-cache --skip-linting

# Set working directory to the service
WORKDIR /app/services/chat-api

# Start the service
CMD ["node", "../../dist/services/chat-api/main.js"] 