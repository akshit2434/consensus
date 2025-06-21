import dotenv from 'dotenv';
import path from 'path';

// Load environment variables from .env file, ensuring it's loaded from the service directory.
dotenv.config({ path: path.resolve(__dirname, '../.env') });

const config = {
  // Twitch Configuration
  TWITCH_USERNAME: process.env.TWITCH_USERNAME,
  TWITCH_OAUTH_TOKEN: process.env.TWITCH_OAUTH_TOKEN,
  TWITCH_CHANNELS: process.env.TWITCH_CHANNELS?.split(',').map((ch) => ch.trim()) || [],
  TWITCH_CLIENT_ID: process.env.TWITCH_CLIENT_ID || '',
  TWITCH_CLIENT_SECRET: process.env.TWITCH_CLIENT_SECRET || '',
  TWITCH_REDIRECT_URI: process.env.TWITCH_REDIRECT_URI || 'http://localhost:3000/auth/callback',
  TWITCH_SCOPES: process.env.TWITCH_SCOPES || 'chat:read chat:edit',

  // RabbitMQ Configuration
  RABBITMQ_HOST: process.env.RABBITMQ_HOST || 'localhost',
  RABBITMQ_PORT: parseInt(process.env.RABBITMQ_PORT || '5672', 10),
  RABBITMQ_USERNAME: process.env.RABBITMQ_USERNAME || 'guest',
  RABBITMQ_PASSWORD: process.env.RABBITMQ_PASSWORD || 'guest',
  RABBITMQ_VHOST: process.env.RABBITMQ_VHOST || '/',

  // Service Configuration
  LOG_LEVEL: process.env.LOG_LEVEL || 'info',
  RECONNECT_DELAY_MS: parseInt(process.env.RECONNECT_DELAY_MS || '5000', 10),
  MAX_RECONNECT_ATTEMPTS: parseInt(process.env.MAX_RECONNECT_ATTEMPTS || '10', 10),
};

export { config };

export const getRabbitMQUrl = (): string => {
  const { RABBITMQ_USERNAME, RABBITMQ_PASSWORD, RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_VHOST } = config;
  return `amqp://${RABBITMQ_USERNAME}:${RABBITMQ_PASSWORD}@${RABBITMQ_HOST}:${RABBITMQ_PORT}${RABBITMQ_VHOST}`;
}; 