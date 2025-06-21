import * as tmi from '@tmi.js/chat';
import { logger } from './logger';
import { config } from './config';
import { RabbitMQService } from './rabbitmq';
import { getAccessToken } from './twitchAuth';

// TODO: Fill in your Twitch Client ID for API requests
// export const TWITCH_CLIENT_ID = "";

// TODO: Fill in the broadcaster's ID you want to listen to
// export const BROADCASTER_ID = "";

// TODO: Define the scopes your application needs
// The specific scopes depend on the actions your bot will perform.
// For example, to read chat, you need "chat:read".
// To send messages, you need "chat:edit".
// See: https://dev.twitch.tv/docs/authentication/scopes
// export const TWITCH_SCOPES = ["chat:read", "chat:edit"];

interface TwitchMessageEvent {
  channel: { login: string };
  user: { login: string };
  message: { text: string };
}

export class TwitchService {
  private client: tmi.Client | null = null;
  private rabbitmq: RabbitMQService;

  constructor(rabbitmq: RabbitMQService) {
    this.rabbitmq = rabbitmq;
  }

  private setupEventHandlers(): void {
    if (!this.client) {
      logger.error('Twitch client not initialized before setting up event handlers.');
      return;
    }
    this.client.on('message', async (event: TwitchMessageEvent) => {
      logger.info(`Received from Twitch #${event.channel.login} <${event.user.login}>: ${event.message.text}`);
      try {
        await this.rabbitmq.publishMessage({
          username: event.user.login,
          message: event.message.text,
          channel: event.channel.login,
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        logger.error('Error processing message:', error);
      }
    });
  }

  async connect(): Promise<void> {
    try {
      const accessToken = await getAccessToken();
      if (!accessToken) {
        logger.error('Failed to get Twitch access token. Please run the authentication server via `ts-node src/twitchAuth.ts` and authorize at http://localhost:3000/auth');
        return;
      }

      this.client = new tmi.Client({
        channels: config.TWITCH_CHANNELS,
        token: accessToken,
      });

      this.setupEventHandlers();
      await this.client.connect();

      logger.info('Connected to Twitch chat');
    } catch (error) {
      logger.error('Failed to connect to Twitch:', error);
    }
  }

  async disconnect(): Promise<void> {
    try {
      // The new library doesn't have a disconnect method.
      // It will disconnect automatically when the process exits.
      logger.info('Twitch client will disconnect on process exit.');
    } catch (error) {
      logger.error('Error disconnecting from Twitch:', error);
    }
  }

  async sendMessage(channel: string, message: string): Promise<void> {
    if (!this.client) {
      logger.error('Cannot send message, Twitch client is not connected.');
      return;
    }
    try {
      await this.client.say(channel, message);
    } catch (error) {
      logger.error(`Failed to send message to ${channel}:`, error);
    }
  }
} 