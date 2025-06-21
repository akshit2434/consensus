import amqplib, { Connection, Channel, Replies } from 'amqplib';
import { logger } from './logger';
import { getRabbitMQUrl, config } from './config';

export interface RabbitMQMessage {
  username: string;
  message: string;
  channel: string;
  timestamp: string;
}

export class RabbitMQService {
  private connection: any = null;
  private channel: any = null;
  private exchange = 'twitch_chat';
  private reconnectDelay = 5000; // 5 seconds

  async connect(): Promise<void> {
    try {
      this.connection = await amqplib.connect(getRabbitMQUrl());
      logger.info('Connected to RabbitMQ.');

      this.connection.on('close', () => {
        logger.warn('RabbitMQ connection closed. Attempting to reconnect...');
        this.connection = null;
        this.channel = null;
        setTimeout(() => this.connect(), this.reconnectDelay);
      });
      this.connection.on('error', (err: Error) => {
        logger.error('RabbitMQ connection error:', err);
      });

      this.channel = await this.connection.createChannel();
      await this.channel.assertExchange(this.exchange, 'topic', { durable: false });
      logger.info('RabbitMQ channel created and exchange asserted.');

    } catch (error) {
      logger.error('Failed to connect to RabbitMQ:', error);
      setTimeout(() => this.connect(), this.reconnectDelay);
    }
  }

  async publishMessage(msg: RabbitMQMessage): Promise<void> {
    if (!this.channel) {
      logger.error('RabbitMQ channel is not available to publish message.');
      return;
    }
    const routingKey = `chat.${msg.channel}`;
    const messageBuffer = Buffer.from(JSON.stringify(msg));
    this.channel.publish(this.exchange, routingKey, messageBuffer);
    logger.info(`Published to RabbitMQ [${routingKey}]: Message from ${msg.username}`);
  }

  async close(): Promise<void> {
    try {
      await this.connection?.close();
    } catch (error) {
      logger.error('Error closing RabbitMQ connection:', error);
    }
  }
} 