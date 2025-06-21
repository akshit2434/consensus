import { logger } from './logger';
import { RabbitMQService } from './rabbitmq';
import { TwitchService } from './twitch';

async function main() {
  logger.info('Starting Twitch chat ingestor service...');

  const rabbitmq = new RabbitMQService();
  const twitch = new TwitchService(rabbitmq);

  // Handle graceful shutdown
  const shutdown = async () => {
    logger.info('Shutting down services...');
    await twitch.disconnect();
    await rabbitmq.close();
    process.exit(0);
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);

  try {
    // Connect to RabbitMQ first
    await rabbitmq.connect();
    
    // Then connect to Twitch
    await twitch.connect();
    
    logger.info('Service started successfully');
  } catch (error) {
    logger.error('Failed to start service:', error);
    process.exit(1);
  }
}

main().catch((error) => {
  logger.error('Unhandled error:', error);
  process.exit(1);
}); 