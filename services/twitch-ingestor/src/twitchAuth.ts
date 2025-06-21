import express from 'express';
import axios from 'axios';
import { promises as fs } from 'fs';
import path from 'path';
import { config } from './config';
import { logger } from './logger';

const app = express();
const port = 3000;

// #####################################################################################
// ## HOW TO USE:
// ## 1. Add your Twitch App credentials to your .env file.
// ##    - You can get these from the Twitch Developer Console: https://dev.twitch.tv/console
// ##    - Ensure the "OAuth Redirect URLs" in your Twitch App settings matches TWITCH_REDIRECT_URI.
// ##
// ## 2. Run this server separately using: `ts-node src/twitchAuth.ts`
// ##
// ## 3. Open your browser and navigate to `http://localhost:3000/auth`
// ##    - This will redirect you to Twitch to authorize your application.
// ##
// ## 4. After authorizing, you will be redirected back to the callback URL.
// ##    - The server will fetch and save your tokens to `src/tokenStore.json`.
// ##
// ## 5. The main application can now use `getAccessToken()` to get a valid token.
// ##    - The function will automatically refresh the token when it expires.
// #####################################################################################

const TOKEN_STORE_PATH = path.join(__dirname, 'tokenStore.json');

interface TokenData {
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
}

// Helper to save tokens
async function saveTokens(tokenData: any): Promise<void> {
  const { access_token, refresh_token, expires_in } = tokenData;
  const expiresAt = Date.now() + (expires_in * 1000);
  const data: TokenData = {
    accessToken: access_token,
    refreshToken: refresh_token,
    expiresAt,
  };
  await fs.writeFile(TOKEN_STORE_PATH, JSON.stringify(data, null, 2));
  logger.info('Tokens saved successfully.');
}

// Helper to load tokens
async function loadTokens(): Promise<TokenData | null> {
  try {
    const data = await fs.readFile(TOKEN_STORE_PATH, 'utf-8');
    return JSON.parse(data) as TokenData;
  } catch (error) {
    return null; // No token file exists
  }
}

// Route to start the OAuth flow
app.get('/auth', (req, res) => {
  const authUrl =
    `https://id.twitch.tv/oauth2/authorize?` +
    `client_id=${config.TWITCH_CLIENT_ID}` +
    `&redirect_uri=${encodeURIComponent(config.TWITCH_REDIRECT_URI)}` +
    `&response_type=code` +
    `&scope=${encodeURIComponent(config.TWITCH_SCOPES)}`;
  res.redirect(authUrl);
});

// Route to handle the callback from Twitch
app.get('/auth/callback', async (req, res) => {
  const { code } = req.query;

  if (!code || typeof code !== 'string') {
    return res.status(400).send('No authorization code provided.');
  }

  try {
    const tokenResponse = await axios.post(
      'https://id.twitch.tv/oauth2/token',
      null,
      {
        params: {
          client_id: config.TWITCH_CLIENT_ID,
          client_secret: config.TWITCH_CLIENT_SECRET,
          code,
          grant_type: 'authorization_code',
          redirect_uri: config.TWITCH_REDIRECT_URI,
        },
      }
    );

    await saveTokens(tokenResponse.data);
    res.send('Authentication successful! You can close this window.');
  } catch (error) {
    logger.error('Error fetching tokens:', error);
    res.status(500).send('Failed to fetch tokens.');
  }
});

/**
 * Gets a valid access token, refreshing it if necessary.
 * @returns The access token, or null if authentication is needed.
 */
export async function getAccessToken(): Promise<string | null> {
  let tokens = await loadTokens();

  if (!tokens) {
    logger.warn('No tokens found. Please run the auth server and authorize.');
    return null;
  }

  // Check if the token is expired or close to expiring (e.g., within the next minute)
  if (Date.now() >= tokens.expiresAt - 60000) {
    logger.info('Access token expired, refreshing...');
    try {
      const refreshResponse = await axios.post(
        'https://id.twitch.tv/oauth2/token',
        null,
        {
          params: {
            grant_type: 'refresh_token',
            refresh_token: tokens.refreshToken,
            client_id: config.TWITCH_CLIENT_ID,
            client_secret: config.TWITCH_CLIENT_SECRET,
          },
        }
      );
      await saveTokens(refreshResponse.data);
      tokens = await loadTokens(); // Reload the new tokens
    } catch (error) {
      logger.error('Failed to refresh access token:', error);
      return null;
    }
  }

  return tokens ? tokens.accessToken : null;
}

if (require.main === module) {
  app.listen(port, () => {
    logger.info(`Twitch auth server running on http://localhost:${port}`);
    logger.info(`Visit http://localhost:${port}/auth to authenticate.`);
  });
} 