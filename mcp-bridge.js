#!/usr/bin/env node

/**
 * MCP HTTP Bridge for Video Watch
 * Proxies MCP protocol (stdio) to Modal HTTP endpoint
 */

import * as readline from 'readline';

const WORKER_URL = process.env.VIDEO_WATCH_URL || 'https://amelia-fjorde--video-watch-mcp-mcp-server.modal.run';

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false
});

function log(msg) {
  console.error(`[video-watch-bridge] ${msg}`);
}

async function proxyRequest(request) {
  try {
    const response = await fetch(WORKER_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      log(`HTTP error: ${response.status} ${response.statusText}`);
      return {
        jsonrpc: '2.0',
        id: request.id,
        error: {
          code: -32603,
          message: `HTTP ${response.status}: ${response.statusText}`
        }
      };
    }

    const result = await response.json();
    return result;
  } catch (error) {
    log(`Request failed: ${error.message}`);
    return {
      jsonrpc: '2.0',
      id: request.id,
      error: {
        code: -32603,
        message: `Request failed: ${error.message}`
      }
    };
  }
}

let pendingRequests = 0;

rl.on('line', async (line) => {
  try {
    const request = JSON.parse(line);

    if (!request.id && request.id !== 0) {
      log(`Received notification: ${request.method}`);
      return;
    }

    log(`Received: ${request.method} (id: ${request.id})`);

    pendingRequests++;
    const response = await proxyRequest(request);

    process.stdout.write(JSON.stringify(response) + '\n');
    log(`Sent response for id: ${request.id}`);

    pendingRequests--;
    if (pendingRequests === 0 && rl.closed) {
      process.exit(0);
    }
  } catch (error) {
    log(`Error processing line: ${error.message}`);
    pendingRequests--;
  }
});

rl.on('close', () => {
  log('Input closed, waiting for pending requests...');
  if (pendingRequests === 0) {
    log('Bridge closed');
    process.exit(0);
  }
});

log(`Bridge started, proxying to ${WORKER_URL}`);
