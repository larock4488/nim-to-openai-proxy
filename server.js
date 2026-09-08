// server.js — Robust Hybrid OpenAI ↔ NIM / OpenRouter Proxy
// Express 5 Compatible
// Fixes: OpenRouter reasoning parsing (reasoning, reasoning_content, reasoning_details), auth bypass

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual } = require('crypto');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Configuration ───────────────────────────────────────────────────────────

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const OPENROUTER_API_BASE = process.env.OPENROUTER_API_BASE || 'https://openrouter.ai/api/v1';
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;

const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;

const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';

const MAX_TOKENS_LIMIT = 65536;
const REQUEST_TIMEOUT_MS = 540000; // 9 Minute
const MAX_BUFFER_SIZE = 1024 * 1024; // 1MB

if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');
if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');

// ─── Config validation ──────────────────────────────────────────────────────

function validateConfig() {
  const fatal = (msg) => { console.error(`[FATAL] ${msg}`); process.exit(1); };
  
  if (!NIM_API_KEY) fatal('NIM_API_KEY is required. Get one at https://build.nvidia.com/');
  
  if (!OPENROUTER_API_KEY) {
    console.warn('[WARN] OPENROUTER_API_KEY not set. OpenRouter models will fail.');
  }

  if (!CLIENT_AUTH_KEY) {
    console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected with 403.');
  }
}

validateConfig();

// ─── Model Mapping ─────────────────────────────────────────────────────────

const MODEL_MAPPING = {
  'nemotron-3-super-120b-a12b': 'nvidia/nemotron-3-super-120b-a12b',
  'nemotron-3-ultra-550b-a55b': 'nvidia/nemotron-3-ultra-550b-a55b',
  'nemotron-3.5-lightning-30b-a3b': 'nvidia/nemotron-3.5-lightning-30b-a3b',
  'kimi-k3': 'moonshotai/kimi-k3',
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro',
  'deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash',
  'deepseek-v4-flash-0731': 'deepseek-ai/deepseek-v4-flash-0731',
  'deepseek-v4-pro-0813': 'deepseek-ai/deepseek-v4-pro-0813',
  'gpt-oss-120b': 'openai/gpt-oss-120b',
  'llama-3.3-70b-instruct': 'meta/llama-3.3-70b-instruct',
  'glm-5.2': 'z-ai/glm-5.2',
  'openrouter/glm-5.2': 'z-ai/glm-5.2:free',
  'openrouter/minimax-m3': 'minimax/minimax-m3:free',
  'mistral-nemotron': 'mistralai/mistral-nemotron',
  'gemma-4-31b-it': 'google/gemma-4-31b-it',
  'minimax-m3': 'minimaxai/minimax-m3',
  'step-3.7-flash': 'stepfun-ai/step-3.7-flash'
};

// ─── Middleware ─────────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json({ limit: '10mb' }));

function extractBearerToken(authHeader) {
  if (!authHeader || typeof authHeader !== 'string') return null;
  const parts = authHeader.trim().split(' ');
  if (parts.length !== 2 || parts[0] !== 'Bearer') return null;
  return parts[1];
}

function safeTimingEqual(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  try {
    return timingSafeEqual(Buffer.from(a), Buffer.from(b));
  } catch {
    return false;
  }
}

app.use((req, res, next) => {
  if (req.path === '/health' || req.path === '/v1/models') {
    return next();
  }

  const token = extractBearerToken(req.headers.authorization);
  
  if (!token || !CLIENT_AUTH_KEY) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid or missing authentication',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  if (!safeTimingEqual(token, CLIENT_AUTH_KEY)) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid authentication credentials',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  next();
});

// ─── Helper: Safe Stream Writing ───────────────────────────────────────────

function safeWrite(res, data) {
  try {
    if (!res.writableEnded && !res.destroyed && res.writable) {
      res.write(data);
      return true;
    }
  } catch (err) {
    console.warn('[STREAM] Write failed:', err.message);
  }
  return false;
}

// ─── Helper: Send Upstream Request ─────────────────────────────────────────

async function callUpstreamModel(baseRequest, model, apiBase, apiKey) {
  return await axios.post(
    `${apiBase}/chat/completions`,
    { ...baseRequest, model },
    {
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      responseType: baseRequest.stream ? 'stream' : 'json',
      timeout: REQUEST_TIMEOUT_MS
    }
  );
}

// ─── Routes ────────────────────────────────────────────────────────────────

app.get('/health', (req, res) => {
  res.json({ status: 'ok', version: '2.2.3' });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id,
      object: 'model',
      created: Date.now(),
      owned_by: id.startsWith('openrouter') ? 'openrouter' : 'nim-proxy'
    }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  let startTime = Date.now();
  let streamEndedCleanly = false;
  let upstreamStream = null;

  try {
    // Note: ...restBody safely packs user settings like temperature, top_p, etc.
    const { model, messages, temperature, max_tokens, stream, chat_template_kwargs, ...restBody } = req.body;

    const targetModel = MODEL_MAPPING[model];
    if (!targetModel) {
      return res.status(400).json({
        error: {
          message: `Model '${model || 'undefined'}' is not supported.`,
          type: 'invalid_request_error',
          code: 400
        }
      });
    }

    let currentApiBase = NIM_API_BASE;
    let currentApiKey = NIM_API_KEY;
    let providerName = 'NIM';

    const isExplicitOpenRouter = model.startsWith('openrouter/');
    const isKnownOpenRouterModel = targetModel && (
      targetModel.startsWith('minimax/') ||
      targetModel.startsWith('minimaxai/') ||
      targetModel.startsWith('deepseek-ai/') ||
      targetModel.startsWith('moonshotai/') ||
      targetModel.startsWith('z-ai/') ||
      targetModel.startsWith('openai/') ||
      targetModel.startsWith('stepfun-ai/')
    );

    if (isExplicitOpenRouter || isKnownOpenRouterModel) {
      currentApiBase = OPENROUTER_API_BASE;
      currentApiKey = OPENROUTER_API_KEY;
      providerName = 'OpenRouter';

      if (!currentApiKey) {
        return res.status(500).json({
          error: {
            message: 'OpenRouter API key is not configured.',
            type: 'server_error',
            code: 500
          }
        });
      }
    }

    const isDeepSeekV4 = targetModel.includes('deepseek-v4') || targetModel.includes('deepseek-r1');
    const isGLM52 = targetModel.includes('glm-5.2');
    const isMiniMaxM3 = targetModel.includes('minimax-m3');
    const isKimiK3 = targetModel.includes('kimi-k3');
    const isMonitoredModel = isDeepSeekV4 || isGLM52 || isMiniMaxM3 || isKimiK3;

    const cleanedMessages = messages.map(msg => {
      if (msg.role === 'assistant' && typeof msg.content === 'string') {
        return {
          ...msg,
          content: msg.content.replace(/<thinking>[\s\S]*?<\/thinking>\n*/g, '').trim()
        };
      }
      return msg;
    });

    // baseRequest combines user settings from restBody with required proxy defaults
    const baseRequest = {
      ...restBody,
      messages: cleanedMessages,
      model: targetModel,
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(max_tokens ?? 10000, MAX_TOKENS_LIMIT),
      stream: stream || false,
      ...(stream ? { stream_options: { include_usage: true } } : {})
    };

    if (ENABLE_THINKING_MODE) {
      if (providerName === 'NIM') {
        if (isKimiK3) baseRequest.reasoning_effort = "high";
        else if (isDeepSeekV4 || isGLM52) baseRequest.reasoning_effort = "medium";

        if (isGLM52) {
          baseRequest.chat_template_kwargs = { enable_thinking: true, thinking: true };
        } else if (isMiniMaxM3) {
          baseRequest.chat_template_kwargs = { thinking_mode: "enabled" };
        } else if (isKimiK3) {
          baseRequest.chat_template_kwargs = { enable_thinking: true };
        } else {
          baseRequest.chat_template_kwargs = { thinking: true };
        }
      } else if (providerName === 'OpenRouter') {
        // Tells OpenRouter to output reasoning tokens for supported models
        baseRequest.reasoning = { enabled: true, effort: 'medium' }; // Options: 'max', 'xhigh', 'high', 'medium', 'low', 'minimal'
      }
    }

    const response = await callUpstreamModel(baseRequest, targetModel, currentApiBase, currentApiKey);
    upstreamStream = response.data;
    console.log(`[PROXY] Routed to ${providerName}. Model used: ${targetModel}`);

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const decoder = new StringDecoder('utf8');
      let buffer = '';
      let reasoningOpen = false;
      let doneSent = false;
      let cleanedUp = false;

      const cleanup = () => {
        if (cleanedUp) return;
        cleanedUp = true;
        if (upstreamStream) {
          upstreamStream.removeAllListeners();
        }
        req.removeAllListeners('close');
      };

      const processLine = (line) => {
        if (!line.startsWith('data: ')) return;

        if (line.includes('[DONE]')) {
          if (!doneSent) {
            safeWrite(res, 'data: [DONE]\n\n');
            doneSent = true;
          }
          streamEndedCleanly = true;
          return;
        }

        try {
          const data = JSON.parse(line.slice(6));

          if (isMonitoredModel && data.usage) {
            console.log(`[TOKEN USAGE] Provider: ${providerName} | Model: ${model} (${targetModel})`);
            console.log(`  - Prompt Tokens: ${data.usage.prompt_tokens ?? 0}`);
            console.log(`  - Completion Tokens: ${data.usage.completion_tokens ?? 0}`);
            console.log(`  - Total Tokens: ${data.usage.total_tokens ?? 0}`);
          }

          const delta = data.choices?.[0]?.delta;

          if (delta) {
            // Extract reasoning from any provider format
            let chunkReasoning = delta.reasoning_content || delta.reasoning;
            if (!chunkReasoning && delta.reasoning_details && Array.isArray(delta.reasoning_details)) {
              chunkReasoning = delta.reasoning_details.map(d => d.text || d.summary || '').join('');
            }

            let content = delta.content || '';

            if (SHOW_REASONING && chunkReasoning) {
              if (!reasoningOpen) {
                content = `<thinking>\n${chunkReasoning}`;
                reasoningOpen = true;
              } else {
                content = chunkReasoning;
              }
            } else if (SHOW_REASONING && reasoningOpen && !chunkReasoning) {
              // Reasoning has finished, close the tag and append normal content
              content = `\n</thinking>\n\n${content}`;
              reasoningOpen = false;
            }

            delta.content = content;
            
            // Clean up upstream fields so clients don't crash on unhandled keys
            delete delta.reasoning_content;
            delete delta.reasoning;
            delete delta.reasoning_details;
          }

          safeWrite(res, `data: ${JSON.stringify(data)}\n\n`);

        } catch (parseErr) {
          console.warn('[STREAM] Invalid JSON line:', line.slice(0, 100));
        }
      };

      upstreamStream.on('data', chunk => {
        buffer += decoder.write(chunk);

        if (buffer.length > MAX_BUFFER_SIZE) {
          console.error('[STREAM] Buffer overflow, destroying connection');
          safeWrite(res, `data: ${JSON.stringify({ error: { message: 'Stream buffer overflow', type: 'stream_error' } })}\n\n`);
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
          upstreamStream.destroy();
          cleanup();
          return;
        }

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          processLine(line);
        }
      });

      upstreamStream.on('end', () => {
        buffer += decoder.end();
        if (buffer.trim()) {
          for (const line of buffer.split('\n')) {
            processLine(line);
          }
        }

        if (!doneSent) { safeWrite(res, 'data: [DONE]\n\n'); }
        streamEndedCleanly = true;
        if (!res.writableEnded) res.end();
        cleanup();
      });

      upstreamStream.on('error', err => {
        console.error('[STREAM] Upstream error:', err.message);
        if (!res.writableEnded) {
          safeWrite(res, `data: ${JSON.stringify({ error: { message: 'Stream interrupted', type: 'stream_error' } })}\n\n`);
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
        }
        cleanup();
      });

      req.on('close', () => {
        if (!streamEndedCleanly && (req.destroyed || !res.writable)) {
          console.warn('[STREAM] Client disconnected prematurely');
        }
        if (upstreamStream && !upstreamStream.destroyed && !streamEndedCleanly) {
          upstreamStream.destroy();
        }
        cleanup();
      });

    } else {
      const usage = response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };

      if (isMonitoredModel) {
        console.log(`[TOKEN USAGE] Provider: ${providerName} | Model: ${model} (${targetModel})`);
        console.log(`  - Prompt Tokens: ${usage.prompt_tokens}`);
        console.log(`  - Completion Tokens: ${usage.completion_tokens}`);
      }

      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: (response.data.choices || []).map((choice, i) => {
          let content = choice.message?.content || '';
          
          let reasoning = choice.message?.reasoning_content || choice.message?.reasoning;
          if (!reasoning && choice.message?.reasoning_details && Array.isArray(choice.message.reasoning_details)) {
            reasoning = choice.message.reasoning_details.map(d => d.text || d.summary || '').join('');
          }

          if (SHOW_REASONING && reasoning) {
            const safeReasoning = reasoning.replace(/\n/g, '\\n');
            content = `<thinking>\n${safeReasoning}\n</thinking>\n\n${content}`;
          }
          
          if (choice.message) {
             delete choice.message.reasoning_content;
             delete choice.message.reasoning;
             delete choice.message.reasoning_details;
          }

          return {
            index: i,
            message: { role: choice.message?.role || 'assistant', content, tool_calls: choice.message?.tool_calls },
            finish_reason: choice.finish_reason || 'stop'
          };
        }),
        usage
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('[PROXY] Fatal error:', error.message);
    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: { message: error.message, type: 'invalid_request_error', code: error.response?.status || 500 }
      });
    } else if (!res.writableEnded) {
      safeWrite(res, `data: ${JSON.stringify({ error: { message: error.message, type: 'proxy_error' } })}\n\n`);
      safeWrite(res, 'data: [DONE]\n\n');
      res.end();
    }
    if (upstreamStream && !upstreamStream.destroyed) upstreamStream.destroy();
  }
});

app.use((req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.method} ${req.path} not found`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`[PROXY] Hybrid proxy running on port ${PORT}`);
});
