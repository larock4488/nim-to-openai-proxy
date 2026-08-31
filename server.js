// server.js — Robust Hybrid OpenAI ↔ NIM / OpenRouter Proxy
// Express 5 Compatible
// Fixes: auth bypass, startup DDoS, silent stream failures, memory leaks, Express 5 deprecations

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
  'openrouter/glm-5.2': 'z-ai/glm-5.2:free', // OpenRouter Mapping added here
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
  res.json({ status: 'ok', version: '2.2.0' });
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
    // Destructure model and messages, capture everything else in restBody
    const { model, messages, temperature, max_tokens, stream, ...restBody } = req.body;

    // Reject immediately with a clear error if the model isn't mapped
    const targetModel = MODEL_MAPPING[model];
    if (!targetModel) {
      return res.status(400).json({
        error: {
          message: `Model '${model || 'undefined'}' is not supported. Please select an available model.`,
          type: 'invalid_request_error',
          code: 400
        }
      });
    }

    // Dynamic routing logic: NIM vs OpenRouter
    let currentApiBase = NIM_API_BASE;
    let currentApiKey = NIM_API_KEY;
    let providerName = 'NIM';

    if (model === 'openrouter/glm-5.2') {
      currentApiBase = OPENROUTER_API_BASE;
      currentApiKey = OPENROUTER_API_KEY;
      providerName = 'OpenRouter';

      if (!currentApiKey) {
        return res.status(500).json({
          error: {
            message: 'OpenRouter API key is not configured on the server.',
            type: 'server_error',
            code: 500
          }
        });
      }
    }

    // Check target models for specialized logging and reasoning behavior
    const isDeepSeekV4 = targetModel.includes('deepseek-v4');
    const isGLM52 = targetModel.includes('glm-5.2');
    const isMiniMaxM3 = targetModel.includes('minimax-m3');
    const isKimiK3 = targetModel.includes('kimi-k3');
    const isMonitoredModel = isDeepSeekV4 || isGLM52 || isMiniMaxM3 || isKimiK3;

    const baseRequest = {
      ...restBody, // Passes frequency_penalty, presence_penalty, top_p, tools, seed, etc. automatically!
      messages,
      model: targetModel,
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(max_tokens ?? 10000, MAX_TOKENS_LIMIT),
      stream: stream || false,
      
      // Include usage stats in stream responses
      ...(stream ? { stream_options: { include_usage: true } } : {}),

      // Keep root-level reasoning_effort ONLY for models that actually use it (DeepSeek & GLM)
      ...(ENABLE_THINKING_MODE && (isDeepSeekV4 || isGLM52) ? { reasoning_effort: "medium" } : {}),
      
      // Pass chat_template_kwargs for MiniMax-M3, Kimi-K3, and GLM-5.2
      ...(ENABLE_THINKING_MODE 
        ? ((isGLM52 || isKimiK3)
            ? { chat_template_kwargs: { enable_thinking: true, thinking: true } }
            : (isMiniMaxM3 
                ? { chat_template_kwargs: { thinking_mode: "enabled" } } 
                : { chat_template_kwargs: { thinking: true } }))
        : {})
    };

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

          // Log token usage for monitored models when usage chunk is emitted
          if (isMonitoredModel && data.usage) {
            console.log(`[TOKEN USAGE] Provider: ${providerName} | Model: ${model} (${targetModel})`);
            console.log(`  - Prompt Tokens: ${data.usage.prompt_tokens ?? 0}`);
            console.log(`  - Completion Tokens: ${data.usage.completion_tokens ?? 0}`);
            console.log(`  - Total Tokens: ${data.usage.total_tokens ?? 0}`);
            console.log(`- Total seconds taken ${Math.floor((Date.now() - startTime) / 1000)}`);
          }

          const delta = data.choices?.[0]?.delta;

          if (delta) {
            let content = delta.content || '';
            const reasoning = delta.reasoning_content;

            if (SHOW_REASONING) {
              if (reasoning && !reasoningOpen) {
                content = `<thinking>\n${reasoning.replace(/\n/g, '\\n')}`;
                reasoningOpen = true;
              } else if (reasoning) {
                content = reasoning.replace(/\n/g, '\\n');
              }

              if (delta.content && reasoningOpen) {
                content += `\n</thinking>\n\n${delta.content}`;
                reasoningOpen = false;
              }
            }

            delta.content = content;
            delete delta.reasoning_content;
          }

          safeWrite(res, `data: ${JSON.stringify(data)}\n\n`);

        } catch (parseErr) {
          console.warn('[STREAM] Invalid JSON line:', line.slice(0, 100));
          safeWrite(res, `data: ${JSON.stringify({ 
            error: { 
              message: 'Upstream sent malformed chunk', 
              type: 'stream_parse_error',
              details: line.slice(0, 100)
            } 
          })}\n\n`);
        }
      };

      upstreamStream.on('data', chunk => {
        buffer += decoder.write(chunk);

        if (buffer.length > MAX_BUFFER_SIZE) {
          console.error('[STREAM] Buffer overflow, destroying connection');
          safeWrite(res, `data: ${JSON.stringify({ 
            error: { 
              message: 'Stream buffer overflow', 
              type: 'stream_error' 
            } 
          })}\n\n`);
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

        if (!doneSent) {
          safeWrite(res, 'data: [DONE]\n\n');
        }

        streamEndedCleanly = true;
        if (!res.writableEnded) {
          res.end();
        }
        cleanup();
      });

      upstreamStream.on('error', err => {
        console.error('[STREAM] Upstream error:', err.message);
        
        if (!res.writableEnded) {
          safeWrite(res, `data: ${JSON.stringify({
            error: {
              message: 'Stream interrupted by upstream error',
              type: 'stream_error'
            }
          })}\n\n`);
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
        }
        cleanup();
      });

      req.on('close', () => {
        const clientGone = req.destroyed || !res.writable;
        
        if (!streamEndedCleanly && clientGone) {
          console.warn('[STREAM] Client disconnected prematurely');
        }

        if (upstreamStream && !upstreamStream.destroyed && !streamEndedCleanly) {
          upstreamStream.destroy();
        }
        cleanup();
      });

    } else {
      // Non-streaming response
      const usage = response.data.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      };

      if (isMonitoredModel) {
        console.log(`[TOKEN USAGE] Provider: ${providerName} | Model: ${model} (${targetModel})`);
        console.log(`  - Prompt Tokens: ${usage.prompt_tokens}`);
        console.log(`  - Completion Tokens: ${usage.completion_tokens}`);
        console.log(`  - Total Tokens: ${usage.total_tokens}`);
        console.log(`- Total seconds taken ${Math.floor((Date.now() - startTime) / 1000)}`);
      }

      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: (response.data.choices || []).map((choice, i) => {
          let content = choice.message?.content || '';

          if (SHOW_REASONING && choice.message?.reasoning_content) {
            const safeReasoning = choice.message.reasoning_content.replace(/\n/g, '\\n');
            content = `<thinking>\n${safeReasoning}\n</thinking>\n\n${content}`;
          }

          return {
            index: i,
            message: {
              role: choice.message?.role || 'assistant',
              content,
              tool_calls: choice.message?.tool_calls
            },
            finish_reason: choice.finish_reason || 'stop'
          };
        }),
        usage
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('[PROXY] Fatal error:', error.message);
    console.error('[PROXY] Upstream response:', error.response?.data);

    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: {
          message: error.message,
          type: 'invalid_request_error',
          code: error.response?.status || 500
        }
      });
    } else if (!res.writableEnded) {
      safeWrite(res, `data: ${JSON.stringify({
        error: {
          message: error.message,
          type: 'proxy_error'
        }
      })}\n\n`);
      safeWrite(res, 'data: [DONE]\n\n');
      res.end();
    }

    if (upstreamStream && !upstreamStream.destroyed) {
      upstreamStream.destroy();
    }
  }
});

app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.method} ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// ─── Startup ───────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`[PROXY] Hybrid proxy running on port ${PORT}`);
  console.log(`[PROXY] Max tokens limit: ${MAX_TOKENS_LIMIT}`);
});
