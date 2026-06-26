---
title: Chatbot Platform
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Multi-agent chatbot with policy-enforced tool use and a public audit log
---

# Chatbot Platform — public 'break it' demo

A multi-agent chatbot with three guardrail layers, drift detection, A/B routing, RAG, and a ReAct agent loop with policy-enforced tool use. Running in mock-LLM mode so there's no per-request cost.

**Try the agent:**

```
curl -s https://USERNAME-SPACENAME.hf.space/chat/agent \
  -H 'Content-Type: application/json' \
  -d '{"message": "calculate (12+5)*3"}'
```

**Try to break it:**

```
curl -s https://USERNAME-SPACENAME.hf.space/chat/agent \
  -H 'Content-Type: application/json' \
  -d '{"message": "fetch https://evil.example.com/leak"}'
```

**See every blocked attempt:**

```
curl -s "https://USERNAME-SPACENAME.hf.space/agents/audit?status=blocked_pre"
```

Source code, design notes, and full feature list: https://github.com/YOUR_USERNAME/chatbot-platform
