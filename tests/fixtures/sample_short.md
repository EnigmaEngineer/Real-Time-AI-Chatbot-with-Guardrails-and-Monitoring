# API Reference

## Authentication

All requests require an `X-API-Key` header. Contact admin for a key.

## Endpoints

### POST /chat/strict

Send a message with strict guardrails.

**Request:**
```json
{"message": "What is Python?", "user_id": "user_123"}
```

**Response:**
```json
{"response": "Python is a programming language.", "confidence": 0.95}
```

### POST /chat/creative

Same as strict but with relaxed guardrails for creative tasks.
