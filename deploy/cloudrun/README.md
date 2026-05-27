# Cloud Run Deployment

A single-command deploy of the chatbot platform to Google Cloud Run, sized for a
public "break it" demo: scale-to-zero when idle, a hard cap on parallel
instances so abuse cannot run up the bill, and `LLM_MOCK_MODE=true` so there
is no per-request LLM cost.

## Prereqs

```bash
# Once
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

## Cost cap first (do this BEFORE deploying a public endpoint)

```bash
gcloud billing budgets create \
    --billing-account=$(gcloud billing accounts list --format='value(name)' --limit=1) \
    --display-name="chatbot-demo" \
    --budget-amount=5USD \
    --threshold-rule=percent=0.5 \
    --threshold-rule=percent=0.9 \
    --threshold-rule=percent=1.0
```

A $5/month cap is overkill for a `LLM_MOCK_MODE=true` deployment (the free
tier covers ~2M requests/mo) but the alert is what you actually want — if it
fires, you'll know something's wrong.

## Deploy

```bash
./deploy/cloudrun/deploy.sh
```

The script prints the service URL plus copy-paste curl commands. Roughly 3
minutes from `git push` to live HTTPS endpoint.

## What the script sets

| Flag                          | Value             | Why                                                |
|-------------------------------|-------------------|----------------------------------------------------|
| `--max-instances`             | 2                 | Wallet protection — at most 2 parallel containers |
| `--min-instances`             | 0                 | Scale to zero when idle (free)                    |
| `--concurrency`               | 20                | Reqs per instance before scaling out               |
| `--memory` / `--cpu`          | 2Gi / 2           | Enough for Detoxify + spaCy on cold start          |
| `--timeout`                   | 60s               | Per-request timeout                                |
| `--allow-unauthenticated`     | (yes)             | Public endpoint for the LinkedIn demo              |
| `LLM_MOCK_MODE=true`          | env               | Zero per-request cost                              |
| `RATE_LIMIT_ENABLED=true`     | env               | Per-API-key RPM limiter active                     |

## What the deployed endpoint exposes

- `POST /chat/agent` — the ReAct loop with policy-enforced tool use
- `GET /agents/tools?profile=default` — what tools are available
- `GET /agents/audit` — every tool call, blocked or not, in the last 24h
- `GET /metrics` — Prometheus scrape endpoint (point an external Grafana at it)
- `GET /docs` — interactive API explorer

## Locking it down later

If you want to keep the deployment up but stop accepting public traffic:

```bash
# Require an API key (set as env on the service)
gcloud run services update chatbot-platform \
    --region us-central1 \
    --set-env-vars CHATBOT_API_KEY=$(openssl rand -hex 24)

# Or pull the public access permission entirely
gcloud run services remove-iam-policy-binding chatbot-platform \
    --region us-central1 \
    --member=allUsers --role=roles/run.invoker
```

## Tear down

```bash
gcloud run services delete chatbot-platform --region us-central1
```
