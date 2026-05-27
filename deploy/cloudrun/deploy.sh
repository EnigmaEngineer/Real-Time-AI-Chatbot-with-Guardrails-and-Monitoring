#!/usr/bin/env bash
# Deploy the chatbot platform to Google Cloud Run.
#
# Prereqs:
#   - gcloud SDK installed and authenticated (`gcloud auth login`)
#   - a GCP project selected (`gcloud config set project YOUR_PROJECT`)
#   - billing enabled on that project (Cloud Run free tier covers ~2M req/mo)
#
# Hard caps applied (specifically chosen for a public 'break it' demo):
#   - max-instances=2      → ceiling on scale-out; protects your wallet
#   - timeout=60s          → per-request timeout
#   - memory=2Gi, cpu=2    → enough for Detoxify + spaCy in mock-LLM mode
#   - LLM_MOCK_MODE=true   → no external API spend, agent loop still works
#   - CHATBOT_API_KEY unset → endpoint is public (so people can try to break it)
#                            change to `--set-env-vars CHATBOT_API_KEY=...` for auth
#
# Cost cap (recommended before running this):
#   gcloud billing budgets create \
#     --billing-account=YOUR_BILLING_ID \
#     --display-name="chatbot-demo" \
#     --budget-amount=5USD \
#     --threshold-rule=percent=0.5 \
#     --threshold-rule=percent=0.9
#
# Usage:
#   ./deploy/cloudrun/deploy.sh                    # uses defaults below
#   SERVICE=my-bot REGION=us-east1 ./deploy/cloudrun/deploy.sh
#
set -euo pipefail

SERVICE="${SERVICE:-chatbot-platform}"
REGION="${REGION:-us-central1}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"

if [[ -z "${PROJECT}" ]]; then
    echo "ERROR: no GCP project set. Run: gcloud config set project YOUR_PROJECT" >&2
    exit 1
fi

IMAGE="gcr.io/${PROJECT}/${SERVICE}:$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M)"

echo "── Building image: ${IMAGE}"
gcloud builds submit --tag "${IMAGE}" --project "${PROJECT}"

echo "── Deploying to Cloud Run: ${SERVICE} (region=${REGION})"
gcloud run deploy "${SERVICE}" \
    --image "${IMAGE}" \
    --region "${REGION}" \
    --project "${PROJECT}" \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 60 \
    --max-instances 2 \
    --min-instances 0 \
    --concurrency 20 \
    --set-env-vars "LLM_MOCK_MODE=true,RATE_LIMIT_ENABLED=true" \
    --port 8080

URL=$(gcloud run services describe "${SERVICE}" --region "${REGION}" --project "${PROJECT}" --format='value(status.url)')

cat <<EOF

── DEPLOYED ──
Service URL:   ${URL}
Health:        ${URL}/health
Agent docs:    ${URL}/docs
Tools:         ${URL}/agents/tools?profile=default
Audit log:     ${URL}/agents/audit
Metrics:       ${URL}/metrics

Try the agent:
  curl -s ${URL}/chat/agent \\
    -H 'Content-Type: application/json' \\
    -d '{"message": "calculate (12 + 5) * 3"}' | python3 -m json.tool

Try to break it:
  curl -s ${URL}/chat/agent \\
    -H 'Content-Type: application/json' \\
    -d '{"message": "fetch https://evil.example.com/leak"}' | python3 -m json.tool

Then view the policy block:
  curl -s ${URL}/agents/audit?status=blocked_pre | python3 -m json.tool

EOF
