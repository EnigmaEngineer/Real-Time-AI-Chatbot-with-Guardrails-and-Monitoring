#!/usr/bin/env bash
# Deploy the chatbot platform to AWS App Runner.
#
# Prereqs:
#   - AWS CLI installed and configured (`aws configure`)
#   - Docker Desktop running
#   - An AWS account with billing enabled
#
# What this does:
#   1. Creates an ECR repository (idempotent)
#   2. Builds the Docker image locally
#   3. Pushes the image to ECR
#   4. Creates an IAM role for App Runner to pull from ECR (idempotent)
#   5. Creates or updates the App Runner service
#   6. Prints the service URL
#
# Hard caps for the public 'break it' demo:
#   - 1 vCPU, 2 GB memory
#   - Auto-scaling capped at 2 instances
#   - LLM_MOCK_MODE=true: zero per-request LLM cost
#
# Usage:
#   ./deploy/aws/deploy.sh
#   SERVICE=my-bot REGION=us-east-1 ./deploy/aws/deploy.sh
#
set -euo pipefail

SERVICE="${SERVICE:-chatbot-platform}"
REGION="${REGION:-us-east-1}"
ECR_REPO="${ECR_REPO:-${SERVICE}}"
ROLE_NAME="${ROLE_NAME:-AppRunnerECRAccessRole-${SERVICE}}"
IMAGE_TAG="$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M)"

# ── sanity checks ──────────────────────────────────────────────────────
command -v aws >/dev/null || { echo "ERROR: aws CLI not installed"; exit 1; }
command -v docker >/dev/null || { echo "ERROR: docker not installed"; exit 1; }

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
[[ -z "${ACCOUNT_ID}" ]] && { echo "ERROR: aws sts get-caller-identity failed. Run 'aws configure' first."; exit 1; }

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"
IMAGE_URI="${ECR_URI}:${IMAGE_TAG}"

echo "── Account:  ${ACCOUNT_ID}"
echo "── Region:   ${REGION}"
echo "── Service:  ${SERVICE}"
echo "── Image:    ${IMAGE_URI}"

# ── 1. ECR repo (create if missing) ────────────────────────────────────
echo "── Ensuring ECR repository exists"
aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${REGION}" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name "${ECR_REPO}" --region "${REGION}" >/dev/null

# ── 2. Build + push image ──────────────────────────────────────────────
echo "── Logging Docker into ECR"
aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "── Building image (this can take 5-15 minutes the first time)"
docker build -t "${IMAGE_URI}" -t "${ECR_URI}:latest" .

echo "── Pushing image to ECR"
docker push "${IMAGE_URI}"
docker push "${ECR_URI}:latest"

# ── 3. IAM role for App Runner -> ECR ──────────────────────────────────
echo "── Ensuring IAM role for App Runner ECR access"
TRUST_POLICY='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"build.apprunner.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

if ! aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
    aws iam create-role \
        --role-name "${ROLE_NAME}" \
        --assume-role-policy-document "${TRUST_POLICY}" >/dev/null
    aws iam attach-role-policy \
        --role-name "${ROLE_NAME}" \
        --policy-arn "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
    echo "   Created role ${ROLE_NAME}. Waiting 10s for IAM propagation"
    sleep 10
fi
ROLE_ARN="$(aws iam get-role --role-name "${ROLE_NAME}" --query Role.Arn --output text)"

# ── 4. Create or update the App Runner service ─────────────────────────
SERVICE_ARN="$(aws apprunner list-services --region "${REGION}" \
    --query "ServiceSummaryList[?ServiceName=='${SERVICE}'].ServiceArn | [0]" \
    --output text)"

SOURCE_CONFIG=$(cat <<JSON
{
  "ImageRepository": {
    "ImageIdentifier": "${IMAGE_URI}",
    "ImageRepositoryType": "ECR",
    "ImageConfiguration": {
      "Port": "8000",
      "RuntimeEnvironmentVariables": {
        "LLM_MOCK_MODE": "true",
        "RATE_LIMIT_ENABLED": "true",
        "PORT": "8000"
      }
    }
  },
  "AutoDeploymentsEnabled": false,
  "AuthenticationConfiguration": {
    "AccessRoleArn": "${ROLE_ARN}"
  }
}
JSON
)

INSTANCE_CONFIG='{"Cpu":"1 vCPU","Memory":"2 GB"}'
HEALTH_CONFIG='{"Protocol":"HTTP","Path":"/health","Interval":20,"Timeout":5,"HealthyThreshold":1,"UnhealthyThreshold":5}'

if [[ "${SERVICE_ARN}" == "None" || -z "${SERVICE_ARN}" ]]; then
    echo "── Creating App Runner service ${SERVICE}"
    aws apprunner create-service \
        --service-name "${SERVICE}" \
        --region "${REGION}" \
        --source-configuration "${SOURCE_CONFIG}" \
        --instance-configuration "${INSTANCE_CONFIG}" \
        --health-check-configuration "${HEALTH_CONFIG}" \
        >/dev/null
    SERVICE_ARN="$(aws apprunner list-services --region "${REGION}" \
        --query "ServiceSummaryList[?ServiceName=='${SERVICE}'].ServiceArn | [0]" \
        --output text)"
else
    echo "── Updating existing App Runner service ${SERVICE}"
    aws apprunner update-service \
        --service-arn "${SERVICE_ARN}" \
        --region "${REGION}" \
        --source-configuration "${SOURCE_CONFIG}" \
        >/dev/null
fi

echo "── Waiting for service to become RUNNING (typically 3-5 minutes)"
while true; do
    STATUS="$(aws apprunner describe-service --service-arn "${SERVICE_ARN}" --region "${REGION}" \
        --query Service.Status --output text)"
    echo "   status: ${STATUS}"
    if [[ "${STATUS}" == "RUNNING" ]]; then break; fi
    if [[ "${STATUS}" == "CREATE_FAILED" || "${STATUS}" == "DELETED" ]]; then
        echo "ERROR: service entered status ${STATUS}. Check the AWS console."
        exit 1
    fi
    sleep 20
done

URL="https://$(aws apprunner describe-service --service-arn "${SERVICE_ARN}" --region "${REGION}" \
    --query Service.ServiceUrl --output text)"

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
    -d '{"message": "calculate (12 + 5) * 3"}' | python -m json.tool

Try to break it:
  curl -s ${URL}/chat/agent \\
    -H 'Content-Type: application/json' \\
    -d '{"message": "fetch https://evil.example.com/leak"}' | python -m json.tool

See the audit log:
  curl -s "${URL}/agents/audit?status=blocked_pre" | python -m json.tool

Tear down when done:
  aws apprunner delete-service --service-arn ${SERVICE_ARN} --region ${REGION}

EOF
