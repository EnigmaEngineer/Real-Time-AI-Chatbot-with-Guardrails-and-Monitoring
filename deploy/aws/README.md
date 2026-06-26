# AWS App Runner deployment

A one-command deploy of the chatbot platform to AWS App Runner. App Runner is
the closest AWS equivalent to Cloud Run: managed, auto-scales, gives you HTTPS
for free.

## Prereqs

- AWS account with billing enabled
- AWS CLI v2 installed (`aws --version`)
- Docker Desktop running
- `aws configure` has been run with a user that can create ECR repos, IAM
  roles, and App Runner services

## Set a billing alarm BEFORE deploying

App Runner does not scale to zero. There is always at least one instance
running. For a public demo, a budget alarm is essential:

```bash
aws budgets create-budget \
    --account-id $(aws sts get-caller-identity --query Account --output text) \
    --budget '{
        "BudgetName": "chatbot-demo",
        "BudgetLimit": {"Amount": "20", "Unit": "USD"},
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST"
    }' \
    --notifications-with-subscribers '[{
        "Notification": {
            "NotificationType": "ACTUAL",
            "ComparisonOperator": "GREATER_THAN",
            "Threshold": 50,
            "ThresholdType": "PERCENTAGE"
        },
        "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "YOUR_EMAIL@example.com"}]
    }]'
```

This sends an email when 50% of the $20 monthly budget is hit. Adjust the
email and the amount to taste.

## Deploy

```bash
./deploy/aws/deploy.sh
```

What it does:

1. Creates an ECR repository named `chatbot-platform`
2. Builds the Docker image locally (5-15 minutes the first time)
3. Pushes the image to ECR
4. Creates an IAM role so App Runner can pull from ECR
5. Creates or updates the App Runner service
6. Waits for it to reach `RUNNING` (3-5 minutes)
7. Prints the URL plus copy-paste curl commands

## Cost expectations

- 1 vCPU + 2 GB memory: about $0.064/hour while running
- Idle (no requests): about $0.007/hour
- Realistic 7-day public demo cost: $5-15 total
- Free tier: 100 vCPU-minutes/month, not enough to be free

## Settings the script applies

| Setting                   | Value         | Why                              |
|---------------------------|---------------|----------------------------------|
| CPU / Memory              | 1 vCPU / 2 GB | Enough for Detoxify + spaCy      |
| Auto-scaling max          | (default 25)  | Override via AWS console if needed |
| `LLM_MOCK_MODE`           | true          | Zero per-request LLM cost        |
| `RATE_LIMIT_ENABLED`      | true          | Per-API-key RPM limits on        |
| `PORT`                    | 8000          | Matches the Dockerfile EXPOSE    |
| Health check              | `/health`     | Used by App Runner readiness     |

## Tear down

The deploy script prints the exact command. The short version:

```bash
aws apprunner delete-service \
    --service-arn $(aws apprunner list-services --region us-east-1 \
        --query "ServiceSummaryList[?ServiceName=='chatbot-platform'].ServiceArn | [0]" \
        --output text) \
    --region us-east-1
```

Delete the ECR repo too if you don't need the image anymore:

```bash
aws ecr delete-repository --repository-name chatbot-platform --region us-east-1 --force
```

## Locking it down later

If you want to keep the service running but stop accepting public traffic,
set an API key on the running service:

```bash
aws apprunner update-service \
    --service-arn YOUR_SERVICE_ARN \
    --source-configuration '{
        "ImageRepository": {
            "ImageIdentifier": "YOUR_IMAGE_URI",
            "ImageRepositoryType": "ECR",
            "ImageConfiguration": {
                "Port": "8000",
                "RuntimeEnvironmentVariables": {
                    "LLM_MOCK_MODE": "true",
                    "CHATBOT_API_KEY": "...long random string..."
                }
            }
        }
    }'
```
