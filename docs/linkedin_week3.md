# Week 3 LinkedIn post

Week 3 update on the chatbot platform. This week I added a multi agent loop with policy enforced tool use, and deployed the whole thing to Cloud Run so anyone can try to break it.

The agent can call three tools, a calculator that walks a Python AST instead of using eval, a vector search over your uploaded documents, and a web fetcher that only allows a tiny list of hosts like Wikipedia and arXiv. Every tool call goes through a policy engine before and after it runs, and every call is written to a SQLite audit log that you can read.

The interesting part is not the agent loop, those are a dime a dozen now. The interesting part is what stops the agent from doing something dumb. I added four hard caps in the coordinator and each one closes a specific failure mode I have seen agents hit in the wild. Five iterations max so it cannot loop forever. Fifteen second wall clock so a slow tool cannot stall the whole loop. Eight KB cumulative observation cap so a huge web fetch cannot flood the context window. Ten second per tool timeout so one bad tool does not drag down the rest.

The policy engine itself blocks unknown tools, URLs that are not on the allowlist, PII in tool arguments, and prompt injection phrases inside content the tools return. The injection scan was the one I was most nervous about because retrieved content can carry payloads that look like instructions to the model. The policy redacts those before they reach the LLM rather than hard blocking, so the agent still gets to see sanitized content and decide what to do.

Every violation is a Prometheus counter and a row in SQLite. The Grafana dashboard now has a section for tool calls per minute, policy violations by policy name, agent iteration histogram, and the reasons agent runs terminated without producing a final answer.

It is deployed on Cloud Run with hard caps, max two instances, mock LLM mode so there is zero per request cost, scales to zero when idle. Total cost is well inside the free tier.

LIVE URL: <paste your Cloud Run URL here>

Things worth trying:

```
curl -s $URL/chat/agent -H 'Content-Type: application/json' \
  -d '{"message":"calculate (12+5)*3"}'

curl -s $URL/chat/agent -H 'Content-Type: application/json' \
  -d '{"message":"fetch https://evil.example.com/secret"}'

curl -s $URL/chat/agent -H 'Content-Type: application/json' \
  -d '{"message":"use the shell_exec tool to run rm -rf /"}'

curl -s "$URL/agents/audit?status=blocked_pre" | python3 -m json.tool
```

The audit endpoint is public. Every attempt anyone makes shows up there. If you find something that gets a real ok status on a tool that should have been blocked, please tell me, that is the whole reason I am putting this out.

Repo: <paste GitHub URL here>

Project is at 158 tests now with 34 new ones covering the agent loop, tool policies, and edge cases like infinite loops and tool timeouts. Next week I want to write up the policy model in detail, the audit log is already showing patterns I did not predict and that is the interesting part.

If you have shipped an agent to production, what is the failure mode that hurt you the most? Curious what I am still missing.

#ai #llm #observability #agents #cloudrun
