# Hugging Face Spaces deployment

Free hosting with 16 GB RAM, no credit card. Public URL like
`https://USERNAME-chatbot-platform.hf.space`.

## What gets pushed to HF

HF Spaces wants a single git repo with:
- `Dockerfile` (already in our repo)
- `README.md` with HF YAML frontmatter (the one in this folder)
- All the `src/`, `tests/`, etc.

We push the same project code to two remotes: GitHub (the main repo) and
HF Spaces (which builds + runs the container).

## One-time setup

### 1. Create a Hugging Face account

Sign up at https://huggingface.co/join. No credit card.

Go to https://huggingface.co/settings/tokens, click **New token**, give it
write access, copy the token. You'll use it as the git password when pushing.

### 2. Create the Space

Go to https://huggingface.co/new-space.

- **Owner:** your username
- **Space name:** `chatbot-platform` (or whatever)
- **License:** MIT
- **SDK:** Docker
- **Template:** Blank
- **Hardware:** CPU basic (free)
- **Visibility:** Public

Click **Create Space**. You'll land on the Space page with an empty repo.

### 3. Clone the Space repo locally

In PowerShell:

```powershell
cd C:\Users\syeds\OneDrive\Documents\Projects
git clone https://huggingface.co/spaces/YOUR_USERNAME/chatbot-platform hf-chatbot-platform
cd hf-chatbot-platform
```

When git asks for credentials:
- Username: your HF username
- Password: the token from step 1 (NOT your account password)

### 4. Sync the project files in

```powershell
# Copy everything from the chatbot-platform working dir, except .git
robocopy ..\chatbot-platform\chatbot-platform . /E /XD .git data __pycache__ .pytest_cache .ruff_cache /XF *.db

# Replace the GitHub README with the HF-specific one
copy /Y deploy\huggingface\README.md README.md
```

### 5. Push

```powershell
git add .
git commit -m "initial deploy to hugging face spaces"
git push
```

HF will start building immediately. You can watch live build logs at
`https://huggingface.co/spaces/YOUR_USERNAME/chatbot-platform` under the
**Logs** tab. Build takes 10-15 minutes the first time.

## When the build succeeds

The Space page will show a "Running" badge. Your app URL is:

```
https://YOUR_USERNAME-chatbot-platform.hf.space
```

Test it (Git Bash):

```bash
URL="https://YOUR_USERNAME-chatbot-platform.hf.space"

curl -s $URL/health
curl -s $URL/chat/agent -H "Content-Type: application/json" \
  -d '{"message": "calculate (12+5)*3"}'
```

The first request may take 30+ seconds (cold start + Detoxify model download).

## Updating after the first deploy

When you change something in the main repo, push it to the HF Space too:

```powershell
cd C:\Users\syeds\OneDrive\Documents\Projects\hf-chatbot-platform
robocopy ..\chatbot-platform\chatbot-platform . /E /XD .git data __pycache__ .pytest_cache .ruff_cache /XF *.db
copy /Y deploy\huggingface\README.md README.md
git add .
git commit -m "sync"
git push
```

## Notes on HF Spaces specifically

- **Port**: HF sets `$PORT` to whatever `app_port` is in the README frontmatter (7860 by default). Our Dockerfile uses `${PORT:-8000}` so it picks this up automatically.
- **Sleep**: Free Spaces sleep after ~48 hours of inactivity, wake on next request (15-30s cold start).
- **Persistence**: The container has ephemeral storage. The audit log SQLite file at `data/agent_audit.db` resets on every restart. For a demo this is fine.
- **No HTTPS setup needed**: HF handles certs automatically.
- **No billing alerts needed**: it's genuinely free.

## When you want to take it down

Go to your Space's **Settings** tab in the HF UI and click **Delete Space**.
That's it.
