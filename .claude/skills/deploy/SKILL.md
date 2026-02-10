---
name: deploy
description: Deploy Morpheus to the production droplet
disable-model-invocation: true
allowed-tools: Bash, Read
---

Deploy Morpheus to the DigitalOcean production droplet.

## Steps

1. Check `git status` for uncommitted changes. If there are changes, ask the user if they want to commit first.
2. Run `git push origin main` to push to GitHub.
3. Deploy with: `ssh morpheus "cd /opt/morpheus && git pull && systemctl restart morpheus"`
4. Wait 5 seconds, then check startup logs: `ssh morpheus "journalctl -u morpheus -n 20 --no-pager"`
5. Verify the bot started successfully (look for `morpheus_v2_started` in logs).
6. Report: engines loaded, account balances, any errors.

## Notes
- The bot runs at `/opt/morpheus` on the droplet as a systemd service called `morpheus`.
- Config is in `config.yaml`, secrets in `.env` on the server.
- If deploy fails with merge conflicts, do NOT force-push. Ask the user.
