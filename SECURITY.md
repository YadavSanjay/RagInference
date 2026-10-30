# Security policy

RagInference is a personal project maintained on a best-effort basis.

## Reporting a vulnerability

Please report a vulnerability privately through GitHub's **Report a vulnerability** button on
the repository's **Security** tab (private vulnerability reporting), not in a public issue.
Include the affected file or endpoint, steps to reproduce, and the impact you expect.

There is no fixed response time. Only the latest commit on `main` gets fixes.

## Scope notes

- **The service has no authentication and no TLS.** Network isolation is its only access control.
  Keep it on loopback (the local default) or a private network, and never expose port 8799 to the
  internet. Containers and `DEPLOY_MODE=aws` bind `0.0.0.0`, so a security group or equivalent
  must restrict who can reach them.
- Anyone who can reach the port can run CPU-heavy inference. `RAGINFER_MAX_ITEMS` and
  `RAGINFER_MAX_TEXT_CHARS` bound a single request, but there is no rate limiting.
- `jinaai/jina-reranker-v3` loads with `trust_remote_code=True`, which runs Python from the model
  repository. The commit is pinned in the code, so changing the pin means reviewing that code.
- Never put real credentials in an issue, pull request or commit. Secrets belong in `.env`
  (gitignored) or a `<NAME>_FILE` mount.
