# sealbox MCP server. Run with --network none and mount the vault + keyfile read-only, e.g.
#   docker run --rm -i --network none -v "$PWD/vault:/vault:ro" \
#     -v "$HOME/.config/sealbox/master.key:/run/secrets/sealbox.key:ro" sealbox
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SEALBOX_VAULT=/vault \
    SEALBOX_KEYFILE=/run/secrets/sealbox.key

# util-linux provides `unshare` for the run_python network/mount namespaces. Inside an unprivileged container
# user namespaces may be unavailable; sealbox then falls back to rlimits + audit hook and says so in every
# run_python response ("isolation"/"warnings"). Set require_network_isolation: true in sealbox.yaml to refuse.
RUN apt-get update \
    && apt-get install -y --no-install-recommends util-linux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[parquet]" \
    && useradd --create-home --uid 10001 sealbox

USER sealbox
ENTRYPOINT ["sealbox"]
CMD ["serve"]
