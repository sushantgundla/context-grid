# Self-hosting context-grid.
#
# The honest answer to "can I run this on documents I cannot upload anywhere?". Everything in
# the default install runs locally with no network access, so this image needs no keys and
# talks to nothing.

FROM python:3.12-slim

# Poppler and its friends are what the PDF parsers link against. Without them the parse extra
# installs and then fails at runtime, which is a worse experience than a larger image.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir ".[parse]"

# Documents and eval sets are mounted rather than baked in: the whole point of self-hosting
# is that the corpus never leaves the machine.
VOLUME ["/data"]

ENTRYPOINT ["contextgrid"]
CMD ["--help"]
