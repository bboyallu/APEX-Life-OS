FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY apex ./apex
COPY KNOWLEDGE_BASE.md ./

RUN pip install --no-cache-dir .

# Knowledge base lives on a mounted volume at /data (raw/, wiki/, outputs/)
VOLUME ["/data"]
WORKDIR /data

ENTRYPOINT ["apex", "--knowledge-root", "/data"]
CMD ["daemon"]
