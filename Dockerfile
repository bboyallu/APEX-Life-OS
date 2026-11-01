FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY apex ./apex
RUN pip install --no-cache-dir . && useradd -m apex && mkdir /data && chown apex /data
USER apex
ENV APEX_HOME=/data
WORKDIR /data
ENTRYPOINT ["apex"]
CMD ["chat"]
