FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY hairfit ./hairfit
COPY kb ./kb
COPY config ./config
COPY assets ./assets

RUN pip install --no-cache-dir -e ".[server]"

# 运行时工作目录指向挂载卷：hairfit.db 与 output/ 都落在卷内，随卷持久化
RUN mkdir /data
WORKDIR /data
VOLUME /data

EXPOSE 8321

CMD ["hairfit", "serve", "--host", "0.0.0.0", "--port", "8321"]
