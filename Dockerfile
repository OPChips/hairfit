FROM python:3.12-slim

# 报告卡 PNG 用 Pillow 渲染中文，镜像内必须有 CJK 字体（路径见 hairfit/core/report.py）
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

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
