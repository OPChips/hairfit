# HairFit — 默认 Docker 部署，make 一条命令管理

COMPOSE := docker compose

.PHONY: up down restart logs health demo clean test lint

## 首次/日常部署：构建 + 后台启动（无 .env 时自动从模板创建，可无 key 跑 mock 链）
up:
	@test -f .env || cp .env.example .env
	$(COMPOSE) up -d --build
	@$(MAKE) --no-print-directory health

## 停止
down:
	$(COMPOSE) down

## 重建并重启（改了代码后用）
restart:
	$(COMPOSE) up -d --build --force-recreate

## 跟踪日志
logs:
	$(COMPOSE) logs -f

## 健康检查（含各 provider key 状态）
health:
	@curl -s http://127.0.0.1:8321/healthz && echo
	@curl -s http://127.0.0.1:8321/healthz/providers | python3 -m json.tool

## 无 key 演示：容器内走 mock 链路生成一张图到挂载卷
demo:
	@docker compose exec hairfit sh -c "\
	  python -c \"from PIL import Image; Image.new('RGB',(400,500),(170,140,120)).save('/tmp/me.png')\" && \
	  hairfit generate /tmp/me.png --styles textured_crop --out /data/output/demo"

## 清理容器（保留数据卷）
clean:
	$(COMPOSE) down --remove-orphans

# ---- 本地开发（不走 Docker）----
test:
	.venv/bin/python -m pytest tests/ -q

lint:
	.venv/bin/ruff check . && .venv/bin/mypy hairfit
