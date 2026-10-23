# HairFit MVP 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 按 `docs/design.md` 实现 HairFit MVP:多模态 LLM 推荐发型 + 图像编辑模型换发,BYOK 多 provider 可配置,CLI + FastAPI 双入口,含完整测试与自检。

**Architecture:** Ports & Adapters 分层(core 领域层只依赖两个 Protocol 端口;providers 基础设施层实现各厂商适配器)。job 持久化 SQLite;provider 故障转移链;QC 降级不在关键路径。含 mock provider 用于测试与无 key 演示。

**Tech Stack:** Python 3.12+, httpx, pydantic, pydantic-settings, pillow, tenacity, structlog, aiosqlite, fastapi; dev: pytest, pytest-asyncio, ruff, mypy

**自检策略:** 无真实 API key,自检走 mock provider 全链路(单元测试 → 契约测试 → 服务端 E2E)→ ruff/mypy → 修复关键问题 → 循环直到全绿。真实模型验证需用户配置 key 后跑 `regression/benchmark.py`。

---

## 目录结构(目标)

```
hairfit/
├── config/models.json           # 默认模型配置
├── kb/face_shapes.json          # 脸型规则库(schema_version + references)
├── kb/hairstyles.json           # 20 款发型元数据
├── assets/styles/               # 参考图(用户后续自举生成填充)
├── regression/                  # benchmark.py + 测试脸集目录
├── hairfit/
│   ├── __init__.py
│   ├── config.py                # pydantic-settings,加载 models.json + .env
│   ├── log.py                   # structlog 配置
│   ├── providers/
│   │   ├── base.py              # AnalysisProvider / GenerationProvider Protocol
│   │   ├── registry.py          # 按 config 实例化 provider + failover 链
│   │   ├── llm/
│   │   │   ├── base.py          # LLM 适配器公共逻辑(httpx + 重试)
│   │   │   ├── openai_compat.py # OpenAI 兼容(chat completions, vision, json_schema)
│   │   │   ├── anthropic.py     # Claude messages API
│   │   │   └── mock.py          # 测试/演示用确定性 mock
│   │   └── image/
│   │       ├── base.py          # 生图适配器公共逻辑(提交/轮询/重试)
│   │       ├── gpt_image.py     # OpenAI images/edits
│   │       ├── nano_banana.py   # Gemini generateContent
│   │       ├── seedream.py      # Ark 异步任务(提交→轮询)
│   │       ├── grok.py          # xAI images
│   │       └── mock.py          # 生成纯色占位图(Pillow)
│   ├── schemas.py               # FaceAnalysis / Recommendation / QCScore 等全部 pydantic 模型
│   ├── kb.py                    # 知识库加载 + 校验
│   ├── prompts.py               # 推荐/QC prompt 模板(按 design.md 第4/5节)
│   ├── core/
│   │   ├── recommend.py         # 组装输入 → LLM → 校验(id 白名单/3-5项/重试)
│   │   ├── qc.py                # 生成图 VLM 评分
│   │   ├── generate.py          # 编排:生成→QC→重试→failover;job 状态机
│   │   ├── jobs.py              # aiosqlite job store
│   │   └── report.py            # Pillow 拼版报告卡
│   ├── server.py                # FastAPI: /healthz /analyze /generate /jobs/{id} /report
│   └── cli.py                   # argparse: recommend / generate / serve / validate-kb
├── tests/
│   ├── conftest.py
│   ├── test_schemas.py  test_kb.py  test_prompts.py
│   ├── test_recommend.py  test_qc.py  test_generate.py  test_jobs.py
│   ├── test_providers_llm.py  test_providers_image.py   # 契约测试(录制的 fixture JSON)
│   └── test_server.py                                     # E2E:mock 全链路
├── pyproject.toml  .env.example  .gitignore  README.md
```

## 任务清单

### Task 1: 项目骨架
- `pyproject.toml`(deps 见上,hatchling 构建,`[project.scripts] hairfit = "hairfit.cli:main"`)
- `.gitignore`(.env, __pycache__, assets 除外)、`.env.example`(`HAIRFIT_OPENAI_API_KEY=` 等)、`git init`
- `hairfit/log.py`:structlog JSON 配置(`configure_logging(verbose: bool)`)
- 验证:`uv sync`(或 `pip install -e .`)成功,`python -c "import hairfit"` 通过

### Task 2: schemas.py(一切 JSON 有 schema)
TDD:`tests/test_schemas.py` 先行。
模型:
- `HairTexture = str` 约束 `^[1-4][abc]$`
- `FaceAnalysis`: face_shape(literal 7 型)/face_ratio/hair_volume/hair_texture/hairline/notes
- `Recommendation`: id/reason;`RecommendationResult`: face_analysis + recommended[3:5]
- `QCScore`: face_consistency(0-10)/style_match(0-10)/verdict(pass|retry)/reason
- `JobStatus` 枚举: queued/generating/qc_passed/qc_retried/done/failed
- `Job`: id/status/style_ids/results/created_at/updated_at/error
- KB 模型:`FaceShapeRule`/`FaceShapesKB`(meta_rule/shapes/references/schema_version)/`Hairstyle`(design.md 3.2 全字段,hair_texture_fit 逐项校验)/`HairstylesKB`

### Task 3: 知识库数据
- `kb/face_shapes.json`:design.md 3.1 的 7 型全量(geo+strategy+suit/avoid),`schema_version: "1.0"`,`references` 字段填 3.5 节来源
- `kb/hairstyles.json`:20 款(10男10女),每款 design.md 3.2 全字段 + description 按五段模板(长度+层次+刘海+顶部+两侧),风格族对齐 Hair20k 10 类,覆盖矩阵配平(short/medium/long 每性别≥2)
- `hairfit/kb.py` + `tests/test_kb.py`:加载、schema 校验、`load_face_shapes()/load_hairstyles()`、id 唯一性断言
- `hairfit/cli.py validate-kb` 子命令;`hairfit/cli.py` 骨架(argparse,子命令分发)
- 验证:`hairfit validate-kb` 退出码 0;测试通过

### Task 4: 配置与 provider 端口
- `config/models.json`:design.md 2.1(默认 gpt-6.1-sol / gpt-image-2.5,各带 fallback 链:analysis→["openai","anthropic","mock"],generation→["openai","nano_banana","mock"],mock 永远兜底)
- `hairfit/config.py`:`ProviderConfig(provider,model,base_url,api_key_env,params,fallback)`;`AppConfig(analysis,generation)` 加载 JSON,key 从 env 读取(`api_key_env` 引用环境变量名,缺 key 时该 provider 不可用并记 warning——支持无 key 跑 mock)
- `hairfit/providers/base.py`:`AnalysisProvider`/`GenerationProvider` Protocol(design.md 9.3 签名)+ `ProviderUnavailable` 异常
- `tests`:配置加载/缺 key 降级/mock 兜底

### Task 5: LLM 适配器(openai_compat + anthropic + mock)
TDD 契约测试:用录制 fixture JSON(存放 `tests/fixtures/`)断言请求构造与响应解析。
- `llm/base.py`:公共 `httpx.AsyncClient` 管理、tenacity 重试(429/5xx/超时→指数退避,4xx 不重试)、结构化日志(provider/model/耗时/估算 token)
- `llm/openai_compat.py`:`POST {base_url}/chat/completions`,image 走 `image_url` data URL,JSON 输出走 `response_format`+prompt 兜底解析(fenced json 提取)
- `llm/anthropic.py`:`POST {base_url}/v1/messages`,image 走 base64 source block
- `llm/mock.py`:无网络,按 prompt 关键词返回确定性 JSON(含一张脸的假分析);构造 `analyze()` 返回合规 RecommendationResult
- 各适配器 4 个契约用例:请求头/体正确、正常响应解析、429 重试后成功、JSON 解析失败抛明确异常

### Task 6: 生图适配器(gpt_image/nano_banana/seedream/grok + mock)
- `image/base.py`:同 base 重试策略 + 轮询辅助(`poll_until(fn, interval, deadline)`)
- `gpt_image.py`:`POST {base_url}/images/edits`(multipart: image[] + prompt + model),同步返回 b64 → bytes
- `nano_banana.py`:`POST {base_url}/models/{model}:generateContent`(inline_data 两张图),取 inlineData
- `seedream.py`:`POST` 提交任务 → 轮询 `GET {task_id}` 直到 succeeded,取 url 下载
- `grok.py`:`POST {base_url}/images/generations`
- `image/mock.py`:Pillow 画 1024x1536 渐变占位图(嵌入款名文字),确定性强,供 E2E/演示
- 契约测试:请求构造/响应解析/异步任务轮询(fakeredis 不需要,monkeypatch 时间)/失败重试

### Task 7: provider registry(实例化 + failover)
- `providers/registry.py`:`build_analysis(cfg) -> list[AnalysisProvider]`(含 fallback 链,缺 key 跳过)、同 generation;`FailoverAnalysis/FailoverGeneration` 包装:逐个尝试,全失败抛 `ProviderUnavailable`
- 测试:主 provider 抛错→fallback 接管;全部失败→聚合异常

### Task 8: 推荐链路 core/recommend.py + prompts.py
- `prompts.py`:`build_recommendation_prompt(kb_face, kb_styles)`(design.md 第4节模板逐字:元规则+规则库+全量条目+4条要求+JSON 格式)、`build_qc_prompt(...)`、`build_generation_prompt(style, keep_color=True)`(design.md 第5节英文模板+发色保留约束)
- `recommend.py`:`RecommendService.analyze(photo: bytes) -> RecommendationResult`:组装→LLM→pydantic 校验→id 白名单校验(非法 id/数量不符→带错误信息重试 1 次→仍失败抛 `RecommendationError`)
- 测试:正常/首答含非法 id 重试后成功/两次都失败

### Task 9: QC + 生成编排 core/qc.py + generate.py + jobs.py
- `qc.py`:`score(image, user_photo, style) -> QCScore`,verdict=retry 阈值 face_consistency<6
- `jobs.py`:aiosqlite(WAL)`create/get/update_job`,`jobs` 表(status/json 字段/时间戳);`init_db()`
- `generate.py`:`GenerateService.run(job_id, photo, style_ids, on_progress)`:
  - 每款:generation provider 生成 1 张 → QC → retry 判定则重生成 1 次(带"上次问题"改进约束)→ 结果记录(qc_passed/qc_retried/未质检降级标记)
  - 单款失败不影响其他款;全部完成置 done
  - failover 由 registry 包装层负责
- 测试:QC 通过直返/QC 失败重试一次/重试仍失败标记降级/单款失败隔离

### Task 10: 报告卡 core/report.py
- `report.py`:`build_report(analysis, results, styles, user_photo) -> bytes`:Pillow 拼版——标题+脸部分析摘要行;每款一列:试戴图(缩放)+款名+✅;不推荐款(可选入参)用参考图+❌+原因;文字代码渲染(默认字体 ImageFont.load_default,有系统中文字体则加载)
- 测试:输出 PNG 尺寸正确、包含 N 列(像素级轻断言即可)

### Task 11: FastAPI server.py
- 端点:`GET /healthz`、`GET /healthz/providers`(各 provider key 是否配置)、`POST /v1/analyze`(multipart photo→RecommendationResult)、`POST /v1/generate`(photo+style_ids→job_id,后台 asyncio.create_task 跑 GenerateService)、`GET /v1/jobs/{id}`、`POST /v1/report`(job_id[+不推荐 ids]→PNG)
- lifespan:init_db + configure_logging;异常处理(ProviderUnavailable→502,RecommendationError→422)
- E2E 测试(mock provider 链):healthz→analyze→generate→轮询 job 至 done→report 返回 PNG

### Task 12: CLI 完整化
- `recommend <photo>`:打印 JSON 结果
- `generate <photo> --styles id1,id2 --out dir/`:串行生成落盘 + QC 日志
- `serve [--port]`
- `validate-kb`(Task 3 已有)
- 测试:argparse 分发 smoke;`recommend` 用 mock 全链路跑通(CLI 级 E2E)

### Task 13: regression/benchmark.py + README
- `regression/benchmark.py`:遍历 `regression/faces/*`,对配置的全部 generation provider 逐款生成+QC,输出 markdown 得分表(面部一致性/编辑执行力均值)——设计 8.4 基准榜工具;需真实 key,提供 `--dry-run`(mock)模式自检
- `README.md`:项目简介、架构图(ASCII)、BYOK 配置说明、快速开始(uv sync → 配 .env → hairfit serve)、发型库入库规范摘要、免责声明
- `scripts/bootstrap_refs.py`:自举生成参考图(读 hairstyles.json 中 ref_image 缺失的款,用 generation provider 按入库规范 prompt 影棚标准化)——需 key,mock 可自检

### Task 14: 全量自检(verification-before-completion)
1. `uv run pytest` 全绿
2. `uv run ruff check . && uv run ruff format --check .` 干净
3. `uv run mypy hairfit` 通过
4. `uv run hairfit validate-kb` 通过
5. 起服务(默认配置含 mock 兜底,无需 key)→ curl 全链路:healthz/analyze/generate/job 轮询/report
6. CLI E2E:mock 链路 recommend+generate
7. 发现关键问题→修复→重跑以上,直到全部通过
8. git 提交(分任务小步提交)

## 关键接口(执行时保持一致)

```python
class AnalysisProvider(Protocol):
    async def analyze(
        self, prompt: str, image: bytes, json_schema: type[BaseModel]
    ) -> BaseModel: ...


class GenerationProvider(Protocol):
    async def edit(self, prompt: str, images: list[bytes], size: str = "1024x1536") -> bytes: ...
```

- 所有适配器:`name: str` 属性;不可用抛 `ProviderUnavailable`
- 失败模式纪律:编辑模型只有"保身份没执行编辑"或"执行编辑改了人"两种失败——QC prompt 按此设计评分维度
