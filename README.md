# HairFit — AI 发型推荐与虚拟换发

多模态大模型分析脸型并推荐发型，图像编辑模型将用户照片换成推荐发型。BYOK（用户自带 API key），provider 全部可配置可替换。

> 设计文档：[docs/design.md](docs/design.md) · 实现计划：[docs/plans/2026-10-23-mvp-implementation.md](docs/plans/2026-10-23-mvp-implementation.md)

## 架构

```
用户照片 → 上传预检 → ① 分析+推荐（多模态 LLM，1 次调用）
                    输出 JSON {脸部分析, top3-5 推荐(库内 id+理由)}
        → ② 用户确认
        → ③ 逐款生成（图像编辑模型: 用户照片 + 该款参考图）
            + 自动质检（VLM 评分, 不及格带纠正指令重试一次）
        → ④ 试戴结果集 / 对比报告卡（代码拼版, 不用模型成图）
```

Ports & Adapters 分层：`core/`（业务编排）只依赖两个端口（`AnalysisProvider` / `GenerationProvider`），`providers/` 实现各厂商适配器，依赖方向向内。

## 支持的模型（config/models.json 切换）

| 环节 | 支持 |
|---|---|
| 分析+推荐 | OpenAI、Anthropic、DeepSeek、火山豆包、阿里千问、智谱、Kimi 等（OpenAI 兼容端点共用一个适配器） |
| 换发生成 | gpt-image、nano-banana（Gemini）、seedream（豆包 Ark）、grok、mock 等 |

不使用任何厂商 SDK，统一 httpx 异步直连 REST。每个环节声明 fallback 链，单厂商故障自动切换。未配置 key 的 provider 自动跳过，mock 永远兜底——**无 key 也能跑通全链路**。

## 快速开始

```bash
python3.12+ -m venv .venv && source .venv/bin/activate
pip install -e ".[server,dev]"

# 1. 校验知识库
hairfit validate-kb

# 2. 无 key 演示（mock 链路）
hairfit recommend tests/fixtures/sample_face.png --config <(echo '{"analysis":{"provider":"mock","model":"m","params":{}},"generation":{"provider":"mock","model":"m","params":{}}}')

# 3. 配置 key 后用真实模型
cp .env.example .env   # 填入你的 key
hairfit recommend face.jpg
hairfit generate face.jpg --styles textured_crop,crew_cut --out output/
hairfit serve          # http://127.0.0.1:8321/docs
```

## API

| 端点 | 说明 |
|---|---|
| `POST /v1/analyze` | 照片 → 脸部分析 + top3-5 推荐 |
| `POST /v1/generate` | 照片 + style_ids → job_id（后台生成） |
| `GET /v1/jobs/{id}` | job 进度与结果 |
| `POST /v1/report` | job → 对比报告卡 PNG（不推荐款用参考图+❌，零生成成本） |
| `GET /healthz` / `GET /healthz/providers` | 健康检查 / 各 provider key 状态 |

## 发型库

`kb/hairstyles.json`（20 款）+ `kb/face_shapes.json`（7 型规则），分类体系均有专业依据（见各 JSON `references` 字段与设计文档 3.5 节）。新增发型 = 一条元数据 + 一张参考图，零代码。

参考图规范与自举生成：

```bash
python scripts/bootstrap_refs.py --dry-run   # mock 自检
python scripts/bootstrap_refs.py             # 真实模型生成标准影棚参考图
```

`assets/styles/` 当前为占位图，用自举脚本或人工摄影图替换。

## 基准测试

```bash
# 放入测试脸 regression/faces/（10-20 张，覆盖不同脸型/性别/发质）
python regression/benchmark.py --dry-run     # mock 自检
python regression/benchmark.py               # 各 provider x 测试脸评分榜
```

## 开发

```bash
pytest            # 单元 + 契约 + E2E（全 mock，无需 key）
ruff check . && mypy hairfit
```

## 免责声明

仅供学习研究。请仅上传本人照片；API key 由使用者自行配置，遵守各模型厂商使用政策；原图即用即删，本项目不存储人脸数据。
