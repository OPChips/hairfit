# HairFit — AI 发型推荐与换发 设计方案

> 2026-10-22 · 纯大模型方案（多模态 LLM 推荐 + 图像编辑模型生成），不含传统 CV/GAN。
> BYOK（用户自带 API key）模式，API key 由使用者配置，仓库不内置任何 key。

## 1. 总体架构

```
用户照片
   ↓
① 分析+推荐（1 次多模态 LLM 调用）
   输入：用户照片 + 脸型规则库 + 发型库元数据
   输出：JSON {脸部分析, top3-5 推荐(库内ID+理由)}
   ↓
② 用户确认（产品层，必做——避免直接生成浪费成本）
   ↓
③ 换发生成（图像编辑模型）
   输入：用户照片 + 选中款参考图
   输出：换发型后的照片
```

要点：
- 分析与推荐合并为**一次调用**（现代上下文放得下全量库），降低延迟和成本
- 推荐只允许输出库内 `id`（约束解码/prompt 约束 + 代码层校验）
- 生成阶段用**编辑模型而非文生图**，保留用户本人面部

## 2. 模型配置

两个环节**解耦，各自可配置**。默认：分析推荐用 `gpt-6.1-sol`，换发生成用 `gpt-image-2.5`。

### 2.1 配置文件 `config/models.json`

```json
{
  "analysis": {
    "provider": "openai",
    "model": "gpt-6.1-sol",
    "params": { "temperature": 0.25, "max_output_tokens": 2048 }
  },
  "generation": {
    "provider": "openai",
    "model": "gpt-image-2.5",
    "params": { "size": "1024x1536", "quality": "high" }
  }
}
```

- 代码只依赖 `analysis` / `generation` 两个抽象接口（`analyze(photo, kb) -> JSON`、`edit(user_photo, ref_image, prompt) -> image`），provider 层适配各家 API
- **实现约束：不使用各厂商专属 SDK，统一用异步 HTTP 请求直连 REST API**（如 httpx/aiohttp）——依赖干净、所有 provider 走同一抽象，新增 provider 只需实现请求/响应适配器
- 换模型 = 改配置文件，不动代码；也支持灰度（如 5% 流量走备选模型对比效果）
- 默认模型说明：
  - **`gpt-6.1-sol`**（分析推荐）：GPT-6 系列中端款（旗舰为 GPT-6 Astra），近旗舰性能、成本低（$2/M input），多模态 + 结构化 JSON 输出稳定。**已确认**适合本环节
  - **`gpt-image-2.5`**（换发生成）：**⚠️ 待基准测试确认的临时默认**。其局部编辑不毁画面其余部分的优势明确，但换发场景的第一质量指标是面部身份保持，而公开对比中 Nano Banana 在身份保持上被引用最多。编辑模型只有两种失败模式——"保住身份但没执行编辑" vs "完成编辑但改了人"，必须用测试脸集实测取舍，不能用默认值赌。MVP 第 1 周跑 gpt-image-2.5 vs Nano Banana 双指标基准（面部一致性 × 编辑执行力），胜者设为默认
  - gpt-image-2.5 API 变体：`flare`（快、便宜，适合批量）与 `sunburst`（精准编辑），基准测试时一并对比

### 2.2 支持的模型（均可通过配置切换）

| 环节 | 支持范围 |
|---|---|
| 分析+推荐（多模态/文本大模型） | OpenAI、Anthropic（Claude）、DeepSeek、火山豆包、阿里千问（Qwen-VL）、智谱（GLM）、Kimi（Moonshot）等主流大模型 |
| 换发生成（生图/编辑模型） | gpt-image（OpenAI）、nano-banana（Gemini）、seedream（豆包）、grok（xAI）等主流生图模型 |

各 provider 差异（端点、鉴权、请求/响应格式、异步任务轮询）由 provider 适配器抹平；生图 API 中异步任务型（提交后轮询取结果）与同步返回型统一封装为 `await edit(...) -> image`。

## 3. 知识库设计

两个文件，均结构化 JSON，随代码走版本管理。每个文件带 `schema_version` 字段，加载时按 pydantic schema 校验；规则变更走 PR + 测试脸集回归验证。运营调整改文件即可，不动 prompt。分类体系与规则均有行业/学术依据（见 3.5）。

### 3.1 脸型规则库 `kb/face_shapes.json`

业界标准 6-8 类。每型：几何判断特征 + 一条核心策略 + suit/avoid 细则。
元规则（注入 prompt 顶部）：**平衡与比例——脸哪里窄就在哪里加体积，哪里有棱角就软化哪里。**

```json
{
  "meta_rule": "Balance and proportion: add volume where the face is narrow; soften where the face has strong angles.",
  "shapes": {
    "oval": {
      "geometry": "长宽比约 1.5:1，额头略宽于下颌，下巴圆润，轮廓无突出棱角",
      "strategy": "百搭脸型，按发质/发量/场合推荐，无需强规则",
      "suit": ["绝大多数款式"],
      "avoid": ["过重过厚的全包刘海（遮盖均衡比例）"]
    },
    "round": {
      "geometry": "长宽比接近 1:1，脸颊饱满，下巴圆润，无棱角",
      "strategy": "纵向拉伸：顶部加高加体积，两侧收窄",
      "suit": ["顶部有高度/体积的款式", "侧边短、顶部长的渐变", "侧分（打破对称）"],
      "avoid": ["齐下巴长度的鲍伯", "厚重齐刘海", "中分贴头皮长直发", "两侧蓬松款"]
    },
    "square": {
      "geometry": "额头、颧骨、下颌宽度接近，下颌线条方正有明显棱角",
      "strategy": "软化棱角：用弧度和层次柔和下颌线",
      "suit": ["带弧度的层次剪裁", "侧分短碎发", "柔和卷发"],
      "avoid": ["水平齐刘海（强化方正感）", "齐平利落的直线条短发（如平头）"]
    },
    "heart": {
      "geometry": "额头/太阳穴宽，颧骨高，下巴尖",
      "strategy": "平衡上宽下窄：下巴附近加体积，顶部压Volume",
      "suit": ["下巴长度的波浪/Lob", "侧分", "下巴附近有层次的款"],
      "avoid": ["厚重顶部+贴头皮两侧（加重上宽）", "超短贴头款（暴露尖下巴）"]
    },
    "diamond": {
      "geometry": "颧骨最宽，额头和下巴均窄，脸部线条有折角",
      "strategy": "加宽额头和下巴视觉宽度，收颧骨存在感",
      "suit": ["带刘海的款（补额头宽度）", "下巴长度内扣卷", "中长发微卷"],
      "avoid": ["全部往后梳（暴露窄额头）", "贴头皮高马尾"]
    },
    "oblong": {
      "geometry": "长宽比明显大于 1.5:1，脸型偏长，宽度均匀",
      "strategy": "横向拓宽：两侧加体积，压低顶部高度",
      "suit": ["两侧有体积的款", "斜刘海/法式刘海（截短脸长）", "中长卷发"],
      "avoid": ["顶部高耸款（油头/高马尾，拉长脸型）", "两侧铲青的渐变", "超长贴头皮直发"]
    },
    "triangle": {
      "geometry": "下颌宽于额头/太阳穴，上窄下宽",
      "strategy": "加宽上半部：顶部和太阳穴位置加体积",
      "suit": ["顶部蓬松/有高度的款", "侧分上梳款", "短款向上造型"],
      "avoid": ["贴头皮顺贴款", "下巴附近向内收的剪裁（加重下宽）"]
    }
  }
}
```

> 注：分布统计（oval ~22%、round ~17%、heart ~14%、square ~10%、triangle ~9%、diamond ~5%）说明 oval/round 是大头，规则库对这两类的质量决定大多数用户体验。

### 3.2 发型库元数据 `kb/hairstyles.json`

每款一条。核心：**多约束维度**，发量/发质是硬过滤项，脸型是打分项。

```json
{
  "items": [
    {
      "id": "textured_crop",
      "name": "纹理烫短碎盖",
      "gender": "male",
      "length": "short",
      "description": "顶部带纹理感的短碎发，前额微微前倾，两侧渐变收短",
      "volume_demand": "medium",
      "hair_texture_fit": ["1a", "1b", "1c", "2a", "2b"],
      "suit_face": ["round", "square", "oval"],
      "avoid_face": ["oblong"],
      "scene_tags": ["daily", "business"],
      "ref_image": "styles/male/textured_crop.jpg"
    },
    {
      "id": "french_bob_wavy",
      "name": "法式微卷波波头",
      "gender": "female",
      "length": "medium",
      "description": "下巴至下颌长度的波波头，带松散法式卷度，配轻薄的法式刘海",
      "volume_demand": "low_to_medium",
      "hair_texture_fit": ["1a", "1b", "1c", "2a", "2b", "2c", "3a"],
      "suit_face": ["heart", "diamond", "oblong", "oval"],
      "avoid_face": [],
      "scene_tags": ["daily", "fashion"],
      "ref_image": "styles/female/french_bob_wavy.jpg"
    }
  ]
}
```

Schema 字段说明：

| 字段 | 作用 | 推荐逻辑中的角色 |
|---|---|---|
| `volume_demand` / `hair_texture_fit` | 该款对用户自身发量发质的要求 | **硬过滤**（不可匹配直接剔除） |
| `suit_face` / `avoid_face` | 脸型适配 | **打分**（suit +2, avoid -5, 其他 0） |
| `length` / `scene_tags` / `gender` | 产品侧过滤条件 | 用户可选偏好，命中 +1 |
| `ref_image` | 换发生成阶段的参考图 | 生成阶段输入 |
| `description` | 注入推荐 prompt，供 LLM 理解款式 | 推荐上下文 |

### 3.3 发型库入库规范（质量护城河，权重高于模型选型）

编辑模型把参考图当视觉真值，**参考图的每个缺陷都会被搬进生成结果**；推荐是闭集选择，**库里没有的风格用户永远得不到**。库不标准或不全，比模型选错更伤质量。

**参考图标准（全部满足才可入库）：**

- 正面或统一标准 3/4 角度，全库角度一致
- 纯色/中性背景，均匀影棚光
- 自然深色发（贴近主要用户群），不用特殊染色——降低发色渗入用户照片的风险
- 肩部以上，发型完整可见、无遮挡（手/帽子/饰品压发）
- 分辨率 ≥ 1024px
- 每款 1 张主图；后续可扩展左右 45° 多视图（Nano Banana 等支持多参考图输入）

**元数据标准：**

- `description` 用固定模板填写：`长度 + 层次 + 刘海形态 + 顶部形态 + 两侧形态`（如"下巴长度 + 无层次 + 轻薄法式刘海 + 顶部贴顺 + 两侧内扣"）——结构一致，LLM 理解才稳定
- 所有条目由同一人按同一标准标注，避免标注漂移

**覆盖矩阵（防"不全"）：**

MVP 20 款按下表配平，低频脸型（diamond/triangle）必须有专属适配款：

| 维度 | 取值 | 配平要求 |
|---|---|---|
| gender | male / female | 各 10 |
| length | short / medium / long | 每性别至少各 2 |
| 风格族 | 抄 Hair20k 的 10 类风格体系（直发/卷发/扎发/刘海类/渐变类等） | 每族 ≥ 2，不许某族空缺 |
| 发质分型 | Andre Walker Type 1-4（见 3.4） | 首发中国市场以 Type 1 为主；出海前补 Type 2-4 分区 |

**类目体系不从零设计**：产品级标准发型库不存在（商业目录如 Perfect Corp 200+ 款均闭源不授权），自建是唯一路径，但直接借鉴学术体系——风格族分类抄 Hair20k（10 类），属性标注 schema 抄 K-Hairstyle（50 万张、含分割与风格标注），图片按本节规范自产。

**参考图生产方式（自举）：**

收集的原始图片（网络图/实拍）先用编辑模型统一"影棚标准化"——统一背景、光线、角度、发色后入库。**用同一生成模型制作库，天然和线上生成风格对齐**。入库前用 VLM 自动质检（角度/背景/遮挡/发色四项），不合格打回。

**覆盖度监控（上线后）：**

推荐日志统计每款的推荐频次和被选率——长期零推荐的款说明元数据或适配规则有问题；某类用户（按 face_analysis 分桶）可选项总是 < 3，说明覆盖矩阵在该维度缺款，进入下批扩充清单。

### 3.4 发质分型与人种适配（Andre Walker Type 1-4）

发质分型采用发业界通行标准 **Andre Walker 分型**（每型再分 a/b/c 表示粗细/卷紧度）：

| 型 | 描述 | 人群分布 |
|---|---|---|
| Type 1 | 直发（1a 细软 → 1c 粗硬） | 东亚人群绝大多数 |
| Type 2 | 波浪（2a 松散 S 波 → 2c 明显波纹） | 欧洲裔常见 |
| Type 3 | 卷发（3a 松圈 → 3c 紧螺旋） | 欧洲裔/南亚 |
| Type 4 | Coily/afro 纹理（4a → 4c 锯齿状紧卷） | 非洲裔为主 |

设计影响：

1. **`hair_texture_fit` 字段取值改为 `["1a"..."4c"]`**（分析侧输出同为 Type 分型）——这是发质约束的标准语言，比 straight/wavy/curly 三分法保留更多约束信息
2. **生成端失败模式**：Type 4 用户选 Type 1 款式，编辑模型易渲染出"假发套"感（纹理变换是编辑模型弱项）——**测试脸集必须覆盖不同发质**，不仅不同脸型
3. **脸型 7 分型存在族群校准偏差**（源自特定人群的美发教材体系）——因此推荐以 VLM 输出的 `face_ratio` + 几何特征为主、脸型标签为辅，规则库的 geometry 描述同样按几何特征表述
4. **皮肤色调**主要影响"推荐发色"而非发型形状——归入换发色姊妹功能（肤色冷暖调 → 发色色板），不进发型推荐主链路
5. **市场策略**：首发中国（Type 1 主导）库负担小；出海前必须为 Type 2-4 建独立分区（其风格词汇——编发/locs/protective styles——与 Type 1 体系完全不同）

### 3.5 知识库专业依据

所有分类体系与规则均有权威来源，不做自造标准：

| 知识 | 依据来源 |
|---|---|
| 脸型 7 分类与几何判据 | 美发师职业教育标准体系（cosmetology 教材：Cinta Aveda Institute 等培训课程；6 基础型 + 扩展型） |
| 元规则（平衡与比例：窄处加体积、棱角处软化） | 美发师咨询培训的通行核心原则，各 salon 培训教材一致 |
| 脸型→发型 suit/avoid 细则 | 同上教材体系 + 学术发型推荐系统论文（Atlantis Press 2019、IRJET）采用的映射规则 |
| 发质分型 Type 1-4 | Andre Walker Hair Typing System——发业界通行标准（Oprah 御用造型师提出）；高卷曲度发型的临床系统综述（PMC13046883）佐证其分型有效性 |
| 发型风格族类目 | Hair20k 数据集的 10 类体系（2.1 万款 3D 发型整理） |
| 发型属性标注 schema | K-Hairstyle 数据集（50 万张，含风格属性标注）的字段体系 |
| 脸型分布统计（用于测试集配平） | 5 万+样本统计（oval ~22%、round ~17%、heart ~14%、square ~10%、triangle ~9%、diamond ~5%） |

维护规则：引用来源记录在各 JSON 文件的 `references` 字段中，规则改动时同步核对来源；无来源支撑的规则不进库。

## 4. 推荐调用 Prompt 模板

单次多模态调用，system/user 结构：

```
[System]
你是专业发型顾问。基于用户照片和知识库，分析面部特征并推荐发型库中最合适的发型。

分析元规则：{face_shapes.json 的 meta_rule}
脸型知识库：{face_shapes.json 的 shapes 全文}
发型库（只能从此列表推荐）：{hairstyles.json 全量条目}

要求：
1. 只输出 JSON，格式如下
2. recommended 数组只能使用发型库中存在的 id，输出 3-5 个
3. reason 必须引用具体规则（如"圆脸适合顶部加高——该款顶部体积符合纵向拉伸策略"）
4. 若用户发量/发质与某款硬性不符，不得推荐该款

[User]
{用户照片}

请分析并推荐，输出：
{
  "face_analysis": {
    "face_shape": "round|square|heart|diamond|oblong|triangle|oval",
    "face_ratio": "约 1.4:1",
    "hair_volume": "thin|medium|thick",
    "hair_texture": "1a~4c（Andre Walker 分型，见知识库 3.4）",
    "hairline": "normal|high|receding",
    "notes": "其他影响推荐的观察（眼镜、发际线形状等）"
  },
  "recommended": [
    { "id": "textured_crop", "reason": "引用规则的一句话理由" }
  ]
}
```

工程要点：
- `temperature` 0.2~0.3（比分类场景略高，理由文本需要自然）
- 代码层校验：输出 id 必须在库中、数量 3-5、违规则重试一次
- 推荐排序：LLM 输出顺序即为展示顺序（LLM 已综合打分）；如需更稳，可让 LLM 每款同时输出 1-10 分，代码按分排序

## 5. 换发生成 Prompt 模板

用户确认推荐后，取该款 `ref_image` 与用户照片一起送编辑模型：

```
输入：[图1] 用户照片, [图2] 发型参考图

Prompt:
Replace the hairstyle of the person in image 1 with the hairstyle
shown in image 2. Keep the person's face, facial features, identity,
expression, skin tone, pose, clothing, and background exactly
unchanged. Only change the hair: {description}。

If the user's hair color differs from the reference, keep the
user's original hair color.
```

要点：
- **保持用户原发色**是高频踩坑点（参考图发色容易被带过去），必须显式约束
- Qwen-Image-Edit 自部署时可用 mask 限定只编辑头发区域，面部保护更稳
- 每款生成 2 张供选择，降低单次翻车率
- **自动质检（必做）**：生成后用廉价 VLM（复用 gpt-6.1-sol）对结果自动评分——面部与原图一致性、发型与参考图匹配度——不及格自动重生成一次再返回用户。文本判断成本近零，显著降低差评率
- **分级生成（控制批量成本）**：批量阶段每款只生成 1 张 + 质检（不合格自动重试）；用户对某款满意想细看时，再为该款补生成第 2 张供对比。5 款批量成本从 10 次生成降为 5 次

**最终输出（试戴结果集）**，前端呈现为三层结构：

```
┌─────────────────────────────────────┐
│  汇总层：脸部分析摘要（脸型/发质/分析理由）│ ← face_analysis
├─────────────────────────────────────┤
│  款式卡片层 ×N（核心交付）：             │
│  [试戴图1（+细看后补第2张）]             │
│  款名 + 推荐理由                        │ ← recommended[].reason
│  👍合适 / 👎不合适（反馈闭环）            │
├─────────────────────────────────────┤
│  动作层：单款重生成 / 一键对比报告卡(5.1)  │
│  / 保存分享                            │
└─────────────────────────────────────┘
```
- 上线前用 10-20 张不同脸型/性别/发质的测试脸做两件事：
  1. **推荐回归**：换模型版本后跑推荐，看结果是否漂移
  2. **生成回归**：面部一致性抽查（眼距、鼻型、肤色）

### 5.1 输出形态：单款试戴图 + 对比报告卡

**形态一（默认）：单款试戴图**——上文流程的直接输出。

**形态二（可选，一键生成）：发型对比报告卡**——N 宫格汇总推荐结果，参考市面上"图文并茂发型报告"类产品，但实现走组合式而非一次成图：

```
逐款生成（复用现有 pipeline 输出）
        ↓
代码层拼版（PIL/HTML canvas）：
宫格 + 款名/理由文字 + ✅推荐 / ❌不推荐 标记 + 脸部分析摘要
```

实现原则：

- **组合式，严禁让编辑模型一次生成整张报告**——格子越多跨格面部一致性越差，文字渲染不可控，一格翻车整张重做。单款生成已过质检，报告卡只是零成本的拼版层
- 文字由代码渲染：零错别字、天然支持中文/i18n
- **不推荐格子用库参考图（模特脸）+ ❌ + 一句原因，不生成**——零成本传达"为什么不合适"；且避免"生成失败碰巧好看反而打脸推荐逻辑"的坑。视觉语义也清晰：用户脸 = 真实试戴，模特脸 = 仅展示。仅当用户主动要求对比时才为不推荐款生成 1 张
- 成本提示：报告卡 = N 次生成 ≈ $0.19/格，demo 中明示 BYOK 花费
- 产品价值：报告卡是天然分享物料——用户晒图即项目传播，比单张试戴图传播力强

## 6. 数据与运营

- 知识库（`kb/*.json`）与发型参考图入库，PR 审核变更——推荐行为可审计、可回滚
- 新增发型 = 加一条元数据 + 一张参考图，无需改代码
- 用户反馈闭环：推荐卡片上加"不喜欢/不合适"按钮，回收数据用于调整 `suit_face` 规则和排序
- 合规：人脸数据是敏感个人信息——用第三方 API（gpt-image-2.5 等）需明确告知并获授权，优先选择不留存训练的条款；自部署 Qwen 方案可规避数据出域

## 7. MVP 落地顺序

1. **第 1 步**：手工准备 20 款发型（10 男 10 女）的参考图 + 元数据，同时准备 10-20 张测试脸
2. **第 2 步**：推荐链路跑通（gpt-6.1-sol + 两个 JSON），CLI 或简单 Web 页验证推荐质量
3. **第 3 步**：**生成模型基准测试**——gpt-image-2.5 vs Nano Banana（含 flare/sunburst 变体），用测试脸集按"面部一致性 × 编辑执行力"双指标评分，胜者设为默认；随后接入自动质检环节
4. **第 4 步**：产品化（用户确认交互、历史记录、换发色姊妹功能、BYOK key 配置界面、demo 部署）
5. **第 5 步**：量大后评估自部署 Qwen-Image-Edit-2511 替换生成环节，控成本和数据出域

## 8. 关键风险与注意事项

### 8.1 责任分配（BYOK）

- **API 使用政策**：各模型厂商对真人照片编辑有各自政策（人脸验证、可识别真人编辑限制）。BYOK 模式下 API key 由使用者配置，相应责任随 key 归属使用者；README 免责声明"仅供学习研究、请上传本人照片"，仓库不内置任何 key
- **隐私自律**：原图即用即删、本地/自部署处理不落服务器，写进 README
- **参考图版权**：网络扒图有版权，随仓库分发即侵权——参考图必须**用编辑模型自举生成**（3.3 节影棚标准化流程），库天生干净可分发
- **License 选择**：发型库 JSON + 自产参考图的许可条款可与代码分开设（如代码 MIT、内容 CC BY-NC）

### 8.2 输入质量门禁

- 上传前质量预检（正脸角度、光线、遮挡、分辨率），不合格现场引导重拍，不进生成链路
- 质量分阈值作为硬门禁——歪斜/逆光照片任何模型都救不回来

### 8.3 成本与体验

- **成本提示**：一次尝试 ≈ 推荐调用（几分钱）+ 2 张生成（gpt-image 高质量约 $0.19/张）≈ **¥3**，demo 里明确告知用户 BYOK 预估花费
- **延迟**：编辑模型生成 10-60s，异步 + 进度反馈必做；可预生成推荐 top1 缩短等待
- **存储**：原图即用即删（24h 内），隐私自律 + 磁盘成本双重要求

### 8.4 项目健康指标

| 指标 | 衡量什么 |
|---|---|
| 生成成功率 / 重试率 | 质检环节直接产出，反映链路稳定性 |
| 用户反馈中"不像我"占比 | 换发自然度，最重要的质量信号 |
| 推荐采纳率（top3 被选中比例） | 知识库质量 |
| 基准测试榜（各模型 × 测试脸集得分） | 差异化卖点，随新模型持续更新 |

## 9. 技术实现

### 9.1 语言与依赖

**Python 3.12+**。理由：httpx 异步生态契合"直连 REST、不用厂商 SDK"的约束；Pillow 做纯本地拼版；FastAPI 一条链路到 Web demo。国内主流厂商（DeepSeek/千问/智谱/豆包/Kimi/Gemini）均提供 OpenAI 兼容端点——一个 `openai_compat` 适配器覆盖大多数，仅 Anthropic 需单独适配。

| 依赖 | 用途 | 必要性 |
|---|---|---|
| `httpx` | 异步 HTTP，全部模型调用 | 核心 |
| `pydantic` + `pydantic-settings` | 配置、LLM 输出校验、知识库 schema | 核心 |
| `pillow` | 报告卡拼版、图片压缩 | 核心 |
| `tenacity` | 重试（指数退避，标准重试库） | 核心 |
| `structlog` | 结构化 JSON 日志 + 成本/延迟打点 | 核心 |
| `aiosqlite` | job 持久化（SQLite WAL） | 核心 |
| `fastapi` + `uvicorn` | Web demo | 阶段引入 |
| `ruff` / `mypy` / `pre-commit` | lint + 严格类型检查 | 开发 |
| `pytest` + `pytest-asyncio` | 测试 + 回归集 | 开发 |

运行时依赖 6 个包，全部是单一职责的标准库级组件，无 SDK、无重框架。

### 9.2 目录结构

```
hairfit/
├── config/
│   └── models.json              # 模型配置（provider/model/base_url/key 引用/参数）
├── kb/
│   ├── face_shapes.json         # 脸型规则库
│   └── hairstyles.json          # 发型库元数据
├── assets/styles/               # 参考图（自举生成）
├── regression/                  # 测试脸集 + 基准测试脚本
├── hairfit/
│   ├── providers/
│   │   ├── base.py              # AnalysisProvider / GenerationProvider 抽象
│   │   ├── llm/
│   │   │   ├── openai_compat.py # 覆盖 DeepSeek/千问/智谱/豆包/Kimi/Gemini 兼容层
│   │   │   └── anthropic.py     # Claude messages 格式
│   │   └── image/
│   │       ├── gpt_image.py     # 同步返回 b64
│   │       ├── nano_banana.py   # Gemini API
│   │       ├── seedream.py      # Ark 异步任务（提交→轮询）
│   │       └── grok.py
│   ├── core/
│   │   ├── kb.py                # 知识库加载 + 校验（pydantic）
│   │   ├── recommend.py         # 组装 prompt → LLM → JSON 校验（id 白名单/数量/重试）
│   │   ├── generate.py          # 编排：生成 → 质检 → 重试 → 分级补图
│   │   ├── qc.py                # VLM 评分（复用 AnalysisProvider）
│   │   └── report.py            # Pillow 拼版报告卡
│   └── server.py                # FastAPI
├── cli.py                       # MVP 先行：推荐/生成/跑回归全走 CLI
└── pyproject.toml
```

### 9.3 架构：标准分层（Ports & Adapters / 六边形架构）

依赖方向向内，核心业务不感知任何厂商：

```
cli.py / server.py        ← 应用层（入口，薄）
        ↓
core/                     ← 领域层（recommend/generate/qc/report，纯业务编排）
        ↓ 依赖抽象接口
providers/base.py         ← 端口（Protocol）
        ↑ 实现端口
providers/llm/ image/     ← 基础设施层（各厂商适配器）
```

**端口只有两个**（对应 2.1 配置）：

```python
class AnalysisProvider(Protocol):
    async def analyze(self, prompt: str, image: bytes,
                      json_schema: type[BaseModel]) -> BaseModel

class GenerationProvider(Protocol):
    async def edit(self, prompt: str, images: list[bytes], size: str) -> bytes
    # 异步任务型（如 seedream）在适配器内部轮询到完成，对上统一 async
```

**一切 JSON 有 schema**：推荐输出、质检评分、知识库（含 `schema_version`）、配置全用 pydantic 模型——LLM 输出校验失败自动带错误信息重试一次，这是推荐链路稳定性的关键。

### 9.4 可靠性与可观测性（高可用）

| 机制 | 标准做法 |
|---|---|
| 重试 | `tenacity` 指数退避 + 抖动；按错误类型分策略（429 限流退避重试 / 4xx 参数错误不重试 / 超时重试） |
| 超时 | 每个 provider 独立超时配置；生图轮询设总时长上限 |
| **provider 故障转移** | 配置中声明 fallback 链（如 gpt-image → nano-banana），主 provider 连续失败自动切换并记录——单厂商故障不影响可用性 |
| 降级 | 质检环节失败不阻塞交付：跳过 QC 直接返回并在结果中标记"未质检"——QC 增强质量但不在关键路径上 |
| job 持久化 | SQLite（WAL 模式，`aiosqlite`）存 job 状态——进程重启任务不丢，比内存 dict 高一档可用性，又不引入消息队列 |
| 幂等 | job 带 client 提供的 idempotency key，重试提交不重复扣调用 |
| 结构化日志 | `structlog` JSON 日志，每次模型调用记录：provider/model/耗时/token 数/图片张数/估算花费——成本和延迟可回溯 |
| 健康检查 | `/healthz`（进程）+ `/healthz/providers`（各 provider 连通性） |

### 9.5 工程质量与维护性

| 维度 | 标准做法 |
|---|---|
| 代码规范 | `ruff`（lint + format）+ `mypy --strict`，pre-commit 强制 |
| 测试分层 | 单测（core 纯逻辑，mock 端口）/ **适配器契约测试**（录制真实响应 fixture，验证各 provider 适配器解析正确，改 API 可快速定位）/ 回归测试（测试脸集跑推荐+生成，护栏行为） |
| CI | GitHub Actions：lint + type + test 三道门；回归测试手动/ nightly 触发（花 API 费） |
| 配置 | `pydantic-settings` + `.env`（12-factor）；API key 只引用环境变量名，永不入库 |
| 依赖管理 | `uv` + `pyproject.toml` 标准打包，lockfile 锁版本 |
| 文档 | README + 本设计文档 + 各 JSON 自带 `references` 来源字段；接口文档由 FastAPI 自动生成 OpenAPI |

### 9.6 落地方式

第 1-2 步（建库 + 推荐链路）纯 CLI 验证，零 Web 代码；第 4 步才加 FastAPI 和页面。`regression/benchmark.py` 即第 8.4 节基准测试榜的生产工具。
