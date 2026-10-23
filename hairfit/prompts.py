"""Prompt templates (design.md sections 4 & 5). Verbatim assembly from KB."""

from __future__ import annotations

import json

from hairfit.schemas import FaceShapesKB, Hairstyle, HairstylesKB

RECOMMENDATION_HEADER = """你是专业发型顾问。基于用户照片和知识库，分析面部特征并推荐发型库中最合适的发型。

分析元规则：{meta_rule}
脸型知识库：{face_kb}
发型库（只能从此列表推荐）：{styles_kb}

要求：
1. 只输出 JSON，格式如下
2. recommended 数组只能使用发型库中存在的 id，输出 3-5 个
3. reason 必须引用具体规则（如"圆脸适合顶部加高——该款顶部体积符合纵向拉伸策略"）
4. 若用户发量/发质与某款硬性不符，不得推荐该款

输出 JSON 格式：
{{
  "face_analysis": {{
    "face_shape": "round|square|heart|diamond|oblong|triangle|oval",
    "face_ratio": "约 1.4:1",
    "hair_volume": "thin|medium|thick",
    "hair_texture": "1a~4c（Andre Walker 分型）",
    "hairline": "normal|high|receding",
    "notes": "其他影响推荐的观察（眼镜、发际线形状等）"
  }},
  "recommended": [
    {{"id": "发型库中的 id", "reason": "引用规则的一句话理由"}}
  ]
}}"""

#: editing models have exactly two failure modes — design.md 5:
#: "kept identity but ignored the edit" vs "did the edit but changed the person"
QC_PROMPT = """你是图像质检员。对比两张图：图1 是用户原始照片，图2 是 AI 换发型后的结果（目标发型：{style_name}——{style_desc}）。

按编辑模型的两种失败模式评分（0-10 分）：
1. face_consistency 面部一致性：图2 的脸是否和图1 是同一个人（五官、脸型、肤色、身份特征）
2. style_match 发型匹配度：图2 的发型是否符合目标发型的描述

verdict 规则：face_consistency < 6 或 style_match < 6 时为 "retry"，否则为 "pass"。

只输出 JSON：
{{"face_consistency": 0-10, "style_match": 0-10, "verdict": "pass|retry", "reason": "一句话"}}"""

GENERATION_PROMPT = """Replace the hairstyle of the person in image 1 with the hairstyle
shown in image 2. Keep the person's face, facial features, identity,
expression, skin tone, pose, clothing, and background exactly
unchanged. Only change the hair: {description}。

If the user's hair color differs from the reference, keep the
user's original hair color."""

#: appended when regenerating after a failed QC round
RETRY_SUFFIX = """

Note: a previous attempt failed quality check for this reason: {qc_reason}.
{fix_hint}"""

FIX_HINTS = {
    "face": "CRITICAL: the face in your last output did not match the person. "
    "The facial identity (eyes, nose, mouth, face shape, skin tone) must remain "
    "pixel-faithful to image 1.",
    "style": "CRITICAL: the hairstyle in your last output did not match the "
    "reference. Follow the hairstyle of image 2 much more closely.",
}


def build_recommendation_prompt(face_kb: FaceShapesKB, styles_kb: HairstylesKB) -> str:
    return RECOMMENDATION_HEADER.format(
        meta_rule=face_kb.meta_rule,
        face_kb=json.dumps(
            {"shapes": {k: v.model_dump() for k, v in face_kb.shapes.items()}},
            ensure_ascii=False,
        ),
        styles_kb=json.dumps(
            {"items": [s.model_dump() for s in styles_kb.items]}, ensure_ascii=False
        ),
    )


def build_qc_prompt(style: Hairstyle) -> str:
    return QC_PROMPT.format(style_name=style.name, style_desc=style.description)


def build_generation_prompt(style: Hairstyle) -> str:
    return GENERATION_PROMPT.format(description=style.description)


def build_retry_suffix(qc_reason: str, weak: str) -> str:
    """weak: 'face' or 'style' — which failure mode to correct."""
    hint = FIX_HINTS["face"] if weak == "face" else FIX_HINTS["style"]
    return RETRY_SUFFIX.format(qc_reason=qc_reason, fix_hint=hint)
