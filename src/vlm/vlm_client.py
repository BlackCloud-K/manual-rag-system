from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

SYSTEM_PROMPT = (
    "你是一个技术文档/软件界面插图分析助手。请严格只描述图片中实际可见的内容，"
    "禁止推断或补充图中未显示的信息。用中文回答。"
)

USER_PROMPT = (
    "用一句中文描述这张图片的主体内容，重点保留图中出现的关键英文标签和中文标注文字"
    "最多列举5个最重要的，控制在100字以内。只描述图中可见内容，不推断图中未显示的信息。"
    "如果图中文字模糊无法确认，只描述可清晰辨认的内容，不猜测或编造标签名称。\n\n"
    "如果图中没有可辨认的文字标注，用简短的位置或类型描述代替，例如：座舱内部右侧面板特写、导弹实物侧视图。\n\n"
    "输出格式要求：直接描述内容，不要以\"这张图片展示了\"、\"图片中显示\"等开头。\n"
    "示例输出：驾驶舱音频控制面板，标注有INTERCOM、TACAN、ILS、HOT MIC开关。"
)

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 5.0


async def get_image_caption(base64_image: str, client: AsyncOpenAI) -> tuple[str, int]:
    """Call GPT-4o-mini with one image and return (caption, total_tokens)."""
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    },
                ],
                temperature=0.2,
                max_tokens=200,
            )

            caption = (response.choices[0].message.content or "").strip()
            total_tokens = int(getattr(response.usage, "total_tokens", 0) or 0)

            if not caption:
                caption = "CAPTION_FAILED"
            return caption, total_tokens
        except Exception:
            if attempt == MAX_RETRIES - 1:
                return "CAPTION_FAILED", 0
            backoff = BASE_BACKOFF_SECONDS * (2**attempt)
            await asyncio.sleep(backoff)

    return "CAPTION_FAILED", 0
