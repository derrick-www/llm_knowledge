#!/usr/bin/env python3
"""
generate_and_send.py (updated for openai>=1.0.0)

兼容：
- openai>=1.0.0 (使用 OpenAI client 的 chat.completions.create)
- 旧版 openai (fallback 到 openai.ChatCompletion.create)
"""
import os
import json
import time
import logging
from typing import Optional

import requests
from dotenv import load_dotenv

# Try to import the new OpenAI client (openai>=1.0.0)
HAS_OPENAI_V1 = False
HAS_OPENAI_OLD = False
try:
    from openai import OpenAI  # new SDK
    HAS_OPENAI_V1 = True
except Exception:
    try:
        import openai  # old SDK
        HAS_OPENAI_OLD = True
    except Exception:
        pass

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "generic"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_API_URL = os.getenv("LLM_API_URL")  # only for generic provider
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

PROMPT_USER = (
    "请**随机**给出一个关于“大模型（large models）”的知识点。"
    " 返回内容必须是严格的 JSON（不要包含其他文本），格式如下："
    '{"topic": "...", "explanation": "...", "example": "...", "source": "..."}。'
    " 各字段说明：\n"
    "- topic: 简短的主题标题（不超过 8 个汉字）\n"
    "- explanation: 简洁解释（2-4 句）\n"
    "- example: 一个简单示例或类比（1-2 句）\n"
    "- source: 如果有参考链接或关键词，可放链接或简短来源说明；没有则空字符串\n"
    "请保证输出是单纯的 JSON 对象，且能被标准 JSON 解析。"
)


def call_openai_chat(api_key: str, model: str, prompt: str, temperature: float = 0.8) -> str:
    """
    支持两种情况：
    - openai>=1.0.0: 使用 from openai import OpenAI -> client.chat.completions.create(...)
    - 旧版 openai: 使用 openai.ChatCompletion.create(...)
    返回字符串（LLM 原始文本）。
    """
    if LLM_PROVIDER.lower() != "openai":
        raise RuntimeError("LLM_PROVIDER is not set to 'openai' for call_openai_chat.")

    if HAS_OPENAI_V1:
        logging.info("Using openai>=1.0.0 client")
        client = OpenAI(api_key=api_key)
        # new chat completions API
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个帮助用户输出 JSON 数据的助手，严格按要求返回 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=800,
        )
        # Try to extract text from response
        try:
            # resp.choices[0].message.content is typical
            return resp.choices[0].message["content"]
        except Exception:
            # fallback: convert to JSON string
            return json.dumps(resp, ensure_ascii=False)
    elif HAS_OPENAI_OLD:
        logging.info("Using legacy openai client")
        openai.api_key = api_key  # type: ignore
        resp = openai.ChatCompletion.create(  # type: ignore
            model=model,
            messages=[
                {"role": "system", "content": "你是一个帮助用户输出 JSON 数据的助手，严格按要求返回 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=800,
        )
        try:
            return resp.choices[0].message.content  # type: ignore
        except Exception:
            return json.dumps(resp, ensure_ascii=False)
    else:
        raise RuntimeError("openai package is not installed. Install openai>=1.0.0 or the legacy package.")


def call_generic_llm(api_key: str, api_url: str, prompt: str, temperature: float = 0.8) -> str:
    """
    通用 LLM 调用（适用于 Gemini / Vertex / 其它提供商）
    根据具体提供商调整请求体或解析。
    """
    if not api_url:
        raise RuntimeError("LLM_API_URL must be set when using generic provider.")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    logging.info("Calling generic LLM API: %s", api_url)
    r = requests.post(api_url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    # 尝试解析常见格式
    if isinstance(data, dict):
        if "choices" in data and len(data["choices"]) > 0:
            first = data["choices"][0]
            if isinstance(first.get("message"), dict):
                return first["message"].get("content", "") or ""
            if "text" in first:
                return first.get("text", "")
        for key in ("output", "candidates", "text", "generated_text", "content", "result"):
            if key in data:
                val = data[key]
                if isinstance(val, str):
                    return val
                if isinstance(val, list) and len(val) > 0:
                    if isinstance(val[0], dict):
                        # try common fields
                        for k in ("content", "text", "output"):
                            if k in val[0]:
                                return val[0][k]
                    elif isinstance(val[0], str):
                        return val[0]
    return json.dumps(data, ensure_ascii=False)


def parse_json_from_text(text: str) -> Optional[dict]:
    # try direct json parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # try to extract first {...} substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def format_message(data: dict) -> str:
    topic = data.get("topic", "").strip()
    explanation = data.get("explanation", "").strip()
    example = data.get("example", "").strip()
    source = data.get("source", "").strip()
    parts = []
    if topic:
        parts.append(f"主题：{topic}")
    if explanation:
        parts.append(f"解释：{explanation}")
    if example:
        parts.append(f"举例：{example}")
    if source:
        parts.append(f"参考：{source}")
    return "\n".join(parts)


def send_to_feishu(webhook: str, text: str) -> None:
    if not webhook:
        raise RuntimeError("FEISHU_WEBHOOK is not set.")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    payload = {
        "msg_type": "text",
        "content": {"text": text}
    }
    logging.info("Sending message to Feishu webhook")
    r = requests.post(webhook, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    try:
        r.raise_for_status()
    except Exception as e:
        logging.error("Feishu webhook returned status %s: %s", r.status_code, r.text)
        raise
    logging.info("Message sent to Feishu successfully.")


def main():
    if not FEISHU_WEBHOOK:
        logging.error("FEISHU_WEBHOOK 环境变量未设置，退出。")
        return 2
    if not LLM_API_KEY:
        logging.error("LLM_API_KEY 环境变量未设置，退出。")
        return 3

    text = None
    last_err = None
    for attempt in range(3):
        try:
            # if LLM_PROVIDER.lower() == "openai":
            #     text = call_openai_chat(LLM_API_KEY, LLM_MODEL, PROMPT_USER, temperature=0.8)
            # else:
            text = call_generic_llm(LLM_API_KEY, LLM_API_URL, PROMPT_USER, temperature=0.8)
            logging.debug("LLM raw response: %s", text)
            parsed = parse_json_from_text(text)
            if parsed:
                message = format_message(parsed)
                logging.info("Parsed JSON and formatted message:\n%s", message)
                send_to_feishu(FEISHU_WEBHOOK, message)
                return 0
            else:
                last_err = f"无法从 LLM 响应中解析出 JSON，响应文本：{(text or '')[:400]}"
                logging.warning("Attempt %d: %s", attempt + 1, last_err)
        except Exception as e:
            last_err = str(e)
            logging.exception("Attempt %d failed: %s", attempt + 1, last_err)
        time.sleep(2 + attempt * 2)

    logging.error("所有尝试失败：%s", last_err)
    # optionally: send failure notice to feishu
    try:
        send_to_feishu(FEISHU_WEBHOOK, f"每日知识点任务失败：{last_err}")
    except Exception:
        pass
    return 1


if __name__ == "__main__":
    exit(main())
