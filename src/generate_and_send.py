#!/usr/bin/env python3
"""
generate_and_send.py (支持 Gemini via google-genai, OpenAI, 或通用 generic HTTP endpoints)

环境变量（主要）:
- FEISHU_WEBHOOK: 飞书自定义机器人 Webhook（必填）
- LLM_PROVIDER: "openai" / "gemini" / "generic"（若未设置，会基于其他变量自动推断）
- LLM_API_KEY: 用于 openai 或 generic (Bearer) 的 key / token
- LLM_API_URL: generic provider 的 endpoint（当 LLM_PROVIDER=generic 时需要）
- GEMINI_API_KEY: 若使用 google-genai 的 API key（可选，优先于 LLM_API_KEY）
- GEMINI_MODEL: 要调用的 Gemini 模型（可选，默认 "gemini-1.5"）
"""
from __future__ import annotations
import os
import json
import time
import logging
from typing import Optional

import requests
from dotenv import load_dotenv

# Try OpenAI new/old clients
HAS_OPENAI_V1 = False
HAS_OPENAI_OLD = False
try:
    from openai import OpenAI  # new client
    HAS_OPENAI_V1 = True
except Exception:
    try:
        import openai  # legacy
        HAS_OPENAI_OLD = True
    except Exception:
        pass

# Try google-genai client availability will be checked at runtime in call_gemini
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Environment / config
FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower() or None
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_URL = os.getenv("LLM_API_URL")
# Gemini-specific vars (prefer these if present)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", os.getenv("LLM_MODEL", "gemini-1.5"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# If LLM_API_URL is set but provider is openai or empty, auto-switch to generic
if LLM_API_URL and (LLM_PROVIDER in (None, "", "openai")):
    logging.info("Detected LLM_API_URL is set; forcing LLM_PROVIDER='generic'")
    LLM_PROVIDER = "generic"

# If GEMINI_API_KEY present and provider not set, use gemini
if GEMINI_API_KEY and not LLM_PROVIDER:
    logging.info("Detected GEMINI_API_KEY; setting LLM_PROVIDER='gemini'")
    LLM_PROVIDER = "gemini"

# Default provider fallback
if not LLM_PROVIDER:
    LLM_PROVIDER = "openai"  # keep backward compatibility
logging.info("LLM_PROVIDER=%s", LLM_PROVIDER)


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


# ------------------------------
# Gemini (google-genai) 调用
# ------------------------------
def call_gemini(api_key: str, model: str, prompt: str, temperature: float = 0.8) -> str:
    """
    使用 google-genai 客户端调用 Gemini（参考 daily_stock_analysis 中的做法）。
    - api_key: google genai API key（通常以 'AIza...' 或其他形式）
    - model: 模型名（如 gemini-1.5 / gemini-2.5 等）
    返回字符串（LLM 的原始文本）
    """
    try:
        from google import genai
    except Exception as e:
        raise RuntimeError(
            "google-genai not installed. Install with: pip install google-genai\n" f"orig: {e}"
        )

    try:
        logging.info("Calling Gemini via google-genai (model=%s)", model)
        client = genai.Client(api_key=api_key)
        # daily_stock_analysis 使用 generate_content -> try to be compatible
        # Some genai versions use client.models.generate() or client.models.generate_content()
        # Try generate_content first, fall back to generate
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            # response may have .text or dict-like structure
            if hasattr(response, "text"):
                return response.text
            # if it's dict-like
            try:
                return response["candidates"][0]["content"] if "candidates" in response else json.dumps(response, ensure_ascii=False)
            except Exception:
                return json.dumps(response, ensure_ascii=False)
        except AttributeError:
            # try alternative API
            response = client.models.generate(model=model, prompt=prompt)
            if hasattr(response, "text"):
                return response.text
            try:
                # try to extract from returned structure
                return response["candidates"][0]["content"] if "candidates" in response else json.dumps(response, ensure_ascii=False)
            except Exception:
                return json.dumps(response, ensure_ascii=False)
    except Exception as e:
        # rethrow with context
        raise RuntimeError(f"Gemini call failed: {e}")


# ------------------------------
# OpenAI 调用（兼容新版/旧版 SDK）
# ------------------------------
def call_openai_chat(api_key: str, model: str, prompt: str, temperature: float = 0.8) -> str:
    if not api_key:
        raise RuntimeError("OpenAI API key not provided")
    if HAS_OPENAI_V1:
        logging.info("Using openai>=1.0.0 client")
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个帮助用户输出 JSON 数据的助手，严格按要求返回 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=800,
        )
        try:
            return resp.choices[0].message["content"]
        except Exception:
            return json.dumps(resp, ensure_ascii=False)
    elif HAS_OPENAI_OLD:
        logging.info("Using legacy openai client")
        import openai as _openai  # type: ignore
        _openai.api_key = api_key
        resp = _openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个帮助用户输出 JSON 数据的助手，严格按要求返回 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=800,
        )
        try:
            return resp.choices[0].message.content
        except Exception:
            return json.dumps(resp, ensure_ascii=False)
    else:
        raise RuntimeError("openai package is not installed. Install openai>=1.0.0 or legacy client.")


# ------------------------------
# Generic HTTP LLM 调用（适用于自建服务或 Vertex endpoint）
# ------------------------------
def call_generic_llm(api_key: str, api_url: str, prompt: str, temperature: float = 0.8) -> str:
    if not api_url:
        raise RuntimeError("LLM_API_URL must be set when using generic provider.")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": os.getenv("LLM_MODEL", ""),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    logging.info("Calling generic LLM API: %s", api_url)
    r = requests.post(api_url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    # 简单解析常见字段
    if isinstance(data, dict):
        if "choices" in data and data["choices"]:
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
                if isinstance(val, list) and val:
                    if isinstance(val[0], dict):
                        for k in ("content", "text", "output"):
                            if k in val[0]:
                                return val[0][k]
                    elif isinstance(val[0], str):
                        return val[0]
    return json.dumps(data, ensure_ascii=False)


# ------------------------------
# 辅助函数：解析 / 格式化 / 发送到飞书
# ------------------------------
def parse_json_from_text(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
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
    payload = {"msg_type": "text", "content": {"text": text}}
    logging.info("Sending message to Feishu webhook")
    r = requests.post(webhook, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    try:
        r.raise_for_status()
    except Exception as e:
        logging.error("Feishu webhook returned status %s: %s", r.status_code, r.text)
        raise
    logging.info("Message sent to Feishu successfully.")


# ------------------------------
# 主流程
# ------------------------------
def main():
    if not FEISHU_WEBHOOK:
        logging.error("FEISHU_WEBHOOK 环境变量未设置，退出。")
        return 2

    # Determine which credentials to use
    # Priority for Gemini: GEMINI_API_KEY -> LLM_API_KEY
    gemini_key = GEMINI_API_KEY or LLM_API_KEY

    if LLM_PROVIDER == "gemini":
        if not gemini_key:
            logging.error("LLM_PROVIDER=gemini，但未提供 GEMINI_API_KEY 或 LLM_API_KEY，退出。")
            return 3
    elif LLM_PROVIDER == "openai":
        if not LLM_API_KEY:
            logging.error("LLM_PROVIDER=openai，但未提供 LLM_API_KEY（OpenAI key），退出。")
            return 3
    elif LLM_PROVIDER == "generic":
        if not LLM_API_URL:
            logging.error("LLM_PROVIDER=generic，但未设置 LLM_API_URL，退出。")
            return 3
        if not LLM_API_KEY:
            logging.warning("LLM_PROVIDER=generic，但未设置 LLM_API_KEY（可能是无鉴权服务或 token 将在 runtime 注入）")

    text = None
    last_err = None
    for attempt in range(3):
        try:
            # if LLM_PROVIDER == "gemini":
            #     # call gemini
            text = call_gemini(gemini_key, GEMINI_MODEL, PROMPT_USER, temperature=0.8)
            # elif LLM_PROVIDER == "openai":
            #     text = call_openai_chat(LLM_API_KEY, os.getenv("LLM_MODEL", "gpt-3.5-turbo"), PROMPT_USER, temperature=0.8)
            # else:
            #     # generic
            #     text = call_generic_llm(LLM_API_KEY, LLM_API_URL, PROMPT_USER, temperature=0.8)

            logging.debug("LLM raw response: %s", (text or "")[:1000])
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
    try:
        send_to_feishu(FEISHU_WEBHOOK, f"每日知识点任务失败：{last_err}")
    except Exception:
        pass
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
