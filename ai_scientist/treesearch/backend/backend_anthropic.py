import time
import os
import re

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import anthropic


ANTHROPIC_TIMEOUT_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.APIStatusError,
)


def get_ai_client(model: str, max_retries=2):
    if model.startswith("anthropic."):
        return anthropic.AnthropicBedrock(max_retries=max_retries)
    return anthropic.Anthropic(max_retries=max_retries)


def _convert_content(content):
    if not isinstance(content, list):
        return content
    converted = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "image_url":
            converted.append(block)
            continue
        url = block.get("image_url", {}).get("url", "")
        match = re.fullmatch(r"data:([^;]+);base64,(.+)", url, re.DOTALL)
        if not match:
            raise ValueError("Anthropic image input must be a base64 data URL")
        converted.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": match.group(1),
                    "data": match.group(2),
                },
            }
        )
    return converted


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    client = get_ai_client(model_kwargs.get("model"), max_retries=0)

    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 8192  # default for Claude models

    if func_spec is not None:
        filtered_kwargs["tools"] = [
            {
                "name": func_spec.name,
                "description": func_spec.description,
                "input_schema": func_spec.json_schema,
            }
        ]
        filtered_kwargs["tool_choice"] = {"type": "tool", "name": func_spec.name}

    # Anthropic doesn't allow not having a user messages
    # if we only have system msg -> use it as user msg
    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message

    # Anthropic passes the system messages as a separate argument
    if system_message is not None:
        filtered_kwargs["system"] = system_message

    messages = opt_messages_to_list(None, _convert_content(user_message))

    t0 = time.time()
    message = backoff_create(
        client.messages.create,
        ANTHROPIC_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0
    if func_spec is not None:
        tool_blocks = [block for block in message.content if block.type == "tool_use"]
        if not tool_blocks:
            raise ValueError(
                "Anthropic response did not contain the requested tool call"
            )
        output = tool_blocks[0].input
    else:
        output = "\n".join(
            block.text for block in message.content if block.type == "text"
        )

    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
    }

    return output, req_time, in_tokens, out_tokens, info
