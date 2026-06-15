#!/usr/bin/env python3
"""Cursor-shaped payloads must survive normalization before vLLM."""

import unittest

from gateway.context_trim import (
    clamp_max_tokens,
    sanitize_tool_chain,
    trim_body_for_vllm,
)
from llm import chat_completions


class VllmPayloadTest(unittest.TestCase):
    env = {"MAX_MODEL_LEN": "32768", "REFINE_MAX_TOKENS": "2048"}

    def test_clamp_max_tokens_zero(self):
        self.assertEqual(clamp_max_tokens(0, self.env), 2048)
        self.assertEqual(clamp_max_tokens(None, self.env), 2048)
        self.assertEqual(clamp_max_tokens("0", self.env), 2048)

    def test_cursor_agent_body(self):
        body = {
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "max_tokens": 0,
            "max_completion_tokens": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [
                {"role": "system", "content": "You are an agent."},
                {"role": "user", "content": "read VoiceChatManager.kt"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_abc", "content": "file text"},
                {"role": "user", "content": "summarize it"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "read_file", "parameters": {"type": "object"}},
                }
            ],
            "tool_choice": "auto",
        }
        out = trim_body_for_vllm(body, self.env)
        self.assertGreaterEqual(out["max_tokens"], 1)
        self.assertNotIn("max_completion_tokens", out)
        self.assertNotIn("stream_options", out)
        self.assertFalse(out["stream"])
        self.assertTrue(out["messages"])
        roles = [m["role"] for m in out["messages"]]
        self.assertIn("user", roles)

    def test_strips_cursor_only_fields(self):
        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "store": True,
            "metadata": {"cursor": "junk"},
        }
        out = trim_body_for_vllm(body, self.env)
        self.assertNotIn("stream_options", out)
        self.assertNotIn("store", out)
        self.assertNotIn("metadata", out)
        self.assertNotIn("max_completion_tokens", out)

    def test_orphan_tool_dropped(self):
        msgs = [
            {"role": "tool", "tool_call_id": "orphan", "content": "x"},
            {"role": "user", "content": "hi"},
        ]
        out = sanitize_tool_chain(msgs)
        self.assertEqual([m["role"] for m in out], ["user"])

    def test_incomplete_tool_calls_stripped_before_user(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"role": "user", "content": "next"},
        ]
        out = sanitize_tool_chain(msgs)
        self.assertNotIn("tool_calls", out[0])

    def test_llm_chat_completions_clamps_max_tokens(self):
        captured: dict = {}
        import os
        import urllib.request

        def fake_urlopen(req, timeout=600):
            captured["payload"] = __import__("json").loads(req.data.decode())

            class Resp:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    pass

                def read(self):
                    return b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'

            return Resp()

        old_urlopen = urllib.request.urlopen
        old_gate = os.environ.get("GATEWAY_ON_POD")
        os.environ["GATEWAY_ON_POD"] = "1"
        urllib.request.urlopen = fake_urlopen
        try:
            chat_completions(
                [{"role": "user", "content": "hi"}],
                max_tokens=0,
                env={
                    **self.env,
                    "RUNPOD_API_KEY": "test",
                    "MODEL_NAME": "test",
                },
            )
        finally:
            urllib.request.urlopen = old_urlopen
            if old_gate is None:
                os.environ.pop("GATEWAY_ON_POD", None)
            else:
                os.environ["GATEWAY_ON_POD"] = old_gate

        self.assertGreaterEqual(captured["payload"]["max_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
