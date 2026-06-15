#!/usr/bin/env python3
import unittest

from gateway.context_trim import sanitize_tool_chain, trim_body_for_vllm, trim_conversation_tail


class ContextTrimTest(unittest.TestCase):
    def test_trim_skips_when_under_budget(self):
        msgs = [
            {"role": "system", "content": "x" * 5000},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        out = trim_conversation_tail(msgs, budget_tokens=100_000)
        self.assertEqual(len(out), 3)
        self.assertEqual(len(out[0]["content"]), 5000)

    def test_stream_includes_usage(self):
        from gateway.router import stream_text_chunks

        completion = {
            "id": "x",
            "model": "test",
            "created": 1,
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
        }
        chunks = stream_text_chunks(completion, include_usage=True)
        self.assertTrue(any('"prompt_tokens": 100' in c for c in chunks))

    def test_trims_long_user(self):
        msgs = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "x" * 200000},
        ]
        out = trim_conversation_tail(msgs, budget_tokens=1000)
        total = sum(len(m.get("content", "")) for m in out)
        self.assertLess(total, 5000)

    def test_trim_body_for_vllm(self):
        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "y" * 300000}],
            "max_tokens": 0,
        }
        out = trim_body_for_vllm(body, env={"MAX_MODEL_LEN": "32768", "REFINE_MAX_TOKENS": "2048"})
        self.assertGreater(out["max_tokens"], 0)
        self.assertLess(len(out["messages"][0]["content"]), 300000)

    def test_max_completion_tokens(self):
        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 0,
        }
        out = trim_body_for_vllm(body, env={"MAX_MODEL_LEN": "32768", "REFINE_MAX_TOKENS": "2048"})
        self.assertGreater(out["max_tokens"], 0)
        self.assertNotIn("max_completion_tokens", out)

    def test_sanitize_orphan_tool(self):
        msgs = [
            {"role": "tool", "tool_call_id": "x", "content": "orphan"},
            {"role": "user", "content": "hi"},
        ]
        out = sanitize_tool_chain(msgs)
        self.assertEqual(out[0]["role"], "user")

    def test_sanitize_mid_chain_orphan_tool(self):
        msgs = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "wrong", "content": "orphan"},
            {"role": "user", "content": "next"},
        ]
        out = sanitize_tool_chain(msgs)
        roles = [m["role"] for m in out]
        self.assertNotIn("tool", roles)
        self.assertNotIn("tool_calls", out[1])

    def test_sanitize_incomplete_tool_calls(self):
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "user", "content": "next"},
        ]
        out = sanitize_tool_chain(msgs)
        self.assertNotIn("tool_calls", out[0])


if __name__ == "__main__":
    unittest.main()
