#!/usr/bin/env python3
import unittest

from gateway.context_trim import trim_body_for_vllm, trim_conversation_tail


class ContextTrimTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
