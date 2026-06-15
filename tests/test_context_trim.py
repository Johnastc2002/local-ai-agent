#!/usr/bin/env python3
import unittest

from gateway.context_trim import estimate_tokens, trim_messages


class ContextTrimTest(unittest.TestCase):
    def test_trims_long_user(self):
        msgs = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "x" * 200000},
        ]
        out = trim_messages(msgs, budget_tokens=1000)
        self.assertLess(estimate_tokens(out[-1]["content"]), 1100)
        self.assertIn("truncated", out[-1]["content"])


if __name__ == "__main__":
    unittest.main()
