from __future__ import annotations

import contextlib
import io
import json
import unittest

from utils.logger import configure_runtime_logger


class TestUtilsLogger(unittest.TestCase):
    def test_configure_runtime_logger_rebinds_stdout_stream(self) -> None:
        logger_name = "tests.runtime.logger.rebind"
        first_stdout = io.StringIO()
        second_stdout = io.StringIO()

        with contextlib.redirect_stdout(first_stdout):
            logger = configure_runtime_logger(logger_name, level="INFO", log_path=None, json_logs=False)
            logger.info("first-message")

        with contextlib.redirect_stdout(second_stdout):
            logger = configure_runtime_logger(logger_name, level="INFO", log_path=None, json_logs=False)
            logger.info("second-message")

        self.assertIn("first-message", first_stdout.getvalue())
        self.assertNotIn("second-message", first_stdout.getvalue())
        self.assertIn("second-message", second_stdout.getvalue())

    def test_json_formatter_includes_message(self) -> None:
        logger_name = "tests.runtime.logger.json"
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            logger = configure_runtime_logger(logger_name, level="INFO", log_path=None, json_logs=True)
            logger.info("source parsed: %s", "https://example.com")
        line = stream.getvalue().strip().splitlines()[-1]
        payload = json.loads(line)
        self.assertEqual(payload["message"], "source parsed: https://example.com")


if __name__ == "__main__":
    unittest.main()

