from __future__ import annotations

import contextlib
import io
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


if __name__ == "__main__":
    unittest.main()

