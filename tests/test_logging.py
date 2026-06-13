import os
import sys
import unittest

import numpy as np
import torch as th


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils.logging import Logger


class _ConsoleLogger(object):
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


class LoggerTest(unittest.TestCase):
    def test_log_stat_converts_tensor_and_numpy_scalars(self):
        console = _ConsoleLogger()
        logger = Logger(console)

        logger.log_stat("grad_norm", th.tensor(2.5), 10)
        logger.log_stat("return_mean", np.float32(3.5), 10)
        logger.log_stat("episode", 1, 10)
        logger.print_recent_stats()

        self.assertIsInstance(logger.stats["grad_norm"][0][1], float)
        self.assertIsInstance(logger.stats["return_mean"][0][1], float)
        self.assertIn("grad_norm:", console.messages[-1])

    def test_log_stat_rejects_non_scalar_tensors(self):
        logger = Logger(_ConsoleLogger())

        with self.assertRaises(ValueError):
            logger.log_stat("invalid", th.ones(2), 10)


if __name__ == "__main__":
    unittest.main()
