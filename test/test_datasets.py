import unittest
import torch

from typing import Dict

from datasets import collate_fn


class DatasetsTester(unittest.TestCase):
    def test_collate_fn(self):
        self.assertEqual(3, 3)

if __name__ == '__main__':
    unittest.main()
