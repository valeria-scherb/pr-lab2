#!/usr/bin/env python
"""
Unit tests for the program
"""

import unittest
import main


class TestMain(unittest.TestCase):

    def test_read_train(self):
        m = main.Main()
        train = m.read_file('train.json')
        self.assertIn('inside', train)
        self.assertIn('outside', train)
        self.assertIsInstance(train['inside'], list)
        self.assertIsInstance(train['outside'], list)
        self.assertGreater(len(train['inside']), 0)
        self.assertGreater(len(train['outside']), 0)

    def test_read_classify(self):
        m = main.Main()
        classify = m.read_file('classify.json')
        self.assertIsInstance(classify, list)
        self.assertGreater(len(classify), 0)

    def test_kernels(self):
        m = main.Main()
        self.assertEqual(m.ker([2, 3]), [2*2, 2*3, 3*2, 3*3, 2, 3, 1])
        self.assertEqual(m.eigker([2, 3]), [2*2, 2*3, 3*2, 3*3, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
