import os
import unittest


def test_suite():
    suite = unittest.TestLoader().discover(os.path.dirname(__file__))
    return suite
