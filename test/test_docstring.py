import doctest
import os
import sys
sys.path.append(os.getcwd())
from analysis_tool import functions as func # noqa


def test_docstring():
    doctest_results = doctest.testmod(func)
    assert doctest_results.failed == 0
