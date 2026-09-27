"""
Module Name: test_run_screening.py
Aim: Verify job-description input validation and case-insensitive category lookup.
Input: Built-in category names and combinations of CLI-equivalent arguments.
Output: Passing or failing unittest results.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import run_screening


class TestRunScreening(unittest.TestCase):
    def test_resolve_job_description_from_category_case_insensitive(self):
        jd_text = run_screening.resolve_job_description(jd_category="information-technology", jd_file=None)
        self.assertIn("software systems", jd_text.lower())

    def test_resolve_job_description_requires_exactly_one_input(self):
        with self.assertRaises(ValueError):
            run_screening.resolve_job_description(jd_category=None, jd_file=None)

        with self.assertRaises(ValueError):
            run_screening.resolve_job_description(jd_category="INFORMATION-TECHNOLOGY", jd_file="custom.txt")


if __name__ == "__main__":
    unittest.main()
