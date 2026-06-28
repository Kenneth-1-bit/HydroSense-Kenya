"""
Run the complete HydroSense test suite.
"""

import sys
import unittest


def main():

    print("=" * 60)
    print("HydroSense Kenya")
    print("Scientific Computing Project")
    print("Running Test Suite...")
    print("=" * 60)

    loader = unittest.TestLoader()

    suite = loader.discover(
        start_dir="tests",
        pattern="test_*.py"
    )

    runner = unittest.TextTestRunner(
        verbosity=2
    )

    result = runner.run(suite)

    print("\n" + "=" * 60)

    if result.wasSuccessful():

        print("ALL TESTS PASSED")
        print(f"Tests Run : {result.testsRun}")
        print("Failures  : 0")
        print("Errors    : 0")

    else:

        print("TEST SUITE FAILED")
        print(f"Tests Run : {result.testsRun}")
        print(f"Failures  : {len(result.failures)}")
        print(f"Errors    : {len(result.errors)}")

    print("=" * 60)

    sys.exit(not result.wasSuccessful())


if __name__ == "__main__":
    main()

