from scripts.tests.workload_test import run_test

import sys

sys.path.append("scripts/tests")

cuda = None
gen = None
mem = None
net = None

if __name__ == "__main__":
    if len(sys.argv) > 0:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        step = int(sys.argv[3])
        iterations = int(sys.argv[4])
        run_test(start, end, step, iterations, sys.argv[5:])
    else:
        run_test(500, 1100, 100, 100, [])