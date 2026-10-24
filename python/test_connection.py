"""Check that the installed CarMaker Python API can be imported."""
import sys


def main():
    print("Python: {}".format(sys.version.split()[0]))
    print("Interpreter: {}".format(sys.executable))
    try:
        from parametrization_testrun import api_path, cmapi
    except (ImportError, OSError) as exc:
        print("CarMaker API check failed: {}".format(exc), file=sys.stderr)
        print(
            "Activate the Python 3.6 environment and set CARMAKER_PYTHON_PATH "
            "to your CarMaker 11.0 Python/python3.6 directory.",
            file=sys.stderr,
        )
        return 1

    print("API directory: {}".format(api_path))
    print("Loaded cmapi: {}".format(cmapi.__file__))
    print("Hello, CarMaker! API import succeeded.")
    print("This checks API availability only; no simulation was started.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
