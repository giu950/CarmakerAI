"""Load the Bernina scenario and the locally installed CarMaker Python API."""
import os
import math
import pathlib
import sys

api_path = os.environ.get(
    "CARMAKER_PYTHON_PATH", "C:/IPG/carmaker/win64-11.0/Python/python3.6"
)
sys.path.append(api_path)

try:
    import cmapi
except ImportError as exc:
    raise ImportError(
        "CarMaker API not found. Set CARMAKER_PYTHON_PATH to your installation's "
        "Python/python3.6 directory."
    ) from exc

PROJECT_PATH = pathlib.Path(os.environ.get(
    "CARMAKER_PROJECT", str(pathlib.Path(__file__).resolve().parents[1] / "carmaker")
))
TESTRUN = os.environ.get("CARMAKER_TESTRUN", "berninaPass")
DEFAULT_TERMINATION_TIME = 180.0


def termination_time_seconds(value):
    """Validate the episode duration for both Python callers and CLI arguments."""
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("termination time must be a finite positive number of seconds")
    return value


async def make_variations(termination_time=DEFAULT_TERMINATION_TIME):
    termination_time = termination_time_seconds(termination_time)
    cmapi.Project.load(PROJECT_PATH)
    testrun = cmapi.Project.instance().load_testrun_parametrization(pathlib.Path(TESTRUN))
    testrun.set_parameter_value("DrivMan.0.TimeLimit", termination_time)
    variation = cmapi.Variation.create_from_testrun(testrun.clone())
    variation.set_name("BerninaPPO")
    return [variation]
