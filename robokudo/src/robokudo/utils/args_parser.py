import argparse
import re
import sys

from robokudo.defs import PACKAGE_NAME


def parse_args() -> argparse.Namespace:
    """Parse the RoboKudo CLI args and return them."""
    parser = argparse.ArgumentParser(prefix_chars="_")
    parser.add_argument(
        "_ae",
        dest="ae",
        type=str,
        nargs="?",
        const=1,
        default="demo",
        help="Analysis Engine to run (module name in descriptors/analysis_engines/).",
    )
    parser.add_argument(
        "_ros_pkg",
        dest="ros_pkg",
        type=str,
        nargs="?",
        const=1,
        default=PACKAGE_NAME,
        help="ROS package name containing the AE (default: robokudo).",
    )
    parser.add_argument(
        "_headless",
        action="store_true",
        help="Boolean parameter to define, if RoboKudo should run with a GUI (true) or headless ("
        "false).",
    )

    parser.add_argument(
        "_debugmode",
        action="store_true",
        help="If set, the rcply root logger will be set to DEBUG log level which will yield many ROS-related debug messages.",
    )

    # Input has to be like '_vis=web,o3d,cv' spaces have to be stripped and split by comma
    parser.add_argument(
        "_vis",
        action="store",
        type=str,
        default="o3d, cv, sharedros, allannotatorsros",
        help="A comma separated list of visualizers to load (invalid visualizers are ignored). "
        "Possible values are: web, o3d, cv, sharedros, allannotatorsros",
    )

    parser.add_argument(
        "_nodesuffix",
        dest="nodesuffix",
        type=str,
        nargs="?",
        const=1,
        default="",
        help="A suffix to add to the ROS node name.",
    )

    parser.add_argument(
        "_tickrate",
        dest="tickrate",
        type=int,
        nargs="?",
        const=1,
        default=5,
        help="Rate (Hz) to tick the Behavior Tree.",
    )

    parser.set_defaults(headless=False)
    parser.set_defaults(debugmode=False)

    args = parser.parse_args()

    if not re.match("^[A-Za-z0-9_-]*$", args.ae):
        print(
            f"Invalid AE name supplied: {args.ae} contains chars other than alphanum, - and _."
        )
        sys.exit(1)

    args.vis = args.vis.replace(" ", "").split(",") if args.vis else args.vis
    args = parser.parse_args()
    return args
