"""
Run the local scan refiner without installing an additional package.
"""

# %% Entry point
from scan_refiner.command_line import CommandLine

if __name__ == "__main__":
    CommandLine.run()
