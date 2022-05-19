# See https://github.com/rodluger/starry/blob/master/notebooks/run_notebooks.py
import glob
import json
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

MATPLOTLIB_HACK = """
plt.show() # HACK"""


def run(infile, outfile=None, timeout=2400):
    print("Executing %s..." % infile)

    if outfile is None:
        outfile = infile

    # Open the notebook
    with open(infile, "r") as f:
        nb = nbformat.read(f, as_version=4)

    # HACK: Replace input in certain cells
    for cell in nb.get("cells", []):

        # Suppress undesired matplotlib output
        if cell["source"].endswith(";"):
            cell["source"] += MATPLOTLIB_HACK

    # Execute the notebook
    if nb["metadata"].get("nbsphinx_execute", True):
        ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
        ep.preprocess(
            nb,
            {"metadata": {"path": os.path.dirname(os.path.abspath(infile))}},
        )

    # HACK: Replace input in certain cells
    for cell in nb.get("cells", []):

        # Custom replacements
        replace_input_with = cell["metadata"].get("replace_input_with", None)
        if replace_input_with is not None:
            cell["source"] = replace_input_with

        # Suppress undesired matplotlib output
        cell["source"] = cell["source"].replace(MATPLOTLIB_HACK, "")

    # Write it back
    with open(outfile, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


if __name__ == "__main__":
    # Run the notebooks
    files = glob.glob(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.ipynb")
    )
    for infile in files:
        run(infile)
