import matplotlib
import matplotlib.pyplot as plt
import warnings
from IPython import get_ipython

# Pretty figures
get_ipython().magic('config InlineBackend.figure_format = "retina"')


# Disable annoying font warnings
matplotlib.font_manager._log.setLevel(50)


# Style
plt.style.use("default")
plt.rcParams["savefig.dpi"] = 100
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["font.size"] = 14
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
plt.rcParams["font.cursive"] = ["Liberation Sans"]
try:
    plt.rcParams["mathtext.fallback"] = "cm"
except KeyError:
    plt.rcParams["mathtext.fallback_to_cm"] = True
