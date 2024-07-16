import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "font.size": 11,
})

TEXT_WIDTH_CM = 15.2286
TEXT_WIDTH_INCH = TEXT_WIDTH_CM * 0.3937007874  # convert cm to inch
HALF_PLOT_SIZE = TEXT_WIDTH_INCH * 0.49  # Value in inches, divided by subfigure size (two images in one line)
CUT_OFF = 0.02  # Padding around tight layout

DPI = 300  # Value in dpi
LINE_WIDTH = 1.0  # Strength of drawn objects
BLOB_SIZE = 2.0

COLOR_MAP = 'brg'


def get_colors(num_colors: int, palette: str = None):
    if num_colors <= 0:
        raise ValueError("Number of colors must be greater than 0")

    if palette is None:
        palette = COLOR_MAP

    cmap = plt.get_cmap(palette)
    indices = np.linspace(0, 1, num_colors)
    return [cmap(idx) for idx in indices]
