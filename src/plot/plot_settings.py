import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "text.latex.preamble": r"\usepackage{libertine}",
    # "font.serif": ["Palatino"],
    # "font.serif": ["Libertine"],
    "font.serif": ["Computer Modern"],
    "font.size": 10,
})

cm = 1/2.54  # Convert centimeters to inches
# text_width = 17.7917
# text_width = 13.70499
text_width = 13.85803  # For thesis
# column_width = 8.47415
dpi = 300
line_width = 1.0
blob_size = 8.0
