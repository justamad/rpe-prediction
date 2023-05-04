import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{libertine}",
    "font.family": "serif",
    # "font.serif": ["Palatino"],
    "font.serif": ["Libertine"],
    "font.size": 9,
})

cm = 1/2.54  # Convert centimeters to inches
text_width = 17.7917
column_width = 8.47415
dpi = 300
line_width = 1.0
blob_size = 10.0
