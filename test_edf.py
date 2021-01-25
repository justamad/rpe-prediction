from pyedflib import highlevel
import pyedflib
import numpy as np
import matplotlib.pyplot as plt

signals, signal_headers, header = highlevel.read_edf("data/14-10-50.EDF")
# print(signal_headers)
# print(signals)

for data, header in zip(signals, signal_headers):
    label = header['label']
    plt.plot(data, label=label)

plt.legend()
plt.show()
