from pyedflib import highlevel


def read_entire_datasource(file_name):
    signals, signal_headers, header = highlevel.read_edf(file_name)

    for data, header in zip(signals, signal_headers):
        label = header['label']
        print(f"Found Feature: {label}")

    return signals


def read_datasource_by_name(file_name, label):
    signals, signal_headers, header = highlevel.read_edf(file_name)
    idx = [header['label'] for header in signal_headers].index(label)
    return signals[idx]
