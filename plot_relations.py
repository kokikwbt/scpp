import networkx as nx
import matplotlib.pyplot as plt


def plot_relations(M, w=10, h=10, fs=20, fn=None):
    """ Function to reproduce Figure 7 """

    fig, ax = plt.subplots(figsize=(w, h))

    if fn is not None:
        fig.tight_layout()
        fig.savefig(fn)
        fig.close()
