import numpy as np
import matplotlib.pyplot as plt

class Metrics:
    def plot_coverage(self, coverage_list):
        plt.plot(coverage_list)
        plt.xlabel("Time (steps)")
        plt.ylabel("Coverage")
        plt.grid(True)
        plt.savefig("coverage_over_time.png")

    def plot_nce(self, nce_list):
        plt.plot(nce_list)
        plt.xlabel("Time (steps)")
        plt.ylabel("NCE")
        plt.grid(True)
        plt.savefig("nce_over_time.png")
