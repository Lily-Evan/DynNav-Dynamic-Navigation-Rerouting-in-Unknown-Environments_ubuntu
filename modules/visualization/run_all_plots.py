from modules.visualization.plot_frontiers import plot_frontiers
from modules.visualization.plot_nbv import plot_nbv
from modules.visualization.plot_nce import plot_nce
from modules.visualization.plot_planners import plot_planners
from modules.visualization.plot_particles import plot_particles
from modules.visualization.plot_icp import plot_icp
from modules.visualization.plot_uncertainty import plot_uncertainty

def main():
    print("Generating all plots...")
    print("Frontiers:",    plot_frontiers())
    print("NBV:",          plot_nbv())
    print("NCE:",          plot_nce())
    print("Planners:",     plot_planners())
    print("Particles:",    plot_particles())
    print("ICP:",          plot_icp())
    print("Uncertainty:",  plot_uncertainty())
    print("Done. Check data/plots/")

if __name__ == "__main__":
    main()
