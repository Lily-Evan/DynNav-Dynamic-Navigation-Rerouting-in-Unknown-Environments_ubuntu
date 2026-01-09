from abstract_planner import AbstractPlanner
from navigation_controller import NavigationController, RobotMetrics

from abstract_planner import AbstractPlanner
from navigation_controller import NavigationController, RobotMetrics
from results_logger import ResultsLogger

def main():
    planner = AbstractPlanner()
    controller = NavigationController(planner)
    logger = ResultsLogger()
    step = 0


    scenario = [
        (RobotMetrics(0.1, 0.1, 0.1, 0.0), "Balanced corridor, nothing special here."),
        (RobotMetrics(0.1, 0.1, 0.1, 0.0), "There are many people and children ahead."),
        (RobotMetrics(0.7, 0.2, 0.2, 0.1), "Be careful, hidden danger around the corner."),
        (RobotMetrics(0.4, 0.3, 0.9, 0.7), "Ο διάδρομος είναι γλιστερός και επικίνδυνος."),
    ]

    for i, (metrics, msg) in enumerate(scenario, start=1):
        print(f"\n===== STEP {i} =====")
        print("Message:", msg)

        result = controller.update(metrics, msg)
        step += 1
        logger.log(step, metrics, result)


        print("\nSelf-Healing Decision:", result["self_healing"])
        print("\nLanguage Safety:", result["language_safety"])
        print("\nEthical Layer:", result["ethical"])
        print("\nTrust State:", result["trust"])
        print("\nPlanner Parameters:", result["planner_state"])




    scenario = [
        (RobotMetrics(0.1, 0.1, 0.1, 0.0), "Balanced corridor, nothing special here."),
        (RobotMetrics(0.1, 0.1, 0.1, 0.0), "There are many people and children ahead."),
        (RobotMetrics(0.7, 0.2, 0.2, 0.1), "Be careful, hidden danger around the corner."),
        (RobotMetrics(0.4, 0.3, 0.9, 0.7), "Ο διάδρομος είναι γλιστερός και επικίνδυνος."),
    ]

    for i, (metrics, msg) in enumerate(scenario, start=1):
        print(f"\n===== STEP {i} =====")
        print("Message:", msg)

        result = controller.update(metrics, msg)

        print("\nSelf-Healing Decision:", result["self_healing"])
        print("\nLanguage Safety:", result["language_safety"])
        print("\nPlanner Parameters:", result["planner_state"])


if __name__ == "__main__":
    main()
