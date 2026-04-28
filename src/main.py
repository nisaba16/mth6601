import argparse
import os
import logging
import json

logging.getLogger().setLevel(logging.WARN)

def main():
    """Main entry point of the program."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run taxi simulation scenarios.")
    parser.add_argument(
        "-sn", "--scenario",
        type=str,
        help="Scenario to run (e.g., TP1, TP2, TP3, TP4_scenario, TP4).",
    )
    args = parser.parse_args()
    config_file = "src/run_test/inputs.json"  # Replace with your JSON file path
    # Load configuration file
    if not os.path.isfile(config_file):
        logging.error(f"Configuration file '{config_file}' does not exist.")
        return

    with open(config_file, 'r') as file:
        config_data = json.load(file)

    task_type = config_data.get("task_type")

    if task_type == "scenarios":
        # Lazy import: running scenarios requires the simulation/solver stack.
        from src.run_test.run_tests import run_scenarios
        SCENARIOS = {scenario["scenario"]: scenario["parameters"] for scenario in config_data.get("scenarios", [])}
        run_scenarios(args.scenario, SCENARIOS)
    elif task_type == "single_test":
        # Lazy import: running a test requires the simulation/solver stack.
        from src.run_test.run_tests import run_single_test
        run_single_test(config_data["single_test"])
    elif task_type == "create_plot":
        # Lazy import: plotting should not require optional solvers (e.g., Gurobi).
        from src.run_test.create_plots import handle_create_plot
        handle_create_plot(config_data["create_plot"], args.scenario)
    elif task_type == "create_instance":
        from src.run_test.create_instance import create_instances
        create_instances(config_data["create_instance"])
    else:
        logging.error(f"Invalid task type '{task_type}'. Must be 'single_test' or 'scenarios'.")


if __name__ == "__main__":
    main()
