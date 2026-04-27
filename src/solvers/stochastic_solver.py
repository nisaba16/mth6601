from copy import deepcopy
from collections import defaultdict
from src.solvers.offline_solver import OfflineSolver
from src.solvers.solver import Solver
from src.utilities.enums import ConsensusParams
from src.utilities.create_scenario import create_random_requests
from typing import Any, Dict, List
from src.utilities.config import SimulationConfig



class StochasticSolver(Solver):
    """Online stochastic solution: scenario generation plus offline solve per scenario.

    Two types of consensus:
        - qualitative: aggregate which request is “best” across scenarios (e.g. by voting).
        - quantitative: use scenario objective values to score best request–vehicle assignments.

    Attributes:
    ------------
        consensus_type : ConsensusParams(Enum)
            Type of consensus algorithm.
        nb_scenario : int
            Number of scenarios to generate and solve at each step.
        scenario_param : Dict[str, Any]
            Parameters for scenario generation (time_window, cust_node_hour, known_portion, advance_notice).

        vehicle_request_assign : Dict[int, VehicleState]
            Mapping vehicle id to VehicleState (inherited from Solver). Each state holds: vehicle,
            assigned_requests, departure_stop, departure_time, last_stop, last_stop_time, assign_possible,
            random_number; used to save assignments and build route plans.

        network : Any
            The road network, including nodes representing stop points.
        durations : Dict
            Travel time matrix between stop points (e.g. self.durations[from_label][to_label]).
        costs : Dict
            Driving cost matrix (same structure as durations).
        algorithm: Algorithm(Enum)
            The optimization algorithm utilized for planning and assigning trips to vehicles.
        objective: Objectives(Enum)
            The objective used to evaluate the effectiveness of the plan (e.g., maximizing profit or minimizing wait time).
        objective_value: float
            The objective value from served requests.
        total_customers_served: int
            The count of customers successfully served.
    """

    def __init__(self,
                 network: Any,
                 vehicles: List[Any],
                 simulation_config: SimulationConfig):
        super().__init__(network, vehicles, simulation_config)
        self.consensus_type = simulation_config.algorithm_params["consensus_param"]
        self.nb_scenario = simulation_config.algorithm_params["nb_scenario"]

        self.scenario_param = {
            'time_window': simulation_config.time_window,       # Time window for picking up the requests
            'cust_node_hour': simulation_config.algorithm_params["cust_node_hour"],  # the average rate of customers per node (in the network) per hour
            'known_portion': simulation_config.known_portion,    # percentage of requests that are known in advance
            'advance_notice': simulation_config.advance_notice,  # Fixed amount of time (in minutes) the requests are released before their ready time.
        }


    def stochastic_solver(self, K, P_not_assigned, current_time):
        """Assign ride requests to vehicles using scenario-based consensus.

        Input:
        ------------
            K : set of vehicles
            P_not_assigned : set of customers not yet assigned to be served
            current_time : current time (used for scenario generation).
        """

        # Step 1: assign requests to the vehicles/ routes
        P_not_assigned = sorted(P_not_assigned, key=lambda x: x.ready_time)
        assigned_requests = []

        if self.consensus_type == ConsensusParams.QUANTITATIVE:
            assigned_requests = self.quantitative_consensus(K, P_not_assigned, current_time)

        elif self.consensus_type == ConsensusParams.QUALITATIVE:
            assigned_requests = self.qualitative_consensus(K, P_not_assigned, current_time)

        # Step 2: check the feasibility of the solution
        self.create_online_solution()
        if self.verify_constraints(K, assigned_requests):
            self.calc_objective_value(K, P_not_assigned)
            self.total_customers_served = sum(1 for f_i in P_not_assigned if self.Z[f_i.id])

        else:
            raise ValueError("The solution is not feasible")

    def qualitative_consensus(self, K, P_not_assigned, current_time):
        """Assign requests using qualitative consensus over scenario solutions.

        Input:
        ------------
            K : set of vehicles
            P_not_assigned : set of customers not yet assigned to be served
            current_time : current time (for scenario generation).

        Output:
        ------------
            assigned_requests : list of assigned requests.

        Hint:
            - Generate multiple scenarios (use create_random_requests from create_scenario.py) and solve each with OfflineSolver.
            - Use the scenario solutions to decide which request to assign to which vehicle (e.g. aggregate
              information across scenarios).
            - Assign one request at a time and update vehicle state before the next.
        """
        assigned_requests = []
        scenario_start_id = 10_000_000

        # Boucle extérieure sur les requêtes réelles non assignées.
        for request in P_not_assigned:
            vehicle_votes = defaultdict(int)

            # Boucle intérieure sur les scénarios.
            for scenario_idx in range(self.nb_scenario):
                scenario = create_random_requests(
                    network=self.network,
                    cust_node_hour=self.scenario_param["cust_node_hour"],
                    start_ID=scenario_start_id,
                    start_time=current_time,
                    durations=self.durations,
                    time_window=self.scenario_param["time_window"],
                    known_portion=self.scenario_param["known_portion"],
                    advance_notice=self.scenario_param["advance_notice"],
                )
                scenario_start_id += len(scenario) + 1

                # Le modèle est résolu sur la requête courante + le scénario stochastique.
                P_combined = [request] + scenario
                scenario_vehicle_assign = deepcopy(self.vehicle_request_assign)

                offline_model = OfflineSolver(self.network, self.objective)
                offline_model.create_model(K, P_combined, scenario_vehicle_assign)
                offline_model.define_objective(K, P_combined, scenario_vehicle_assign)
                offline_model.solve()

                rejected_trips = []
                offline_model.extract_solution(K, P_combined, rejected_trips, scenario_vehicle_assign)

                # Extraire le véhicule qui a reçu la 1re requête de la solution du scénario.
                for vehicle_id, state in scenario_vehicle_assign.items():
                    if not state.assigned_requests:
                        continue
                    if state.assigned_requests[0].id == request.id:
                        vehicle_votes[vehicle_id] += 1
                        break

            if not vehicle_votes:
                continue

            selected_vehicle_id = max(vehicle_votes, key=vehicle_votes.get)
            selected_state = self.vehicle_request_assign[selected_vehicle_id]

            # Incrémente un compteur de requêtes pour le véhicule sélectionné.
            current_counter = getattr(selected_state, "request_counter", 0)
            setattr(selected_state, "request_counter", current_counter + 1)

            self.assign_trip_to_vehicle(selected_state, request)
            assigned_requests.append(request)
        return (assigned_requests)



    def quantitative_consensus(self, K, P_not_assigned, current_time):
        """Assign requests using quantitative consensus (score by scenario objective values).

        Input:
        ------------
            K : set of vehicles
            P_not_assigned : set of customers not yet assigned to be served
            current_time : current time (for scenario generation).

        Output:
        ------------
            assigned_requests : list of assigned requests.

        Hint:
            - Generate multiple scenarios (use create_random_requests from create_scenario.py) and solve each with OfflineSolver.
            - Use each scenario’s objective value to score or rank request–vehicle options; then choose
              assignments.
            - Assign one request at a time and update vehicle state before the next.
        """
        assigned_requests = []
        scenario_start_id = 10_000_000

        # Boucle extérieure sur les requêtes réelles non assignées.
        for request in P_not_assigned:
            vehicle_votes = defaultdict(int)

            # Boucle intérieure sur les scénarios.
            for scenario_idx in range(self.nb_scenario):
                scenario = create_random_requests(
                    network=self.network,
                    cust_node_hour=self.scenario_param["cust_node_hour"],
                    start_ID=scenario_start_id,
                    start_time=current_time,
                    durations=self.durations,
                    time_window=self.scenario_param["time_window"],
                    known_portion=self.scenario_param["known_portion"],
                    advance_notice=self.scenario_param["advance_notice"],
                )
                scenario_start_id += len(scenario) + 1

                # Le modèle est résolu sur la requête courante + le scénario stochastique.
                P_combined = [request] + scenario
                scenario_vehicle_assign = deepcopy(self.vehicle_request_assign)

                offline_model = OfflineSolver(self.network, self.objective)
                offline_model.create_model(K, P_combined, scenario_vehicle_assign)
                offline_model.define_objective(K, P_combined, scenario_vehicle_assign)
                offline_model.solve()

                rejected_trips = []
                offline_model.extract_solution(K, P_combined, rejected_trips, scenario_vehicle_assign)

                # Extraire le véhicule qui a reçu la 1re requête de la solution du scénario.
                for vehicle_id, state in scenario_vehicle_assign.items():
                    if not state.assigned_requests:
                        continue
                    if state.assigned_requests[0].id == request.id:
                        vehicle_votes[vehicle_id] += offline_model.objective_value
                        break

            if not vehicle_votes:
                continue

            selected_vehicle_id = max(vehicle_votes, key=vehicle_votes.get)
            selected_state = self.vehicle_request_assign[selected_vehicle_id]

            # Incrémente un compteur de requêtes pour le véhicule sélectionné.
            current_counter = getattr(selected_state, "request_counter", 0)
            setattr(selected_state, "request_counter", current_counter + 1)

            self.assign_trip_to_vehicle(selected_state, request)
            assigned_requests.append(request)
        return (assigned_requests)

