from src.solvers.offline_solver import OfflineSolver
from src.solvers.solver import Solver, VehicleState
from src.utilities.enums import ConsensusParams
from src.utilities.create_scenario import create_random_requests
from typing import Any, Dict, List
from src.utilities.config import SimulationConfig



class StochasticSolver(Solver):
    """Online stochastic solution: scenario generation plus offline solve per scenario.

    Two types of consensus:
        - qualitative: aggregate which request is "best" across scenarios (e.g. by voting).
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

    def _build_vehicle_state_copy(self, K):
        """Create a shallow copy of vehicle states for use in a scenario solve.

        We copy only the positional fields (departure_stop, departure_time,
        last_stop, last_stop_time) needed by OfflineSolver, and start each
        vehicle with an empty assignment list so the scenario solve is clean.
        """
        veh_assign_copy = {}
        for veh in K:
            state = self.vehicle_request_assign[veh.id]
            new_state = VehicleState(vehicle=veh)
            new_state.departure_stop = state.departure_stop
            new_state.departure_time = state.departure_time
            new_state.last_stop = state.last_stop
            new_state.last_stop_time = state.last_stop_time
            new_state.assigned_requests = []
            veh_assign_copy[veh.id] = new_state
        return veh_assign_copy

    def _solve_scenario(self, K, P_combined, veh_assign_copy):
        """Solve a single scenario with OfflineSolver.

        Returns the solved OfflineSolver instance, or None if the solve failed.
        Each scenario uses a fresh Gurobi model so scenarios are independent.
        """
        offline_model = OfflineSolver(self.network, self.objective)
        rejected = []
        try:
            offline_model.offline_solver(K, P_combined, veh_assign_copy, rejected)
        except Exception:
            return None
        return offline_model

    def _greedy_assign_by_score(self, K, P_not_assigned, scores):
        """Assign requests to vehicles one at a time in descending score order.

        At each iteration we pick the (vehicle, request) pair with the highest
        score that is still feasible (vehicle can reach the request in time).
        We then update the vehicle state so subsequent assignments respect the
        new last-stop position.  A score of 0 or less means no scenario ever
        recommended this pair, so we stop.
        """
        assigned_requests = []
        assigned_ids = set()
        remaining = list(P_not_assigned)

        while remaining:
            best_score = 0
            best_veh = None
            best_req = None

            for veh in K:
                veh_info = self.vehicle_request_assign[veh.id]
                for req in remaining:
                    if req.id in assigned_ids:
                        continue
                    reach_time = self.calc_reach_time(veh_info, req)
                    if reach_time <= req.latest_pickup:
                        score = scores[veh.id].get(req.id, 0)
                        if score > best_score:
                            best_score = score
                            best_veh = veh
                            best_req = req

            # No feasible (vehicle, request) pair with positive score — stop.
            if best_req is None or best_score <= 0:
                break

            veh_info = self.vehicle_request_assign[best_veh.id]
            self.assign_trip_to_vehicle(veh_info, best_req)
            assigned_requests.append(best_req)
            assigned_ids.add(best_req.id)
            remaining = [r for r in remaining if r.id not in assigned_ids]

        return assigned_requests

    def qualitative_consensus(self, K, P_not_assigned, current_time):
        """Assign requests using qualitative consensus over scenario solutions.

        For each scenario we solve the combined (real + random future) request
        pool optimally.  For each vehicle we then check which *real* request
        was scheduled as its first trip in that optimal solution.  A counter
        for that (vehicle, request) pair is incremented by 1 (vote).

        After all scenarios the pair with the most votes is assigned first,
        vehicle state is updated, and the process repeats until no positively-
        scored feasible assignment remains.

        Input:
        ------------
            K : set of vehicles
            P_not_assigned : set of customers not yet assigned to be served
            current_time : current time (for scenario generation).

        Output:
        ------------
            assigned_requests : list of assigned requests.
        """
        if not K or not P_not_assigned:
            return []

        # scores[veh.id][req.id] = number of scenarios that assigned req first to veh
        scores = {veh.id: {req.id: 0 for req in P_not_assigned} for veh in K}

        # Use a large offset for scenario request IDs to avoid clashing with
        # real request IDs (which are small integers encoded as strings).
        base_start_id = 1_000_000

        for s in range(self.nb_scenario):
            start_id = base_start_id + s * 10_000

            # Generate random future requests (required parameters only, per TP note 2).
            try:
                scenario_requests = create_random_requests(
                    network=self.network,
                    cust_node_hour=self.scenario_param['cust_node_hour'],
                    start_ID=start_id,
                    start_time=current_time,
                    durations=self.durations,
                    time_window=self.scenario_param['time_window'],
                )
            except Exception:
                continue

            P_combined = list(P_not_assigned) + scenario_requests

            veh_assign_copy = self._build_vehicle_state_copy(K)

            offline_model = self._solve_scenario(K, P_combined, veh_assign_copy)
            if offline_model is None:
                continue

            # For each vehicle, find which real request (if any) is its first trip
            # in the scenario solution (Y_var[veh_id, req_id] == 1) and cast a vote.
            for veh in K:
                for req in P_not_assigned:
                    key = (veh.id, req.id)
                    if key not in offline_model.Y_var:
                        continue
                    try:
                        if offline_model.Y_var[key].X > 0.5:
                            scores[veh.id][req.id] += 1
                    except Exception:
                        pass

        return self._greedy_assign_by_score(K, P_not_assigned, scores)


    def quantitative_consensus(self, K, P_not_assigned, current_time):
        """Assign requests using quantitative consensus (score by scenario objective values).

        Like qualitative consensus but instead of incrementing by 1, we credit
        the (vehicle, request) pair by the optimal objective value of that
        scenario.  This gives more weight to scenarios where the chosen first
        assignment leads to a high-value overall solution.

        Input:
        ------------
            K : set of vehicles
            P_not_assigned : set of customers not yet assigned to be served
            current_time : current time (for scenario generation).

        Output:
        ------------
            assigned_requests : list of assigned requests.
        """
        if not K or not P_not_assigned:
            return []

        # scores[veh.id][req.id] = cumulative objective value from scenarios
        # where req was assigned as the first trip of veh.
        scores = {veh.id: {req.id: 0.0 for req in P_not_assigned} for veh in K}

        base_start_id = 1_000_000

        for s in range(self.nb_scenario):
            start_id = base_start_id + s * 10_000

            try:
                scenario_requests = create_random_requests(
                    network=self.network,
                    cust_node_hour=self.scenario_param['cust_node_hour'],
                    start_ID=start_id,
                    start_time=current_time,
                    durations=self.durations,
                    time_window=self.scenario_param['time_window'],
                )
            except Exception:
                continue

            P_combined = list(P_not_assigned) + scenario_requests

            veh_assign_copy = self._build_vehicle_state_copy(K)

            offline_model = self._solve_scenario(K, P_combined, veh_assign_copy)
            if offline_model is None:
                continue

            obj_value = offline_model.objective_value
            # Only credit scenarios that yielded a positive objective (profitable).
            # A zero or negative value would penalise good assignments.
            if obj_value <= 0:
                continue

            for veh in K:
                for req in P_not_assigned:
                    key = (veh.id, req.id)
                    if key not in offline_model.Y_var:
                        continue
                    try:
                        if offline_model.Y_var[key].X > 0.5:
                            scores[veh.id][req.id] += obj_value
                    except Exception:
                        pass

        return self._greedy_assign_by_score(K, P_not_assigned, scores)
