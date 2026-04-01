from src.solvers.offline_solver import OfflineSolver
from src.utilities.enums import DestroyMethod
from src.solvers.solver import Solver
from src.utilities.config import SimulationConfig
from typing import Any, Dict, List



class ReOptimizer(Solver):
    """Re-optimize vehicle routing and trip-route assignment using destroy-and-repair (LNS) ideas.

    Includes three destroy methods:
        1. destroy_fix_arrival_times
        2. destroy_fix_variables
        3. destroy_bonus

    Attributes:
    ------------
        destroy_method : DestroyMethod(Enum)
            Method used for destruction in the LNS algorithm.
        initial_solution : dict
            Initial values for decision variables (X, Y, Z, U, assignment_dict).

        vehicle_request_assign : Dict[int, VehicleState]
            Mapping vehicle id to VehicleState (inherited from Solver). Each state holds: vehicle,
            assigned_requests, departure_stop, departure_time, last_stop, last_stop_time, assign_possible,
            random_number; used to save assignments and build route plans.

        durations : dictionary
            travel time matrix between possible stop points
            example: for duration between destination of trip_i and the origin of trip_j use:
                     self.durations[trip_i.destination.label][trip_i.origin.label]

        costs: dictionary
            driving costs (it works based on location ids like durations)
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
                 simulation_config: SimulationConfig) -> None:
        super().__init__(network, vehicles, simulation_config)
        self.initial_solution : Dict[str, Any] = {}
        self.destroy_method = simulation_config.algorithm_params["destroy_method"]


    def re_optimizer(self, K, P_not_served, rejected_trips):
        """Re-optimize the solution using destroy and repair (LNS).

        Input:
        ------------
            K : set of vehicles
            P_not_served : set of customers not yet served
            rejected_trips : list of trips rejected in the optimization process.

        Steps:
            1. Create the mathematical model (OfflineSolver).
            2. If initial_solution exists, destroy (fix) part of it according to destroy_method.
            3. Repair (re-optimize) with the MIP solver and extract solution.
            4. Save the final solution to initial_solution.
        """

        # Create and configure the offline model
        offline_model = OfflineSolver(self.network, self.objective)
        offline_model.create_model(K, P_not_served, self.vehicle_request_assign)

        if self.initial_solution:
            if self.destroy_method == DestroyMethod.FIX_ARRIVALS:
                # destroy by fixing a time window around the arrival times in the initial solution
                self.destroy_fix_arrival_times(P_not_served, offline_model)

            elif self.destroy_method == DestroyMethod.FIX_VARIABLES:
                # destroy by fixing some of the variables based on the initial solution
                self.destroy_fix_variables(K, P_not_served, offline_model)
            elif self.destroy_method == DestroyMethod.BONUS:
                # destroy the solution by your suggested function
                self.destroy_bonus(K, P_not_served, offline_model)

        # add objective
        offline_model.define_objective(K, P_not_served, self.vehicle_request_assign)

        # solve and get solution
        offline_model.solve()
        offline_model.extract_solution(K, P_not_served, rejected_trips, self.vehicle_request_assign)

        self.save_solution(offline_model)

    def save_solution(self, offline_model: OfflineSolver):
        """Save the solution from the offline model into initial_solution.

        Input:
        ------------
            offline_model : OfflineSolver instance (Gurobi MIP model).
        """
        # Extracting values of decision variables
        self.Y: Dict[Any, Dict[Any, float]] = {
            key[0]: {sub_key[1]: var.X for sub_key, var in offline_model.Y_var.items() if sub_key[0] == key[0]}
            for key in offline_model.Y_var.keys()
        }
        self.X: Dict[Any, Dict[Any, float]] = {
            key[0]: {sub_key[1]: var.X for sub_key, var in offline_model.X_var.items() if sub_key[0] == key[0]}
            for key in offline_model.X_var.keys()
        }
        self.Z: Dict[Any, float] = {key: var.X for key, var in offline_model.Z_var.items()}
        self.U: Dict[Any, float] = {key: var.X for key, var in offline_model.U_var.items()}

        assignment_dict: Dict[Any, Dict[str, List[Any]]] = {
            vehicle_id: {
                'assigned_requests': state.assigned_requests
            }
            for vehicle_id, state in self.vehicle_request_assign.items()
        }

        self.initial_solution = {
            'X': self.X,
            'Y': self.Y,
            'U': self.U,
            'Z': self.Z,
            'assignment_dict': assignment_dict
        }

    def destroy_fix_arrival_times(self, P, offline_model: OfflineSolver) -> None:
        """Fix pickup times to a time window around values in initial_solution (e.g. ±2 min).

        Input:
        ------------
            P : set of customers to serve
            offline_model : OfflineSolver instance (Gurobi MIP model).

        """
        """you should write your code here ..."""


        prev_u = self.initial_solution["U"]

        delta = 120.0

        for req in P:
            req_id = req.id
            center = prev_u.get(req_id)
            if center is None:
                continue
            if req_id not in offline_model.U_var:
                continue

            lb = max(req.ready_time, center - delta)
            ub = min(req.latest_pickup, center + delta)

            # Just to make sure that the window isn't empty
            if lb > ub:
                continue

            offline_model.U_var[req_id].LB = lb
            offline_model.U_var[req_id].UB = ub

        offline_model.model.update()


    def destroy_fix_variables(self, K, P, offline_model: OfflineSolver):
        """ Fix some of Y_var, X_var variables based on the initial solution.

        Input:
        ------------
            K : set of vehicles
            P : set of customers to serve
            offline_model : OfflineSolver instance (Gurobi model).

        Hint:
            - Forbid the arcs that goes from one request to another one that were in different vehicle
            - Forbid the arcs that goes from departing node of a vehicle to other requests that were in different
                  vehicle
        """
        """you should write your code here ..."""

        assignment_dict = self.initial_solution["assignment_dict"]

        # request_to_vehicle[request_id] = vehicle_id from previous solution
        request_to_vehicle: Dict[Any, Any] = {}
        for vehicle_id, data in assignment_dict.items():
            for req in data.get("assigned_requests", []):
                request_to_vehicle[req.id] = vehicle_id
        
        for i in P:
            veh_i = request_to_vehicle.get(i.id)
            if veh_i is None: #if the request was not assigned initially, we don't care
                continue
            for j in P:
                if i.id == j.id:#same request
                    continue
                veh_j = request_to_vehicle.get(j.id)
                if veh_j is None: #if the request was not assigned initially, we don't care
                    continue
                if veh_j!=veh_i and (i.id, j.id) in offline_model.X_var:
                    offline_model.X_var[i.id, j.id].UB = 0.0
                    offline_model.X_var[i.id, j.id].LB = 0.0
        for k in K:
            for i in P:
                prev_vehicle = request_to_vehicle.get(i.id)
                if prev_vehicle is None:
                    continue
                if k.id != prev_vehicle and (k.id, i.id) in offline_model.Y_var:
                    offline_model.Y_var[k.id, i.id].UB = 0.0
                    offline_model.Y_var[k.id, i.id].LB = 0.0

        offline_model.model.update()


        


    def destroy_bonus(self, K, P, offline_model: OfflineSolver):
        """Custom destroy method

        Input:
        ------------
            K : set of vehicles
            P : set of customers to serve
            offline_model : OfflineSolver instance (Gurobi model).

        Hint:
            - Include comments where necessary to explain your proposed function
            - you can use any of the inputs if required
        """
        """you should write your code here ..."""

