import random
from typing import Any, List
import logging

from src.utilities.config import SimulationConfig
from src.utilities.enums import Algorithm, Objectives
from src.solvers.solver import Solver

logger = logging.getLogger(__name__)


class OnlineSolver(Solver):
    """Provide online solution to optimize the vehicle routing and the trip-route assignment. This
    method includes:
        1. greedy solver
        2. random solver
        3. ranking solver

    Attributes:
    ------------
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

        X: Dict[int, Dict[int, bool]]
            Binary variables indicating if customer i is picked immediately after j by a taxi.
        Y: Dict[int, Dict[int, bool]]
            Binary variables indicating if customer i is picked up by vehicle k as the first customer.
        U: Dict[int, float]
            Pickup times for customers.
        Z: Dict[int, bool]
            Binary variables indicating if customer i is selected to be served.
    """

    def __init__(self,
                 network: Any,
                 vehicles: List[Any],
                 simulation_config: SimulationConfig):
        super().__init__(network, vehicles, simulation_config)
        # Assign random numbers to each vehicle request assignment
        for veh_id, state in self.vehicle_request_assign.items():
            state.random_number = random.random()

    def determine_available_vehicles(self, trip):
        """Determine whether each vehicle can feasibly serve the trip (e.g. reach pickup by latest_pickup).

        Input:
        ------------
            trip : ride request to serve

        Hint:
            - Iterate over self.vehicle_request_assign and set assign_possible for each vehicle.
            - Use the parent class method calc_reach_time(veh_info, trip).
        """
        for veh_id, veh_info in self.vehicle_request_assign.items():
            # A vehicle can serve the trip only if it can reach the pickup
            # location no later than the customer's latest acceptable pickup time.
            reach_time = self.calc_reach_time(veh_info, trip)
            veh_info.assign_possible = (reach_time <= trip.latest_pickup)


    def online_solver(self, K, P_not_assigned, rejected_trips):
        """Find a solution to assign ride requests to vehicles after arrival.

        Input:
        ------------
            K : set of vehicles
            P_not_assigned : set of customers that are not assigned to be served
            rejected_trips : list of trips rejected in the optimization process.

        Steps:
            1. Assign requests to vehicles/routes based on the chosen algorithm.
            2. Check the feasibility of the solution.
        """

        # Step 1: assign requests to the vehicles/ routes
        sorted_requests = sorted(P_not_assigned, key=lambda x: x.ready_time)

        if self.algorithm == Algorithm.GREEDY:
            assigned_requests = self.greedy_assign(sorted_requests, rejected_trips)
        elif self.algorithm == Algorithm.RANDOM:
            assigned_requests = self.random_assign(sorted_requests, rejected_trips)
        elif self.algorithm == Algorithm.RANKING:
            assigned_requests = self.ranking_assign(sorted_requests, rejected_trips)
        else:
            logger.error("Unsupported algorithm: %s", self.algorithm)
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        # Step 2: check the feasibility of then solution
        self.create_online_solution()
        if self.verify_constraints(K, assigned_requests):
            self.calc_objective_value(K, sorted_requests)
            self.total_customers_served = sum(1 for f_i in P_not_assigned if self.Z[f_i.id])
            logger.info("Online solver solution is feasible. Served %d customers.", self.total_customers_served)

        else:
            logger.error("The solution is not feasible.")
            raise ValueError("The solution is not feasible")

    def greedy_assign(self, P_not_assigned: List[Any], rejected_trips: List[Any]) -> List[Any]:
        """Assign ride requests to vehicles using a greedy method (e.g. best objective per request).

        Input:
        ------------
            P_not_assigned : set of customers that are not assigned to be served
            rejected_trips : list of trips rejected in the optimization process.

        Output:
        ------------
            assigned_requests : list of assigned requests

        Hint:
            - for each trip in P_not_assigned you have to select a vehicle to assign or reject the request.
            - evaluating the feasibility of assigning a trip to a vehicle should be done inside "determine_available_vehicles" function.
            - If no vehicle is available, append the trip to rejected_trips.
            - if a vehicle is selected to assign a request:
                - Use the assign_trip_to_vehicle function to assign the task to the selected vehicle
                - add trip to the list of assigned_requests
        """
        # for each request find the best insertion position
        assigned_requests = []

        for trip in P_not_assigned:
            # Mark which vehicles can feasibly pick up this trip in time.
            self.determine_available_vehicles(trip)

            # Among feasible vehicles, pick the one with the earliest reach
            # time (arrives at pickup soonest).  This minimises both customer
            # wait and empty driving, and keeps vehicles well-positioned for
            # future requests.
            best_veh_info = None
            best_reach_time = float("inf")

            for veh_id, veh_info in self.vehicle_request_assign.items():
                if veh_info.assign_possible:
                    reach_time = self.calc_reach_time(veh_info, trip)
                    if reach_time < best_reach_time:
                        best_reach_time = reach_time
                        best_veh_info = veh_info

            if best_veh_info is None:
                # No feasible vehicle found — reject this trip.
                rejected_trips.append(trip)
            else:
                self.assign_trip_to_vehicle(best_veh_info, trip)
                assigned_requests.append(trip)

        return assigned_requests

    def random_assign(self, P_not_assigned: List[Any], rejected_trips: List[Any]) -> List[Any]:
        """Assign ride requests to vehicles based on random solution method

        Input:
        ------------
            P_not_assigned : set of customers that are not assigned to be served
            rejected_trips : list of trips rejected in the optimization process.

        Output:
        ------------
            assigned_requests : list of assigned requests

        Hint:
            - for each trip in P_not_assigned you have to select a vehicle to assign or reject the request.
            - evaluating the feasibility of assigning a trip to a vehicle should be done inside "determine_available_vehicles" function.
            - If no vehicle is available, append the trip to rejected_trips.
            - if a vehicle is selected to assign a request:
                - Use the assign_trip_to_vehicle function to assign the task to the selected vehicle
                - add trip the list of assigned_requests
        """
        # for each request find the best insertion position
        assigned_requests = []

        for trip in P_not_assigned:
            # Mark which vehicles can feasibly pick up this trip in time.
            self.determine_available_vehicles(trip)

            # Collect all feasible vehicles then pick one uniformly at random.
            feasible_vehicles = [
                veh_info
                for veh_info in self.vehicle_request_assign.values()
                if veh_info.assign_possible
            ]

            if not feasible_vehicles:
                # No feasible vehicle found — reject this trip.
                rejected_trips.append(trip)
            else:
                chosen_veh_info = random.choice(feasible_vehicles)
                self.assign_trip_to_vehicle(chosen_veh_info, trip)
                assigned_requests.append(trip)

        return assigned_requests

    def ranking_assign(self, P_not_assigned: List[Any], rejected_trips: List[Any]) -> List[Any]:
        """Assign ride requests to vehicles by ranking solution method.

        Input:
        ------------
            P_not_assigned : set of customers that are not assigned to be served
            rejected_trips : list of trips rejected in the optimization process.

        Output:
        ------------
            assigned_requests : list of assigned requests

        Hint:
            - for each trip in P_not_assigned you have to select a vehicle to assign or reject the request.
            - evaluating the feasibility of assigning a trip to a vehicle should be done inside "determine_available_vehicles" function.
            - If no vehicle is available, append the trip to rejected_trips.
            - if a vehicle is selected to assign a request:
                - Use the assign_trip_to_vehicle function to assign the task to the selected vehicle
                - add trip the list of assigned_requests

        """
        # for each request find the best insertion position
        assigned_requests = []

        for trip in P_not_assigned:
            # Mark which vehicles can feasibly pick up this trip in time.
            self.determine_available_vehicles(trip)

            # Among feasible vehicles, pick the one with the highest
            # random_number (assigned once at initialisation).  This creates a
            # fixed global priority ranking that is consistent across requests,
            # unlike the purely random selection in random_assign.
            best_veh_info = None
            best_rank = -1.0

            for veh_id, veh_info in self.vehicle_request_assign.items():
                if veh_info.assign_possible:
                    rank = veh_info.random_number if veh_info.random_number is not None else 0.0
                    if rank > best_rank:
                        best_rank = rank
                        best_veh_info = veh_info

            if best_veh_info is None:
                # No feasible vehicle found — reject this trip.
                rejected_trips.append(trip)
            else:
                self.assign_trip_to_vehicle(best_veh_info, trip)
                assigned_requests.append(trip)

        return assigned_requests
