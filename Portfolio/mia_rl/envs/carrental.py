from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
from click import Tuple
import numpy as np

@dataclass(frozen=True)
class CarRentalParams:
    max_cars_1: int = 20
    max_cars_2: int = 20
    max_moveable: int = 5
    revenue_per_rental: float = 10.0
    cost_per_moved: float = 2.0
    # lambdas: (req1, req2, ret1, ret2)
    lambdas: Tuple[float, float, float, float] = (3.0, 4.0, 3.0, 2.0)
    # truncation caps for Poisson r.v.s (last bucket is a tail bucket)
    max_requests_1: int = 8
    max_requests_2: int = 10
    max_returns_1: int = 8
    max_returns_2: int = 8

def poisson_pmf_truncated(lam: float, max_k: int) -> np.ndarray:
    """Return a Poisson(lam) pmf over k=0..max_k, where the last bucket includes the tail mass P(K>=max_k)."""
    probs = np.zeros(max_k + 1, dtype=float)
    # compute up to max_k-1 exactly (stable recurrence)
    p0 = np.exp(-lam)
    probs[0] = p0
    for k in range(1, max_k):
        probs[k] = probs[k - 1] * lam / k
    probs[max_k] = max(0.0, 1.0 - probs[:max_k].sum())
    # normalize tiny numerical error
    probs /= probs.sum()
    return probs

class CarRentalMDP:
    def __init__(self, params: CarRentalParams):
        self.params = params

        # Precompute truncated distributions (with tail bucket)
        self.req1 = poisson_pmf_truncated(params.lambdas[0], params.max_requests_1)
        self.req2 = poisson_pmf_truncated(params.lambdas[1], params.max_requests_2)
        self.ret1 = poisson_pmf_truncated(params.lambdas[2], params.max_returns_1)
        self.ret2 = poisson_pmf_truncated(params.lambdas[3], params.max_returns_2)

        # Cache location-wise outcome distributions:
        # key: (loc_id, cars_available_after_move) -> (p_next_cars, expected_rentals)
        self._loc_cache: Dict[Tuple[int, int], Tuple[np.ndarray, float]] = {}

    def states(self) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(self.params.max_cars_1 + 1) for j in range(self.params.max_cars_2 + 1)]

    def is_terminal(self, s: Tuple[int, int]) -> bool:
        # No terminal states in this continuing task
        return False

    def possible_actions(self, s: Tuple[int, int]) -> List[int]:
        """Actions are bounded by max_moveable AND by available cars / capacity."""
        n1, n2 = s
        a_min = -min(self.params.max_moveable, n2, self.params.max_cars_1 - n1)  # move from 2->1 (negative)
        a_max =  min(self.params.max_moveable, n1, self.params.max_cars_2 - n2)  # move from 1->2 (positive)
        #                 max cap in loc  | cars in loc  |  capacity in other loc
        return list(range(a_min, a_max + 1))

    def _loc_outcomes(self, loc_id: int, cars_after_move: int) -> Tuple[np.ndarray, float]:  #'What happens during the day'
        """For one location, return:
        - p_next: distribution over next cars p_next[c] = P(next_cars = c)
        - exp_rented: expected number of cars rented during the day, i.e.
            exp_rented = E[min(cars_after_move, requests)]
          where requests ~ Poisson(λ_requests) (truncated with a tail bucket).
          Notes:
          - exp_rented is an expectation (can be non-integer).
          - exp_rented is computed *before* applying returns, because returns happen after rentals.
          - exp_rented is NOT revenue; revenue is computed later as exp_rented * revenue_per_rental.

        given cars available after the move (start of the day).

        ---------------------------------------------------------
        Example interpretation of p_next:

        Suppose:
            cap = 3
            cars_after_move = 2

        Then p_next might look like:
            p_next = [0.1, 0.3, 0.4, 0.2]

        Meaning:
            P(next_cars = 0) = 0.1
            P(next_cars = 1) = 0.3
            P(next_cars = 2) = 0.4
            P(next_cars = 3) = 0.2

        These probabilities come from:
            - random rental requests (Poisson)
            - random returns (Poisson)
            - capacity cap at 'cap'
        ---------------------------------------------------------
        """
        key = (loc_id, cars_after_move)
        if key in self._loc_cache:
            return self._loc_cache[key]

        if loc_id == 1:
            req = self.req1
            ret = self.ret1
            cap = self.params.max_cars_1
        else:
            req = self.req2
            ret = self.ret2
            cap = self.params.max_cars_2

        p_next = np.zeros(cap + 1, dtype=float)
        exp_rented = 0.0

        # For each request count -> determine rentals and cars remaining before returns
        for k_req, p_req in enumerate(req): #k_req -> number of req, p_req -> probability
            rented = min(cars_after_move, k_req)
            exp_rented += p_req * rented
            cars_left = cars_after_move - rented

            # For each returns count -> next cars (with capacity cap)
            for k_ret, p_ret in enumerate(ret):
                next_cars = min(cap, cars_left + k_ret)
                p_next[next_cars] += p_req * p_ret

        # normalize tiny numeric error
        p_next /= p_next.sum()

        self._loc_cache[key] = (p_next, exp_rented)
        return p_next, exp_rented

    def after_move(self, s: Tuple[int, int], a: int) -> Tuple[int, int]:
        n1, n2 = s
        return (n1 - a, n2 + a) #number of cars at loc1 and loc2

    def expected_transition(self, s: Tuple[int, int], a: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return:
        - p_next_1: dist over next n1'
        - p_next_2: dist over next n2'
        - expected_revenue (from rentals only, not move cost)
        """
        if a not in self.possible_actions(s):
            raise ValueError("Illegal action")

        n1m, n2m = self.after_move(s, a)
        p_next_1, e_rent1 = self._loc_outcomes(1, n1m)
        p_next_2, e_rent2 = self._loc_outcomes(2, n2m)

        exp_revenue = (e_rent1 + e_rent2) * self.params.revenue_per_rental
        return p_next_1, p_next_2, exp_revenue
