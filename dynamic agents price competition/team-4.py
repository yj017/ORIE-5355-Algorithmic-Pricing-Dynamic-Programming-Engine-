import random
import pickle
import os
import numpy as np


class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number
        self.project_part = params['project_part']

        # Inventory parameters
        inv_param = params.get('inventory_limit', 12)
        if isinstance(inv_param, dict):
            self.inventory_limit = inv_param.get('max', 20)
        else:
            self.inventory_limit = inv_param

        self.remaining_inventory = self.inventory_limit
        self.inventory_replenish = params.get('inventory_replenish', 20)

        # Opponent index
        self.opponent_number = 1 - agent_number

        # Load DP-derived optimal price table and customer discretizer
        POLICY_PATH = 'agents/team-4/dp_policy_table.pkl'
        DISCRETIZER_PATH = 'agents/team-4/customer_discretizer.pkl'

        with open(POLICY_PATH, 'rb') as f:
            self.PolicyTable = pickle.load(f)

        with open(DISCRETIZER_PATH, 'rb') as f:
            self.discretizer = pickle.load(f)

        # Buffer for recent win/loss observations used for win-rate tracking
        self.last_k_outcomes = []      # 1 if I won the sale, 0 otherwise
        self.winrate_window = 15       # sliding window size

        self.NO_SALE_PRICE = 99999.0

    def _update_winrate(self, last_sale):
        """
        Update recent performance history based on the previous customer:
        - 1 = sale went to this agent
        - 0 = sale went to opponent or no purchase occurred
        """
        last_winner = last_sale[0]

        if np.isnan(last_winner):
            outcome = 0
        else:
            outcome = 1 if (int(last_winner) == self.this_agent_number) else 0

        self.last_k_outcomes.append(outcome)
        if len(self.last_k_outcomes) > self.winrate_window:
            self.last_k_outcomes.pop(0)

    def _get_competition_factor(self):
        """
        Generate a competition-response multiplier based on recent win-rate:

        Low win-rate   → price discount to increase competitiveness  
        Neutral state  → no adjustment  
        High win-rate  → price uplift to extract surplus when dominating
        """
        if len(self.last_k_outcomes) == 0:
            return 1.0

        win_rate = sum(self.last_k_outcomes) / len(self.last_k_outcomes)

        if win_rate < 0.30:
            return 0.90    # losing → slightly lower price
        elif win_rate > 0.70:
            return 1.10    # dominating → slightly higher price
        else:
            return 1.00    # balanced → no change

    def _get_inventory_factor(self, current_inventory, time_until_replenish):
        """
        Compute an inventory-adjustment multiplier using deviation from
        a linear benchmark consumption trajectory.

        If sales lag behind → discount prices  
        If demand is strong / inventory tight → increase prices
        """
        B = self.inventory_replenish
        C = self.inventory_limit
        tau = time_until_replenish

        # Linear target inventory path: higher early in the cycle, lower near replenishment
        target_inv = C * (tau / max(B, 1))

        # Directional deviation
        gap = current_inventory - target_inv
        beta = 0.8  # sensitivity

        if gap > 0:
            # too much inventory left → discount
            factor = 1.0 - beta * (gap / C)
        else:
            # inventory is tight → increase price
            factor = 1.0 - beta * (gap / C)  # gap<=0 makes this + adjustment

        return max(0.8, min(1.25, factor))

    def _process_last_sale(self, last_sale, state, inventories, time_until_replenish):
        """
        Internal state updates after each transaction:
        - refresh inventory level
        - update rolling win-rate
        """
        self.remaining_inventory = inventories[self.this_agent_number]

        if last_sale is not None:
            self._update_winrate(last_sale)

    def action(self, obs):
        """
        Core pricing decision function.
        """
        new_buyer_covariates, last_sale, state, inventories, time_until_replenish = obs

        # incorporate previous sale feedback
        self._process_last_sale(last_sale, state, inventories, time_until_replenish)

        current_inventory = inventories[self.this_agent_number]

        # If we are out of stock, quote a prohibitively high price (no sale)
        if current_inventory <= 0:
            return self.NO_SALE_PRICE

        # 1) Map customer attributes to discrete customer type
        customer_type_k = self.discretizer.predict(
            [new_buyer_covariates.tolist()]
        )[0]

        # 2) Extract baseline monopoly price from dynamic-programming table
        monopoly_price = self.PolicyTable[time_until_replenish,
                                          current_inventory,
                                          customer_type_k]

        # 3) Inventory-based price adjustment
        inv_factor = self._get_inventory_factor(current_inventory,
                                                time_until_replenish)

        # 4) Win-rate responsiveness (competition adaptive multiplier)
        comp_factor = self._get_competition_factor()

        # 5) Composite price before exploration
        base_price = monopoly_price * inv_factor * comp_factor

        # 6) Randomized exploration factor for stochastic learning
        explore_factor = random.uniform(0.97, 1.03)
        final_price = base_price * explore_factor

        # 7) Enforce reasonable price bounds
        final_price = max(5.0, min(200.0, final_price))

        return final_price
