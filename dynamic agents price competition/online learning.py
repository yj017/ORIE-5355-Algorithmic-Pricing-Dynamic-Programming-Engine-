import random
import pickle
import os
import numpy as np


'''
This template serves as a starting point for your agent.
'''


class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        
        self.project_part = params['project_part'] 

        ### starting remaining inventory and inventory replenish rate are provided
        ## every time the inventory is replenished, it is set to the inventory limit
        ## the inventory_replenish rate is how often the inventory is replenished
        ## for example, we will run with inventory_replenish = 20, with the limit of 11. Then, the inventory will be replenished every 20 time steps (time steps 0, 20, 40, ...) and the inventory will be set to 11 at those time steps. 
        inv_param = params.get('inventory_limit', 11)
        
        if isinstance(inv_param, dict):
            self.inventory_limit = inv_param.get('max', 20)
        else:
            self.inventory_limit = inv_param
            
        self.remaining_inventory = self.inventory_limit
        self.inventory_replenish = params.get('inventory_replenish', 20)

        ### useful if you want to use a more complex price prediction model
        ### note that you will need to change the name of the path and this agent file when submitting
        ### complications: pickle works with any machine learning models defined in sklearn, xgboost, etc.
        ### however, this does not work with custom defined classes, due to the way pickle serializes objects
        ### refer to './yourteamname/create_model.ipynb' for a quick tutorial on how to use pickle
        # self.filename = './[yourteamname]/trained_model'
        # self.trained_model = pickle.load(open(self.filename, 'rb'))

        ### potentially useful for Part 2 -- When competition is between two agents
        ### and you want to keep track of the opponent's status
        # self.opponent_number = 1 - agent_number  # index for opponent
        self.opponent_number = 1 - agent_number

        POLICY_PATH = 'agents/team-4/dp_policy_table.pkl'
        DISCRETIZER_PATH = 'agents/team-4/customer_discretizer.pkl'

        with open(POLICY_PATH, 'rb') as f:
            self.PolicyTable = pickle.load(f)

        with open(DISCRETIZER_PATH, 'rb') as f:
            self.discretizer = pickle.load(f)
        
        self.opponent_ratios = []
        self.avg_opponent_ratio = 1.0
        self.last_monopoly_price = 50.0
        
        # Track history to calculate "Dumb" prediction
        self.last_opponent_raw_price = 50.0
        
        # Track ERRORS to decide which model is better
        # We start with a bias towards "Smart" to be safe
        self.error_smart = 10.0 
        self.error_dumb = 20.0

        self.NO_SALE_PRICE = 999.0

    def _process_last_sale(
            self, 
            last_sale,
            state,
            inventories,
            time_until_replenish
        ):
        '''
        This function updates your internal state based on the last sale that occurred.
        This template shows you several ways you can keep track of important metrics.
        '''
        ### keep track of who, if anyone, the customer bought from
        did_customer_buy_from_me = (last_sale[0] == self.this_agent_number)
        ### potentially useful for Part 2
        # did_customer_buy_from_opponent = (last_sale[0] == self.opponent_number)

        ### keep track of the prices that were offered in the last sale
        my_last_prices = last_sale[1][self.this_agent_number]
        ### potentially useful for Part 2
        # opponent_last_prices = last_sale[1][self.opponent_number]

        ### keep track of the profit for this agent after the last sale
        my_current_profit = state[self.this_agent_number]
        ### potentially useful for Part 2
        # opponent_current_profit = state[self.opponent_number]

        ### keep track of the inventory levels after the last sale
        self.remaining_inventory = inventories[self.this_agent_number]
        ### potentially useful for Part 2
        # opponent_inventory = inventories[self.opponent_number]

        ### keep track of the time until the next replenishment
        time_until_replenish = time_until_replenish

        ### TODO - add your code here to potentially update your pricing strategy 
        ### based on what happened in the last round
        opponent_price = last_sale[1][self.opponent_number]
                
        if opponent_price > 0:
   
            pred_smart_was = self.last_monopoly_price * self.avg_opponent_ratio
                    
            last_winner = last_sale[0]
            if last_winner == self.opponent_number:
                pred_dumb_was = self.last_opponent_raw_price * 1.1
            else:
                pred_dumb_was = self.last_opponent_raw_price * 0.9
                    
                current_err_smart = abs(pred_smart_was - opponent_price)
                current_err_dumb = abs(pred_dumb_was - opponent_price)
                    
                self.error_smart = 0.8 * self.error_smart + 0.2 * current_err_smart
                self.error_dumb = 0.8 * self.error_dumb + 0.2 * current_err_dumb

                self.last_opponent_raw_price = opponent_price

                if self.last_monopoly_price > 1.0:
                    ratio = opponent_price / self.last_monopoly_price
                    ratio = np.clip(ratio, 0.6, 1.4)
                    self.opponent_ratios.append(ratio)

                    if len(self.opponent_ratios) > 10:
                        self.opponent_ratios.pop(0)
                    self.avg_opponent_ratio = np.percentile(self.opponent_ratios, 40)

    def action(self, obs):
        '''
        This function is called every time the agent needs to choose an action by the environment.

        The input 'obs' is a 5 tuple, containing the following information:
        -- new_buyer_covariates: a vector of length 3, containing the covariates of the new buyer.
        -- last_sale: a tuple of length 2. The first element is the index of the agent that made the last sale, if it is NaN, then the customer did not make a purchase. The second element is a numpy array of length n_agents, containing the prices that were offered by each agent in the last sale.
        -- state: a vector of length n_agents, containing the current profit of each agent.
        -- inventories: a vector of length n_agents, containing the current inventory level of each agent.
        -- time_until_replenish: an integer indicating the time until the next replenishment, by which time your (and your opponent's, in part 2) remaining inventory will be reset to the inventory limit.

        The expected output is a single number, indicating the price that you would post for the new buyer.
        '''

        new_buyer_covariates, last_sale, state, inventories, time_until_replenish = obs
        self._process_last_sale(last_sale, state, inventories, time_until_replenish)

        ### currently output is just a deterministic price for the item
        ### but you are expected to use the new_buyer_covariates
        ### combined with models you come up with using the training data 
        ### and history of prices from each team to set a better price for the item
        current_inventory_i = inventories[self.this_agent_number]
        time_remaining_t = time_until_replenish

        if current_inventory_i <= 0:
            return self.NO_SALE_PRICE
        
        # Map Customer to Type (k)
        # Predict the cluster index using the loaded K-Means model
        # Input must be 2D array [[cov1, cov2, cov3]]
        customer_type_k = self.discretizer.predict([new_buyer_covariates.tolist()])[0]
        
        # DP Policy Lookup (Monopoly Price)
        # Look up the optimal price from the pre-calculated Bellman table
        # This price P* maximizes expected revenue given the inventory constraint.
        monopoly_price = self.PolicyTable[time_remaining_t, current_inventory_i, customer_type_k]
        self.last_monopoly_price = monopoly_price

        pred_smart = monopoly_price * self.avg_opponent_ratio
        
        # Calculate Dumb Prediction (History Based)
        last_winner = last_sale[0]
        if last_winner == self.opponent_number:
            pred_dumb = self.last_opponent_raw_price * 1.1
        else:
            pred_dumb = self.last_opponent_raw_price * 0.9
            
        # Choose the Best Model
        # If the Dumb model has lower error, trust it. Otherwise, trust Smart.
        if self.error_dumb < self.error_smart:
            # Target = Dumb Prediction (but capped at monopoly price so we don't overbid)
            target_bid = min(pred_dumb, monopoly_price) - 0.10
        else:
            # We are playing against a Smart Agent
            # Target = Smart Prediction
            target_bid = pred_smart - 0.10
        
        target_inventory = (time_remaining_t / 20.0) * self.inventory_limit
        
        if current_inventory_i > target_inventory:
            # BEHIND SCHEDULE (Too much stock)
            # We are losing the volume war. We must sell.
            # We will undercut the dummy no matter how low they go.
            floor_price = 5
            
        elif current_inventory_i < 3:
            # LAST UNIT DEFENSE
            # We are ahead of schedule, with only 1 unit.
            # Save it for a massive whale. High floor.
            floor_price = monopoly_price * 0.90
            
        else:
            # ON SCHEDULE
            # We are comfortable. Be smart but not desperate.
            # Sell for 70% of value.
            floor_price = monopoly_price * 0.70

        if target_bid >= floor_price:
            return target_bid
        else:
            return self.NO_SALE_PRICE