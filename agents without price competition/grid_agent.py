import random
import pickle
import pandas as pd
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
        self.remaining_inventory = params['inventory_limit']
        self.inventory_replenish = params['inventory_replenish']

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
        self.this_agent_number = agent_number
        
        # --- 1. Model and Feature Setup ---
        # Load the BASELINE model for the first test run
        MODEL_PATH = 'agents/team-4/xgb_model.pkl' 
    
        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
            # FEATURES FOR MODEL A (Baseline: 4 raw features)
        self.model_features = ['Covariate1', 'Covariate2', 'Covariate3', 'price_item'] 
            
        # --- 2. Define the Smart Price Range (Fixed for testing) ---
        dense_range = np.linspace(0, 80, 160)
        sparse_range = np.linspace(81, 387, 40)
        self.prices_to_check = np.concatenate([dense_range, sparse_range])
        
        # --- 3. Define the Threshold Strategy (FIXED for model testing) ---
        # These numbers must REMAIN CONSTANT during the entire model bake-off (A-F).
        self.LOW_INVENTORY_THRESHOLD = 90.0  
        self.HIGH_INVENTORY_THRESHOLD = 40.0
        self.LOW_STOCK_LEVEL = 5          
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
        pass

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
        self.remaining_inventory = inventories[self.this_agent_number]

        # --- A. Check Stock and Model Status ---
        if self.remaining_inventory <= 0:
            return self.NO_SALE_PRICE
        if self.model is None:
            return 112.358 # Fallback

        # --- B. Find Customer's Max Value (Mini Grid Search) ---
        max_revenue = -1
        best_price = self.NO_SALE_PRICE
        
        # Prepare base features
        base_features = [
            new_buyer_covariates[0],
            new_buyer_covariates[1],
            new_buyer_covariates[2]
        ]
        
        for price in self.prices_to_check:
            
            input_features = [base_features + [price]]
            
            # Create the input DataFrame (This is crucial for prediction)
            X_predict = pd.DataFrame(input_features, columns=self.model_features)

            # Predict probability and calculate revenue
            purchase_prob = self.model.predict_proba(X_predict)[0][1]
            revenue = price * purchase_prob
            
            if revenue > max_revenue:
                max_revenue = revenue
                best_price = price

        current_threshold = self.HIGH_INVENTORY_THRESHOLD # Default to the low threshold

        T_REMAIN = time_until_replenish
        I_REMAIN = self.remaining_inventory
        # Case 1: Must Sell (Unsold item will be discarded, so sell for anything)
        if T_REMAIN <= 1 and I_REMAIN >= 1:
            current_threshold = 0.01 
        
        # Case 2: High Scarcity (Low Inventory AND High Time Remaining)
        # This is the most defensive state. Inventory is low, but we have many chances for a better customer.
        elif I_REMAIN <= self.LOW_STOCK_LEVEL and T_REMAIN >= 10:
            current_threshold = self.LOW_INVENTORY_THRESHOLD
            
        # Case 3: Normal/Low Scarcity (All other cases)
        else:
            current_threshold = self.HIGH_INVENTORY_THRESHOLD

        # --- E. Final Decision ---
        if max_revenue >= current_threshold:
            # Immediate Reward >= Opportunity Cost. SELL.
            return best_price
        else:
            # Immediate Reward < Opportunity Cost. REJECT (save product).
            return self.NO_SALE_PRICE

