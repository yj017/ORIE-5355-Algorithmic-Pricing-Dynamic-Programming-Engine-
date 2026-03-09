import pickle
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans 

class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number
        
        self.project_part = params.get('project_part', None)

        # Extracts Imax for DP table sizing from the dict parameter
        inventory_param = params.get('inventory_limit', 11)
        
        if isinstance(inventory_param, dict):
            self.inventory_limit_max = inventory_param.get('max', 20) 
            self.inventory_limit_dict = inventory_param 
        else:
            self.inventory_limit_max = inventory_param
            self.inventory_limit_dict = {'min': inventory_param, 'max': inventory_param}
        
        # Store the cycle length (T_max)
        self.inventory_replenish = params.get('inventory_replenish', 20)
        self.T_max = max(1, self.inventory_replenish)
        
        self.remaining_inventory = self.inventory_limit_max 

        # Load Pre-Calculated DP Policy Table
        # The agent loads the results of the offline Bellman Solver.
        POLICY_PATH = 'agents/team-4/dp_policy_table.pkl' 
        DISCRETIZER_PATH = 'agents/team-4/customer_discretizer.pkl'
        
        
            # Policy Table: PI[t][i][k]
        with open(POLICY_PATH, 'rb') as f:
                self.PolicyTable = pickle.load(f)
            
            # Customer Mapping Function (KMeans object)
        with open(DISCRETIZER_PATH, 'rb') as f:
                self.discretizer = pickle.load(f)
            
            # Initialize the Value Function Table (V) size (for internal reference)
        self.V = np.zeros((self.T_max + 1, self.inventory_limit_max + 1), dtype=float) 

            
        self.NO_SALE_PRICE = 999.0
        
    def _process_last_sale(
            self, 
            last_sale,
            state,
            inventories,
            time_until_replenish
        ):
        # Update internal state for tracking purposes
        self.remaining_inventory = inventories[self.this_agent_number]
        self.time_until_replenish = time_until_replenish
        pass

    def action(self, obs):
        new_buyer_covariates, last_sale, state, inventories, time_until_replenish = obs
        
        # Define State Variables (t and i)
        current_inventory_i = inventories[self.this_agent_number]
        time_remaining_t = time_until_replenish 
        
        # Check Feasibility
        if current_inventory_i <= 0:
            return self.NO_SALE_PRICE

        # Determine Customer Type (k)
        # Map the incoming continuous covariates (c) to a discrete type (k)
        # This replaces the slow grid search calculation.
        X_cluster_input = pd.DataFrame(
            [new_buyer_covariates], 
            columns=['Covariate1', 'Covariate2', 'Covariate3']
        )
        
        # Now predict using the DataFrame
        customer_type_k = self.discretizer.predict(X_cluster_input)[0]
        
        # Policy Lookup (The Final DP Decision)
        # The optimal price is retrieved directly from the pre-calculated table: PI[t][i][k]

        optimal_price = self.PolicyTable[time_remaining_t, current_inventory_i, customer_type_k]
        return optimal_price
 