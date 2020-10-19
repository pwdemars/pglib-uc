#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:18:16 2020

@author: patrickdemars
"""

import numpy as np 
import os
import json

from rl4uc.rl4uc.environment import make_env

RESERVE_MARGIN = 0.1

def calculate_piecewise_production(gen_info, idx, n_hrs, N=4):
    mws = np.linspace(gen_info.min_output[idx], gen_info.max_output[idx], N)
    costs = n_hrs*(gen_info.a[idx]*(mws**2) + gen_info.b[idx]*mws + gen_info.c[idx])
    pairs = []
    for mw, cost in zip(mws, costs):
        pairs.append({"mw": float(mw), "cost": float(cost)})
    return pairs

def create_problem_dict(demand, wind=None, **params):
    """
    Create a dictionary defining the problem for input to the pglib model.

    Args:
        - demand (array): demand profile
        - params (dict): parameters file, gives number of generators, dispatch frequency.
    """
    if wind is not None:
        net_demand = demand - wind
    env = make_env(**params)
    gen_info = env.gen_info
#    reserves = [a*RESERVE_MARGIN for a in demand]
    reserves = [sum(gen_info.max_output)*0.1]*len(net_demand) # Reserve margin is a fixed percentage of the total capacity
    dispatch_freq = params.get('dispatch_freq_mins')/60
    num_periods = len(net_demand)

    all_gens = {}
    for g in range(params.get('num_gen')):
        GEN_NAME = 'GEN'+str(g)
        foo = {"must_run": 0,
               "power_output_minimum": float(gen_info.min_output[g]), 
               "power_output_maximum": float(gen_info.max_output[g]),
               "ramp_up_limit": 10000., 
               "ramp_down_limit": 10000., 
               "ramp_startup_limit": 10000., 
               "ramp_shutdown_limit": 10000., 
               "time_up_minimum": int(gen_info.t_min_up[g]), 
               "time_down_minimum": int(gen_info.t_min_down[g]),
               "power_output_t0": 0.0,
               "unit_on_t0": int(1 if gen_info.status[g] > 0 else 0), 
               "time_up_t0": int(gen_info.status[g] if gen_info.status[g] > 0 else 0), 
               "time_down_t0": int(abs(gen_info.status[g]) if gen_info.status[g] < 0 else 0),
               "startup": [{"lag": 1, "cost": float(gen_info.hot_cost[g])}],
               "piecewise_production": calculate_piecewise_production(gen_info, g, dispatch_freq),
               "name": GEN_NAME}
        all_gens[GEN_NAME] = foo
        
    all_json = {"time_periods":num_periods,
                "demand":list(net_demand),
                "reserves":reserves,
                "thermal_generators":all_gens,
                "renewable_generators": {}}
    
    return all_json


