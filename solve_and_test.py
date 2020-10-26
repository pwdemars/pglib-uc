#!/usr/bin/env python

import numpy as np
import pandas as pd
import argparse
import json
import os 
import time

from rl4uc.rl4uc.environment import make_env
from create_milp_dict import create_problem_dict
from uc_model import solve_milp, solution_to_schedule

SEED=2

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='MILP solutions to UC problem (and testing with stochastic environment')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--params_fn', type=str, required=True,
                        help='Filename for parameters for the environment')
    parser.add_argument('--test_data_dir', type=str, required=True,
                        help='Directory containing batch of .txt demand profiles to solve')
    parser.add_argument('--num_samples', type=int, required=False, default=100,
                        help='Number of demand realisation to compute costs for')

    args = parser.parse_args()
    params = json.load(open(args.params_fn))

    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Save params file to save_dir 
    with open(os.path.join(args.save_dir, 'params.json'), 'w') as fp:
        json.dump(params, fp)

    # get list of test profiles
    test_profiles = [f for f in os.listdir(args.test_data_dir) if '.csv' in f]
    test_profiles.sort()

    all_test_costs = {}
    all_times = []

    for f in test_profiles:
        
        prof_name = f.split('.')[0]

        # Formulate the problem dictionary (with wind)
        df = pd.read_csv(os.path.join(args.test_data_dir, f))
        demand = df.demand.values
        wind = df.wind.values
        problem_dict = create_problem_dict(demand, wind, **params)

        # Solve the MILP
        s = time.time()
        solution = solve_milp(problem_dict)
        time_taken = time.time()-s
        all_times.append(time_taken)

        # convert solution to binary schedule
        schedule = solution_to_schedule(solution, problem_dict)
        
        # Save the binary schedule as a .csv file
        columns = ['schedule_' + str(i) for i in range(params.get('num_gen'))]
        df = pd.DataFrame(schedule, columns=columns)
        df.to_csv(os.path.join(args.save_dir, '{}_solution.csv'.format(prof_name)), index=False)

        # initialise environment for sample operating costs
        env = make_env(mode='test',demand=demand, wind=wind, **params)

        test_costs = []
        print("Testing schedule...")
        np.random.seed(SEED)
        for i in range(args.num_samples):
            env.reset()
            total_reward = 0 
            for action in schedule:
                action = np.where(np.array(action)>0, 1, 0)
                obs,reward,done = env.step(action)
                total_reward += reward
            test_costs.append(-total_reward)
        all_test_costs[prof_name] = test_costs
        print("Done")
        print()

    np.savetxt(os.path.join(args.save_dir, 'time_taken.txt'), np.array(all_times), fmt='%1.2f')

    all_test_costs = pd.DataFrame(all_test_costs)
    all_test_costs.to_csv(os.path.join(args.save_dir, 'sample_costs.csv'), index=False)

