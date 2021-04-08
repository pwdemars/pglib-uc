#!/usr/bin/env python

import numpy as np
import pandas as pd
import argparse
import json
import os 
import time

from rl4uc.rl4uc.environment import make_env
from helpers import test_schedule, save_results, get_scenarios
from create_milp_dict import create_problem_dict
from uc_model import solve_milp, solution_to_schedule

SEED=999

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='MILP solutions to UC problem (and testing with stochastic environment')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--env_params_fn', type=str, required=True,
                        help='Filename for environment parameters.')
    parser.add_argument('--test_data_dir', type=str, required=True,
                        help='Directory containing batch of .txt demand profiles to solve')
    parser.add_argument('--num_samples', type=int, required=False, default=1000,
                        help='Number of demand realisation to compute costs for')
    parser.add_argument('--reserve_pct', type=int, required=False, default=None,
                        help='Reserve margin as percent of forecast net demand')
    parser.add_argument('--reserve_sigma', type=int, required=False, default=None,
                        help='Number of sigma to consider for reserve constraint')
    parser.add_argument('--perfect_forecast', type=bool, required=False, default=False,
                        help='Boolean for perfect forecast')

    args = parser.parse_args()

    # If using perfect forecast, set reserve margin to 0
    if args.perfect_forecast: args.reserve_pct=0

    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Update params
    params = vars(args)

    res = 'perfect' if args.perfect_forecast else args.reserve_pct
    params.update({'milp': 'true',
                   'reserve': res})
    # Read the parameters
    env_params = json.load(open(args.env_params_fn))

    # Save params file to save_dir 
    with open(os.path.join(args.save_dir, 'params.json'), 'w') as fp:
        fp.write(json.dumps(params, sort_keys=True, indent=4))

    # Save env params to save_dir
    with open(os.path.join(args.save_dir, 'env_params.json'), 'w') as fp:
        fp.write(json.dumps(env_params, sort_keys=True, indent=4))

    # If using sigma for reserve constraint, determine reserve constraint here:
    if args.reserve_sigma is not None:
        np.random.seed(SEED)
        env = make_env(mode='train', **env_params)
        scenarios = get_scenarios(env, 1000)
        sigma = np.std(scenarios)
        reserve_mw = args.reserve_sigma * sigma
    else:
        reserve_mw = None

    # get list of test profiles
    test_profiles = [f for f in os.listdir(args.test_data_dir) if '.csv' in f]
    test_profiles.sort()

    all_test_costs = {}
    all_times = []

    for f in test_profiles:
        
        prof_name = f.split('.')[0]
        print(prof_name)

        # Formulate the problem dictionary (with wind)
        profile_df = pd.read_csv(os.path.join(args.test_data_dir, f))
        demand = profile_df.demand.values
        wind = profile_df.wind.values
        problem_dict = create_problem_dict(demand, wind, env_params=env_params, reserve_pct=args.reserve_pct, reserve_mw=reserve_mw)
        

        fn = prof_name + '.json'
        with open(os.path.join(args.save_dir, fn), 'w') as fp:
            json.dump(problem_dict, fp)

        # Solve the MILP
        s = time.time()
        solution = solve_milp(problem_dict)
        time_taken = time.time()-s
        all_times.append(time_taken)

        # convert solution to binary schedule
        schedule = solution_to_schedule(solution, problem_dict)
        
        # Save the binary schedule as a .csv file
        columns = ['schedule_' + str(i) for i in range(env_params.get('num_gen'))]
        df = pd.DataFrame(schedule, columns=columns)
        df.to_csv(os.path.join(args.save_dir, '{}_solution.csv'.format(prof_name)), index=False)

        # initialise environment for sample operating costs
        env = make_env(mode='test', profiles_df=profile_df, **env_params)

        TEST_SAMPLE_SEED=999
        test_costs, lost_loads = test_schedule(env, schedule, TEST_SAMPLE_SEED, args.num_samples, args.perfect_forecast)
        save_results(prof_name, args.save_dir, env.num_gen, schedule, test_costs, lost_loads, time_taken)

        print("Done")
        print()
        print("Mean costs: ${:.2f}".format(np.mean(test_costs)))
        print("Lost load prob: {:.3f}%".format(100*np.sum(lost_loads)/(args.num_samples * env.episode_length)))
        print("Time taken: {:.2f}s".format(time_taken))
        print() 

