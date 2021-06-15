from create_milp_dict import create_problem_dict
from uc_model import solve_milp, solution_to_schedule
from rl4uc.rl4uc.environment import make_env
import numpy as np
import pandas as pd


def test_kazarlis():
    """Kazarlis 1996 problem"""

    demand = np.array([700, 750, 850, 950, 1000, 1100,
                       1150, 1200, 1300, 1400, 1450, 1500, 1400,
                       1300, 1200, 1050, 1000, 1100, 1200,
                       1400, 1300, 1100, 900, 800])
    wind = np.array([0]*24)
    env_params = {'num_gen': 10, 'dispatch_freq_mins': 60}
    problem_dict = create_problem_dict(demand,
                                       wind,
                                       env_params,
                                       reserve_pct=10)
    solution = solve_milp(problem_dict)

    print("${:.2f}".format(solution.obj()))

#    assert (solution.obj() < 570000) and (solution.obj() > 550000)

    schedule = solution_to_schedule(solution, problem_dict)
    df = pd.DataFrame({'demand': demand, 'wind': wind})
    env = make_env(mode='test', profiles_df=df, **env_params)
    env.reset()
    cost = 0
    for a in schedule:
        print(a)
        o, r, d = env.step(a, deterministic=True)
        cost -= r
    print(cost)
        
    


if __name__ == '__main__':
    test_kazarlis()

    
