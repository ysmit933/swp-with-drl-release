from predefined_models import *
from utils import test_model, load_model
from heuristic_model import Heuristic


test_name = 'test-name'
samplesize = 10
time_horizon = 60

# List of environments to use in evaluate
envs = [
    TestSettingTarget, 
    TestSettingCombined, 
    TestSettingTarget, 
    TestSettingCombined
]
# List of model names to run on the environments, or specify heuristic 
models = [
    'test_setting_target', 
    'test_setting_combined', 
    'heuristic', 
    'heuristic'
]
# List of labels for results
mappings = [
    'test_setting_target_ppo', 
    'test_setting_combined_ppo', 
    'test_setting_target_heuristic', 
    'test_setting_combined_heuristic'
]

results = {
    x: {} for x in mappings
}
for env, model_name, mapping in zip(envs, models, mappings):
    print(f"Running {model_name}")

    env_kwargs = dict(
        random_start_percentage = 0.0,
        random_start_around_fixed = True,
        time_horizon=time_horizon,
        early_termination=False,
    )
    test_env = env(**env_kwargs)

    model = Heuristic(test_env) if model_name == 'heuristic' else load_model(model_name)

    results[mapping]["rew"] = []
    results[mapping]["sal"] = []
    results[mapping]["dev"] = []
    results[mapping]["soc"] = []
    results[mapping]["oob"] = []
    results[mapping]["ill"] = []
    results[mapping]["sem"] = []
    results[mapping]["tot"] = []
    results[mapping]["cos"] = []
    
    for _ in range(samplesize):
        result = test_model(test_env, model, show=False, random_start=False)
        results[mapping]["rew"].append(result['average_reward'])
        results[mapping]["sal"].append(sum(result['salary_costs']))
        results[mapping]["dev"].append(np.mean(result['deviations']))
        results[mapping]["soc"].append(result['soc_deviations'])
        results[mapping]["oob"].append(result['out_of_bounds'])
        results[mapping]["ill"].append(sum(result['illegal_actions']))
        results[mapping]["sem"].append(sum(result['semi_illegal_actions']))
        results[mapping]["tot"].append(sum(result['total_hires']) - sum(result['total_fires']))
        results[mapping]["cos"].append(sum(result['total_costs']))

np.save(f'./test-results/{test_name}.npy', results)
