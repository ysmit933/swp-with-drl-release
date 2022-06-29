from predefined_models import CaseStudyTarget
from utils import train_model, load_model, test_model


show = True
constrained_firing = False
model_name = 'model-name'

env = CaseStudyTarget
env_kwargs = dict(
    random_start_percentage=0.5,
)

policy_kwargs = dict(
    net_arch=[dict(vf=[128, 128], pi=[128, 128])]
)

agent_kwargs = dict(
    learning_rate = 1e-5,
    n_steps = 256,
    batch_size = 32,
    n_epochs = 9,
    gamma = 0.95,
    gae_lambda = 0.9,
    clip_range = 0.2,
    ent_coef = 0.01,
    vf_coef = 0.5,
    policy_kwargs = policy_kwargs,
    verbose = 1,
)

model = train_model(
    env, 
    time_steps=1e4, 
    agent_kwargs=agent_kwargs, 
    env_kwargs=env_kwargs, 
    model_name=model_name, 
    constrain_firing=constrained_firing, 
    show=show, 
)

if show:
    print('Trained model {}'.format(model_name))
    model = load_model(model_name)
    print(test_model(env, model, show=True))
    