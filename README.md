# Strategic Workforce Planning with Deep Reinforcement Learning

This repository contains the code and data for the paper `Strategic Workforce Planning with Deep Reinforcement Learning` presented at the [LOD
22](https://lod2022.icas.cc) conference. If you use this code, please cite:

```
@inproceedings{smit2022,
  title={Strategic Workforce Planning with Deep Reinforcement Learning},
  author={Smit, Yannick and den Hengst, Floris and Bhulai, Sandjai and Mehdad, Ehsan},
  booktitle={International Conference on Machine Learning, Optimization, and Data Science},
  year={2022},
  organization={Springer}
}
```

In the paper, an approach is presented to optimize management decisions about
an organizations workforce using a cohort model. If a target headcount can be
formulated up-front, an analytical optimum can be determined with linear
programming (LP). However, such an _operational_ target is hard to formulate in
practice, in which case high-level _strategic_ objectives are to be used
instead.

We argue that DRL is preferred over the established LP approach for strategic
objectives.
We show that this is the case if these objectives are nonlinear in the workforce composition and if
the system is stochastic.

In practice, we expect both to nonlinearity and stochasticity in realistic SWP
problems.

## Installation

Install the requirements in `requirements.txt` using e.g. conda:

```
conda create --name swp-with-drl --file requirements.txt
```

## Optimizing your own workforce

In order to run the code on your own workforce, two types of input are required:

1. a custom Markov chain, which can be inferred from data.
   See `example_transition_probability_matrix.xlsx` for an example of the
   Markov chain format in this code.
2. Organisation parameters such as salary costs, hiring costs, goal workforce etc.

### Custom Markov Chain

The Excel file containing transition probabilities expects an _n+1 x n+1_ table,
where _n_ is the number of cohorts. The entry at row _i_ and column _j_ contains
the probability of movement from cohort _i_ to cohort _j_. The first row and the
first column contain the labels of the cohorts.

The transition probabilities can be estimated from corporate data as described in
Section 3.2 the aforementioned paper.

Finally, set the path to the `.xlsx` file with these probabilities in the
class for your experiment in `predefined_models.py`.

### Organisation parameters

A number of parameters that change per organisation can be set in
`predefined_models.py`. These parameters are set per cohort. If more
fine-grained information is available, you can choose to either change the
problem structure by introducing new cohorts or to aggregate the costs to a
cohort-level by a weighted average.

## Usage

Select an environment model and train a DRL agent:

```
from predefined_models import CaseStudyTarget
from utils import train_model


# Set environmental parameters (optional)
env_kwargs = dict(
    random_start_percentage=0.5,
)

# Set PPO parameters (optional)
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
    verbose = 1,
)

# Train DRL agent
model = train_model(
    CaseStudyTarget,  # Set environment model here
    time_steps=1e4,  # Number of training steps
    agent_kwargs=agent_kwargs,
    env_kwargs=env_kwargs,
    model_name='example-model-name',  # Set a custom name for the model
    constrain_firing=False,  # Set to True for constrained firing models
    show=True,
)
```

Test the trained model:

```
from utils import load_model, test_model

model = load_model(model_name)
test_model(env, model, show=True)
```
