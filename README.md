# Compass Under Constraint

```yaml
STATUS  : ACTIVE
AUTHOR  : Bradley D. Saucier
COURSE  : SNHU CS-370 - Current and Emerging Trends in Computer Science
```

> [!IMPORTANT]
> **BOTTOM LINE UP FRONT**
>
> A reinforcement learning project about teaching a pirate to stop choosing bad turns and find the treasure in an 8 by 8 maze.
>
> Python 3.11 | TensorFlow 2.16 | Keras 3 | NumPy | Jupyter

[![CI](https://github.com/bradsaucier/compass-under-constraint/actions/workflows/ci.yml/badge.svg)](https://github.com/bradsaucier/compass-under-constraint/actions/workflows/ci.yml)

## What the project does

The pirate sees the maze as a flattened vector of 64 values. Its neural network produces one Q-value for each possible move: left, up, right, or down. The network has two 64-unit hidden layers with PReLU activations and a four-value output layer.

Because the maze is small and does not change, the final notebook records all four actions from each of 50 legal starting cells. Those 200 transitions are used to calculate Bellman targets, and the network learns to approximate them. The course project describes this as deep Q-learning. More specifically, this implementation uses fitted Q updates over a complete one-step transition table and then trains a neural network on the resulting values.

The main artifact is [Saucier_Bradley_ProjectTwo.ipynb](Saucier_Bradley_ProjectTwo.ipynb). It includes the implementation and the saved output from my submitted run.

## Results from the submitted run

| Measure | Result |
| --- | ---: |
| Training seed | 31 |
| Warm-up transitions | 200 |
| Stopping epoch | 1200 |
| Final reported loss | 0.0002629923 |
| Greedy evaluation | 32 wins in 32 starts |
| Completion check | Passed from every legal start |
| Top-left start | Passed |
| Training time | 99.1 seconds |

These numbers came from one saved run. They do not promise that another computer or library build will stop at epoch 1200. The portfolio copy now sets the seed before the model is created, which makes future runs easier to reproduce, but small numerical differences can still change the path to convergence.

## What I was given and what I wrote

Southern New Hampshire University provided the notebook scaffold, the fixed maze, the TreasureMaze environment, and the GameExperience replay class. The supplied material already handled movement, rewards, terminal states, experience storage, visualization, and the basic 64-64-4 network shape.

I completed the qtrain function and connected the transition collection, Bellman targets, neural-network updates, and evaluation checks. I also adjusted the evaluation helpers and the model settings used in the final run.

The first version of my confidence came from watching the loss fall. The pirate did not share that confidence and could still choose the wrong turn. That was the point where I stopped treating loss as the answer and started treating it as one diagnostic among several.

## Reflection

### What do computer scientists do, and why does it matter?

Computer scientists take a problem that makes sense to a person and turn it into something a computer can represent and test. A person looks at this maze and sees walls, routes, and dead ends. The network gets 64 numbers and no built-in understanding that two squares are next to each other.

The choices made during that translation matter. A reward, test, or assumption can be repeated thousands of times once it becomes software. Getting the code to run is only part of the job. We also need to know what it is doing and where its answer stops being trustworthy.

### How do I approach a problem as a computer scientist?

I try to define success before I start tuning the solution. Here, success was not a small loss value. It was reaching the treasure from every legal starting cell without a random action rescuing a weak policy. From there I could work backward through the state, actions, rewards, targets, and training loop.

I also prefer small checks before an expensive run. Testing maze boundaries and reward behavior is much faster than waiting for training only to discover that the environment was wrong. A fixed seed helps with debugging, but I still want a test that examines the final behavior directly.

### What are my ethical responsibilities?

I owe the end user a clear description of what the system can and cannot do. I owe the organization the same honesty, even when one convenient metric makes the project look finished. In this case, a low loss did not prove that the pirate could navigate the maze, so reporting loss alone would have been misleading.

The reward function deserves the same attention. An agent follows the objective it receives, not the intention that never made it into the code. People remain responsible for choosing that objective, checking failure cases, protecting user interests, and deciding whether the evidence is strong enough for the intended use.

## Limits and why they matter

This is an educational model for a static, fully observable maze. Exhaustively collecting 200 transitions works here because the environment is tiny. A larger maze would make that method expensive, and moving obstacles could leave the stored transitions describing a maze that no longer exists. The project is not evidence for real-world or safety-critical navigation.

This is also where the work connects to my interest in AI governance. The [NIST AI Risk Management Framework 1.0](https://www.nist.gov/itl/ai-risk-management-framework) treats evaluation and documented limits as part of trustworthy AI work. NIST's current [AI measurement and evaluation](https://www.nist.gov/ai-measurement-and-evaluation) material makes a similar point: a useful evaluation has to measure the behavior that matters in context. This notebook is a small example, but the habit carries forward.

## Run locally

Use Python 3.11. Create and activate a virtual environment, then run:

```text
python -m pip install --requirement requirements.txt
jupyter lab Saucier_Bradley_ProjectTwo.ipynb
```

The saved notebook already contains the submitted output. Rerunning the training cell can produce a different stopping epoch when the hardware or software environment changes.

## Tests and continuous integration

Run the focused checks with:

```text
python -m pytest -q
```

The CI workflow checks the dependency set, compiles the helper modules, validates the notebook, tests maze behavior, and confirms that the declared network builds. It does not retrain the model on every commit. That would add a slow and potentially flaky check without telling me much about a routine source change.

## Provenance and reuse

TreasureMaze.py, GameExperience.py, and the original notebook scaffold are course-provided material. The repository copy of TreasureMaze.py includes two small corrections to preserve its documented invalid-action behavior. The final training and evaluation work is in the notebook.

No open-source license is attached because the reuse rights for the supplied course code have not been established. This repository is a portfolio artifact, not a grant of permission to reuse that material.
