# LTL2Action

This is the code repository accompanying the paper [**LTL2Action: Generalizing LTL Instructions for Multi-Task RL**](https://arxiv.org/abs/2102.06858), which will appear in ICML 2021. 

In the **animation** below, our agent's behaviour, compared to a myopic one is demonstrated on ``ZoneEnv``, a custom environment based on OpenAI's SafetyGym:

<p align="center">
    <img width="700" src="https://github.com/LTL2Action/LTL2Action/blob/master/README_files/zone_env.gif">
<!--     <figcaption class="figure-caption text-center">Figure 1. (animation) Myopic agent vs. ours. </figcaption> -->
</p>

## New Installation instructions
1. Create conda environment (only works for Python 3.6 or 3.7)
    - `conda create -n ltl2action python=3.7`
    - `conda activate ltl2action`
2. Install LTL2Action
    - `git clone git@github.com:jasonxyliu/LTL2Action.git`
    - `cd LTL2Action`
    - `pip install -r requirements.txt`
3. Install dependencies
    - `conda install -c conda-forge spot`
    - `conda install -c conda-forge dill`
    - `pip install ring`
4. (Optional) to run without MuJoCo installation, comment out ZonesEnv in the following files
    - `envs/__init__.py`
    - `utils/format.py`
    - `env_models.py`


## Installation instructions


We recommend using Python 3.6 to run this code.

1. `pip install -r requirements.txt`
2. Install [Spot-2.9](https://spot.lrde.epita.fr/install.html)
    - Follow the installation instructions at the link. Spot should be installed in `/usr/local/lib/python3.6/site-packages/spot`. This step usually takes around 20 mins.
3. (Optional) To run the OpenAI Safety Gym and the customized ZoneEnv based on Safety Gym, you will need Mujoco installed, as well as an [active license](http://www.mujoco.org/index.html). 
    - `pip install mujoco-py==2.0.2.9`
    - `pip install -e src/envs/safety/safety-gym/`

## Training Agents

Instructions for training and evaluating RL agents on each of our domains is available in the [`src`](src/) folder.

## Citation

```
@article{DBLP:journals/corr/abs-2102-06858,
  author    = {Pashootan Vaezipoor and
               Andrew C. Li and
               Rodrigo Toro Icarte and
               Sheila A. McIlraith},
  title     = {LTL2Action: Generalizing {LTL} Instructions for Multi-Task {RL}},
  journal   = {CoRR},
  volume    = {abs/2102.06858},
  year      = {2021},
  url       = {https://arxiv.org/abs/2102.06858},
  archivePrefix = {arXiv},
  eprint    = {2102.06858},
  timestamp = {Thu, 18 Feb 2021 15:26:00 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2102-06858.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
