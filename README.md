# Zero Shot Generalization in Reinforcement Learning from Few Training Contexts

## Installation

```bash
conda create -n cmdp python=3.12
```

```bash
conda activate cmdp
pip install -r requirements.txt
```

## Experiments

### Tabular

To run the tabular experiments, just run

```bash
python tabular.py
```

### Context Sample Enhancement (CSE)

To run the CSE experiments from the paper, just run the corresponding script in `scripts/`. For example, to run `simple_dir`:

```bash
./scripts/simple_dir.sh
```

By default, the results will be saved in `ray_results/` on your local machine. Move the experiments into the `experiments/` folder before proceeding to the next step. When this is done, you can run sweeps over the contexts by running the following command:

```bash
python sweep.py --env simple_dir --nmesh 50
```

To plot the results, run the following command:

```bash
python plot_evaluation.py --env simple_dir --save_fig
```

Plots are saved in `figures/`.

These commands should be run for the following environments:

- `simple_dir`
- `ant_dir`
- `cart_goal`
- `cheetah_vel`
- `pen_goal`
