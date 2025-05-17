#!/bin/bash


python plot_evaluation.py --env simple_dir --save_fig
python plot_evaluation.py --env pen_goal --save_fig
python plot_evaluation.py --env cart_goal --save_fig
python plot_evaluation.py --env cheetah_vel --save_fig
python plot_evaluation.py --env ant_dir --save_fig