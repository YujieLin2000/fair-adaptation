
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
import AXY
from AXY import plot_sweep_axy, script_experiments_axy
from normal_pkg import adaptation, distances, plot_adaptation, plot_distances

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate the speed of adpatation of cause-effect models.')
    parser.add_argument('distribution', type=str, choices=['categorical', 'normal'])
    parser.add_argument('action', type=str, choices=['distance', 'adaptation', 'plot'])

    args = parser.parse_args()

    if args.distribution == 'categorical':
        results_dir = 'axy_categorical_results'
        if args.action == 'distance':
            # categorical.script_experiments.all_distances(savedir=results_dir)
            raise DeprecationWarning("Categorical distances does not work.")

        elif args.action == 'adaptation':
            for init_dense in [True, False]:#
                for k in [10]:
                    for intervention in ['X','A','AandX','Y']:#
                        AXY.script_experiments_axy.parameter_sweep(
                            intervention, k, init_dense, savedir=results_dir)

        elif args.action == 'plot':
            for dense in [True, False]:
                AXY.plot_sweep_axy.all_plot(dense=dense, input_dir=results_dir)


