"""Functionality for running experiments for optimising hyperparameters."""

from warnings import warn
from ast import literal_eval
import os
import subprocess
from commandr import command
import hyperopt
import numpy as np
from util import parse_param_str


@command
def search_hyperparams(base_cmd, log_dir, base_model_params=None, model_params_space=None, max_evals=10,
                       learning_rate_range=None):
    # TODO: docstring

    def join_command_line_args(param_dict):
        dict_to_param_str = lambda d: ':'.join('%s=%s' % (k, v) for k, v in sorted(d.iteritems()))
        args = []
        for param_name, param_value in sorted(param_dict.iteritems()):
            if param_name == 'mirror_crops':
                if not param_value:
                    args.append('--no-mirror-crops')
            elif param_name == 'model_params':
                if param_value:
                    args.append('--model-params %s' % dict_to_param_str(param_value))
            elif param_name == 'update_func':
                args.append('--update-func-name %s' % param_value.pop('name'))
                args.append('--update-func-kwargs %s' % dict_to_param_str(param_value))
            else:
                args.append('--%s %s' % (param_name, param_value))
        return ' '.join(args)

    def objective(param_dict):
        # TODO: fix possible collisions
        cmd_args = join_command_line_args(param_dict)
        log_filename = os.path.join(log_dir, 'experiment.%s.log' % hash((base_cmd, cmd_args)))
        cmd = '%s %s 2>&1 | tee %s' % (base_cmd, cmd_args, log_filename)
        print
        output = None
        try:
            if os.path.exists(log_filename):
                print('Loading results of %s' % cmd)
                with open(log_filename, 'rb') as log_file:
                    output = log_file.read()
            else:
                print('Running %s' % cmd)
                output = subprocess.check_output(cmd, shell=True)

            if 'OverflowError' in output or 'MemoryError' in output:
                error_rate = np.inf
            else:
                error_rate = 100 - float(output.strip().split()[-1].strip('%'))
        except:
            print('Command output: %s' % output)
            os.unlink(log_filename)
            raise
        print('\tError rate: %.2f%%' % error_rate)
        return dict(loss=error_rate, status=hyperopt.STATUS_OK, cmd=cmd)

    if os.path.exists(log_dir):
        warn('Log directory %s exists. Existing log files may be read to avoid repeating experiments.' % log_dir)
    else:
        os.makedirs(log_dir)

    learning_rate_range = literal_eval(learning_rate_range) if learning_rate_range else (-12, -5)
    model_params = parse_param_str(base_model_params)
    for param_name_and_hp_func, hp_func_args in parse_param_str(model_params_space).iteritems():
        param_name, hp_func_name = param_name_and_hp_func.split('__')
        model_params[param_name] = getattr(hyperopt.hp, hp_func_name)(param_name, *hp_func_args)
    space = dict(
        update_func=hyperopt.hp.choice('update_func', [
            dict(name='adam',
                 beta1=hyperopt.hp.uniform('beta1', 0.0, 0.9),
                 beta2=hyperopt.hp.uniform('beta2', 0.99, 1.0)),
            dict(name='nesterov_momentum',
                 momentum=hyperopt.hp.uniform('momentum', 0.5, 1.0))
        ]),
        learning_rate=hyperopt.hp.loguniform('learning_rate', *learning_rate_range),
        mirror_crops=hyperopt.hp.choice('mirror_crops', [False, True]),
        num_crops=hyperopt.hp.choice('num_crops', [1, 5]),
        model_params=model_params
    )

    trials = hyperopt.Trials()
    hyperopt.fmin(objective, space=space, algo=hyperopt.tpe.suggest, trials=trials, max_evals=max_evals)
    print('---\nBest command line: %(cmd)s\nError rate: %(loss).2f%%' % trials.best_trial['result'])
