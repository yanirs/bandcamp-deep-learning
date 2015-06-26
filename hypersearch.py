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
    """Run a sequential hyperparameter search using hyperopt.fmin().

    To enable restartability and reproducibility, each hyperparameter combination is evaluated by calling base_cmd
    with the extra hyperparameters, and logging the results to a (hopefully) unique file in log_dir.

    The experiment-running command is called in a separate shell rather than in-process, because it may depend on
    global random state. Calling run_experiment() successively in-process would hurt reproducibility.

    Arguments:
     * base_cmd (str) - base command line to call, typically "python manage.py run_experiment ..." with a small number
                        of epochs (see experiment.run_experiment())
     * log_dir (str) - path where the outputs of individual runs will be stored.
                       Note: each run's filename is "experiment.<command_line_hash>.log", meaning that collisions are
                       possible in very rare cases
     * base_model_params (str) - model_params to pass to the run_experiment command that are not part of the search
     * model_params_space (str) - model_params to experiment with, in the same format as normal model_params, except
                                  that each key is of the form <param_name>__<hp_func_name>, where hp_func_name is
                                  a member of hyperopt.hp, and the parameter value is interpreted as arguments to
                                  pass to hp_func_name.
                                  For example, if the architecture is ConvNet, "ld0_dropout__uniform=0.0,0.75" will
                                  experiment with dropout values drawn from uniform(0.0, 0.75) for the dropout layer
                                  that comes after the first dense layer.
     * max_evals (int) - number of experiments to run.
                         Note: it's possible to run once with a small value of max_evals, and the do a subsequent run
                         with an increased number of experiments. The second run will read the results of the first run
                         and continue from the point where that run stopped.
     * learning_rate_range (str) - a pair of comma-separated values that specifies the range from which the
                                   learning_rate will be drawn, according to hyperopt.hp.loguniform
    """

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
    hyperopt.fmin(lambda param_dict: _eval_objective(param_dict, log_dir, base_cmd), space=space,
                  algo=hyperopt.tpe.suggest, trials=trials, max_evals=max_evals)
    print('---\nBest command line: %(cmd)s\nError rate: %(loss).2f%%' % trials.best_trial['result'])


def _create_command_args(param_dict):
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


def _eval_objective(param_dict, log_dir, base_cmd):
    # TODO: handle hash collisions? unlikely to be an issue with a small number of experiments.
    cmd_args = _create_command_args(param_dict)
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

