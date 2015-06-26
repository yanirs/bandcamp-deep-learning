# pylint: disable=unused-import
from warnings import filterwarnings
from commandr import Run
import sys

import data
import experiment
import hypersearch


if __name__ == '__main__':
    # Filter unnecessary Lasagne warnings
    for warning in ['The uniform initializer no longer uses Glorot',
                    r'get_all_layers\(\) has been changed to return layers in topological order']:
        filterwarnings('ignore', warning)
    print('Command line: %s' % ' '.join(sys.argv))
    Run()
