from fabric.api import cd, env, local, put, run, sudo, task
from fabric.contrib import files

env.project_name = 'bandcamp-deep-learning'
env.virtualenv_path = '~/.virtualenvs/bandcamp-deep-learning'


def setup_virtualenv():
    """Set up the project's virtual environment (bandcamp-deep-learning) and install requirements."""

    # Install apt requirements
    sudo('apt-get -qy update')
    sudo('apt-get -qy install %s' % ''.join(open('requirements-apt.txt')).replace('\n', ' '))

    # Create virtual environment and install pip requirements
    if not files.exists(env.virtualenv_path):
        run('virtualenv %s' % env.virtualenv_path)
    for requirement in open('requirements.txt'):
        run('%s/bin/pip install %s' % (env.virtualenv_path, requirement))

    # Install Lasagne
    if not files.exists('lib/Lasagne'):
        run('mkdir -p lib')
        with cd('lib'):
            run('git clone https://github.com/Lasagne/Lasagne.git')
            with cd('Lasagne'):
                run('%s/bin/pip install -r requirements.txt' % env.virtualenv_path)
                run('%s/bin/python setup.py install' % env.virtualenv_path)


def package_and_upload_project():
    """Package the project and upload it to the remote environment."""
    local(r'7z a -x@.gitignore -x\!.git/ -x\!\*.ipynb -r deploy.7z ./')
    run('rm -rf %s-prev' % env.project_name)
    if files.exists(env.project_name):
        run('mv %s %s-prev' % (env.project_name, env.project_name))
    run('mkdir %s' % env.project_name)
    put('deploy.7z', env.project_name)
    with cd(env.project_name):
        run('7z x deploy.7z')
        run('rm deploy.7z')
    local('rm deploy.7z')


@task
def test_cuda():
    """Test that the CUDA installation works.

    This assumes that CUDA has been installed on Ubuntu 14.04, as described here:
    https://github.com/BVLC/caffe/wiki/Caffe-on-EC2-Ubuntu-14.04-Cuda-7
    """
    run('/usr/local/cuda/bin/cuda-install-samples-7.0.sh ~/')
    with cd('NVIDIA_CUDA-7.0_Samples/1_Utilities/deviceQuery'):
        run('make')
        run('./deviceQuery')


@task
def deploy(skip_env_setup=False):
    """Create a virtualenv, update requirements, and deploy the project."""
    if not skip_env_setup:
        setup_virtualenv()
    package_and_upload_project()
