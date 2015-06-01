from fabric.api import cd, env, local, put, run, sudo, task
from fabric.contrib import files

env.project_name = 'bandcamp-deep-learning'
env.virtualenv_path = '~/.virtualenvs/bandcamp-deep-learning'


def setup_virtualenv():
    """Set up the project's virtual environment (bandcamp-deep-learning) and install requirements."""

    read_requirements = lambda requirement_file: ''.join(open(requirement_file)).replace('\n', ' ')

    # Install apt requirements
    sudo('apt-get -qy update')
    sudo('apt-get -qy install %s' % read_requirements('requirements-apt.txt'))

    # Create virtual environment and install pip requirements
    if not files.exists(env.virtualenv_path):
        run('virtualenv %s' % env.virtualenv_path)
    run('%s/bin/pip install -q %s' % (env.virtualenv_path, read_requirements('requirements.txt')))

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
    local(r'7z a -x@.gitignore -x\!.git/ -r deploy.7z ./')
    run('rm -rf %s-prev' % env.project_name)
    run('mv %s %s-prev' % (env.project_name, env.project_name))
    run('mkdir %s' % env.project_name)
    put('deploy.7z', env.project_name)
    with cd(env.project_name):
        run('7z x deploy.7z')
        run('rm deploy.7z')
    local('rm deploy.7z')


@task
def deploy():
    """Create a virtualenv, update requirements, and deploy the project."""
    setup_virtualenv()
    package_and_upload_project()
