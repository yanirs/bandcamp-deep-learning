# Bandcamp Deep Learning

A small project for classifying Bandcamp album covers by their genre.

# Installation

* Locally: 
    * Create virtual environment (assumed to be named `bandcamp-deep-learning` below)
    * [Ubuntu 14.04] Install packages from `requirements-apt.txt`
    * Install Python requirements from `requirements.txt`
    * Install Lasagne: http://lasagne.readthedocs.org/en/latest/user/installation.html

* Remotely (on Ubuntu 14.04):
    * Install Fabric
    * Run `fab -i <pem_file> -H <host> -u <user> deploy` to setup the virtual enviroment, and deploy the project to
      the server. The project directory will be `~/bandcamp-deep-learning` and the virtual environment will be named
      `bandcamp-deep-learning` (surprisingly).

# Data collection

**TODO:** add files & explanations

# Running experiments

Check command line help to run experiments:

    $ workon bandcamp-deep-learning
    $ cd bandcamp-deep-learning
    $ python experiment.py run_experiment --help

See notebooks for some examples and experimental results.
