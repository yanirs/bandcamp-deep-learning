# Bandcamp Deep Learning

A small project for classifying Bandcamp album covers by their genre.

# Installation

* Locally: 
    * Create virtual environment
    * [Ubuntu 14.04] Install packages from `requirements-apt.txt`
    * Install Python requirements from `requirements.txt`
    * Install Lasagne: http://lasagne.readthedocs.org/en/latest/user/installation.html

* Remotely (on Ubuntu 14.04):
    * Install Fabric
    * Run `fab -i <pem_file> -H <host> -u <user> deploy`

# Data collection

**TODO:** add files & explanations

# Running experiments

Command line to run experiments:

    $ workon bandcamp-deep-learning
    $ python experiment.py run_experiment --help

See notebooks for some examples and experimental results.
