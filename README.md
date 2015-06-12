# Bandcamp Deep Learning

A small project for classifying Bandcamp album covers by their genre.

# Installation

* Locally: 
    * Create virtual environment (assumed to be named `bandcamp-deep-learning` below)
    * [Ubuntu 14.04] Install packages from `requirements-apt.txt`
    * Install Python requirements from `requirements.txt`
    * Install [Lasagne](http://lasagne.readthedocs.org/en/latest/user/installation.html)

* Remotely (on Ubuntu 14.04):
    * Install Fabric
    * Run `fab -i <pem_file> -H <host> -u <user> deploy` to setup the virtual enviroment, and deploy the project to
      the server. The project directory will be `~/bandcamp-deep-learning` and the virtual environment will be named
      `bandcamp-deep-learning` (surprisingly).

# Data collection

Download the images to, e.g., `~/data/images` by running:

    $ python manage.py download_dataset_images --out-dir ~/data/images \
                                               --dataset-links-tsv dataset-links.tsv
    
This may take a while. You can check progress by running:

    $ find ~/data/images -maxdepth 1 -printf "echo \`ls %p | wc -l\`'\t'%p\n" | bash

After the downloads finish, you should have 10 sub-directories under `~/data/images`, each representing a genre, with
1000 album covers in each sub-directory. These were chosen at random from common album tags, as crawled for
[Bandcamp Recommender](http://www.bcrecommender.com/). Bandcamp album tags are assigned by artists, and each album
can have multiple tags. The chosen albums match only one of the chosen tags/genres.

Then, create the training/validation/testing split and dataset pickles by running:

    $ python manage.py create_datasets ~/data/images ~/data

**Note:** as a sanity check, it may be worth downloading the MNIST dataset and converting it to the format used by the
`run_experiment` command. This can be done by running:

    $ python manage.py download_mnist ~/data

# Running experiments

Check command line help to run experiments:

    $ python manage.py run_experiment --help

For example, to run a multi-layer perceptron with a single hidden layer and 256 hidden units (assuming you created the
dataset as above):

    $ THEANO_FLAGS=floatX=float32 python manage.py run_experiment \
        --dataset-path ~/data/local.pkl.zip --model-architecture SingleLayerMlp \
        --model-params num_hidden_units=256

See notebooks for some examples and experimental results (though the notebooks may not be runnable due to recent code
changes).

## Using a GPU

Set up CUDA as described in one of the many manuals (I simply used the AWS AMI from
[Caffe](https://github.com/BVLC/caffe/wiki/Caffe-on-EC2-Ubuntu-14.04-Cuda-7)) and run experiments with additional
with the correct environment variables. For example:

    $ THEANO_FLAGS=device=gpu,floatX=float32 python manage.py run_experiment \
        --dataset-path ~/data/local.pkl.zip --model-architecture SingleLayerMlp

This should be much faster than running with the default settings.

## Experimenting with the full dataset on a GPU

Depending on the size of your GPU's memory, you may need to copy the training dataset to the GPU in chunks. This is
achieved by passing the --training-chunk-size parameter to run_experiment. For example, the following would work on
an AWS g2.2xlarge instance (GPU memory of 4GB).

    $ THEANO_FLAGS=device=gpu,floatX=float32 python manage.py run_experiment \
        --dataset-path ~/data/full.pkl.zip --model-architecture SingleLayerMlp \
        --batch-size 500 --training-chunk-size 4000
