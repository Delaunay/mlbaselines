Olympus
=======

Decades of machine learning research at your fingertips.

Features
~~~~~~~~

- Deterministic Blocks
- Reproducible baselines for a variety of tasks
- Integrated Hyperparameter Optimizer (Orion)
- Experiment Tracking
- Model Zoo
- Pretrained Models
- Multi GPU training
- Automatic Checkpointing
- Mixed precision Available


Baselines
~~~~~~~~~

Run any baselines in a few lines of code

.. code-block:: bash

    $ pip install olympus
    $ export OLYMPUS_DATA_PATH=/fast
    $ olympus --devices 0 classification --batch-size 32 --epochs 10 --dataset mnist --model resnet18
    {
      "train_accuracy": 0.6458333333333334,
      "train_loss": 2.109870990117391,
      "elapsed_time": 9,
      "sample_count": 960,
      "epoch": 9,
      "adversary_accuracy": 0.3020833333333333,
      "adversary_loss": 2.234758218129476,
      "adversary_distortion": 0.2575291295846303,
      "validation_accuracy": 0.5986421725239617,
      "validation_loss": 2.108673614815782
    }
    {
      "temperature.gpu": 34.083333333333336,
      "utilization.gpu": 10.333333333333334,
      "utilization.memory": 0.0,
      "memory.total": 32480.0,
      "memory.free": 31672.833333333332,
      "memory.used": 807.1666666666666
    }


Deterministic Blocks
~~~~~~~~~~~~~~~~~~~~

Writing a full pipeline has never been easier,
even when optimizing over hyper parameters !

.. literalinclude:: ../examples/hpo_simple.py
   :language: python
   :linenos:


Install
~~~~~~~

.. code-block:: bash

    pip install git+git://github.com/mila-iqia/olympus.git


with fANOVA
-----------

.. code-block:: bash

    sudo apt-get install swig
    # pip install pyrex
    pip install fanova