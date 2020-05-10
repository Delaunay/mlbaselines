~~~~~~~~~~~~~~~~~~~~~
Reproducibility tests
~~~~~~~~~~~~~~~~~~~~~

Installation
------------

.. code-block:: bash

   $ git clone git@github.com:mila-iqia/olympus.git
   $ pip install -e olympus

For optional dependencies, for example for running the NLP tasks, 
you can install them with:

.. code-block:: bash

    $ pip install -e olympus[nlp]

Make sure to download the data you need and set
the environment variable ``OLYMPUS_DATA_PATH`` to the
folder where the data is stored.

Configuration
-------------

All configuration files are located in
``olympus/studies/variance/configs``.

Here's the ``tiny`` task for example:

.. code-block:: yaml

   function: 'olympus.baselines.tiny.main'

   objective: 'validation_error_rate'

   variables:
       random_state: 1
       bootstrap_seed: 1

   resumable: 0

   defaults: {}

   params:
       max_depth: 10
       min_samples_split: 2
       min_samples_leaf: 1
       min_weight_fraction_leaf: 0

The ``function`` must be a string pointing 
to the function that will be used for training.
Make sure that it supports ``uid``,
``experiment_name`` and ``client``,
which are required to log the metrics in the database.

The ``objective`` is the metric that will be used to verify reproducibility.

The ``variables`` are the sources of variation that will be 
tested. The values passed will serve as the default values while
we vary another variable. For each variable seperately, we will execute
the training with n different seeds (sequential from 1 to n).

The ``resumable`` option tells the pipeline if it should run tests for resumption as well.
The tiny example trains in a single step so there is no room for resumption. For
any training with gradient descent you should set ``resumable: 1``.

The ``defaults`` are default values to pass to all tasks that are not hyperparameters.

The ``params`` are the default hyperparameter values to use for these experiments.
Ideally, use good defaults from the literature. If unapplicable, use
results from the *study* ``searchspace`` to select good values.

Note that ``epoch`` must be defined in ``defaults`` if your task needs it. Otherwise
it is set by default to 1.

Execution
---------

The execution is divided between a master process and workers.

The master process can be started using the ``main.py`` script:

.. code-block:: bash

   $ python olympus/studies/repro/main.py \
       --uri 'mongo://{username}:{password}@{host}/{db}?authSource={db}' \
       --database {db} \
       --config olympus/studies/repro/configs/tiny.yaml \
       --namespace tiny-repro \
       --num-experiments 10 \
       --num-repro 10 \
       --save-dir olympus/studies/repro/results

This will register all tasks to test the difference sources of variation and wait for
all tasks to complete before parsing the results and saving them
in ``olympus/studies/repro/results/repro_tiny-repro.json``. You can run this
from your laptop, no need to run it on the cluster. The script is resumable.
You can restart the script to monitor the processes or fetch the final results to get
the json file.

To execute the trials you must start workers with:

.. code-block:: bash
   
   $ olympus-hpo-worker \ 
       --uri 'mongo://{username}:{password}@{host}/{db}?authSource={db}' \
       --database {db} \
       --rank 1

If your task is resumable, make sure to first set the environment variable
OLYMPUS_STATE_STORAGE to the folder where checkpoints will be saved.

If you use pre-trained models, don't forget to set
OLYMPUS_MODEL_CACHE to the folder where they are saved.

And finally don't forget to set OLYMPUS_DATA_PATH.

For execution on the cluster (namely on Beluga) see the example script at
``olympus/studies/searchspace/{cluster_name}.sh``. The workers will do the heavy job.

Results
-------

TODO: Where are the results files? How do we make the plots?
