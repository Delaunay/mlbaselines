~~~~~~~~~~~~~~~~~~~~~~
Simulation Experiments
~~~~~~~~~~~~~~~~~~~~~~

Installation
------------

Locally
=======

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

Clusters
========

To setup on the clusters, copy the corresponding file
in your home from `olympus/studies/search_space/{cluster_name}.rc` and
add `source {cluster_name}.rc` at the end of your `.bashrc` file.

Make sure to replace all these templates with proper values.

:{DB_URL}:
:{DB_NAME}:
:{DB_PASSWORD}:

If using beluga, copy the wheels in `olympus/wheels/beluga/` to `$WHEEL_DIR`
and git clone the repo at $PROJECT/repos/.

Download your data and save it in `$OLYMPUS_DATA_PATH`.

You can use the corresponding file 
`olympus/studies/search_space/{cluster_name}.sh` to submit jobs.
You will need to replace the following templates

:{NUM_JOBS}:           Number of jobs to submit on the cluster.
:{TASK}:               The task name.
:{DATA_FOLDER_NAME}:   The name of the dataset folder under $OLYMPUS_DATA_PATH.
:{ACCOUNT}:            Account for compute-canada allocation.
:{DATA_FOLDER_NAME}:   This is to copy your specific dataset to local drive, not all datasets.
:{NUM_WORKER_PER_GPU}: Number of workers that can run on the same GPU simultaneously.

Configuration
-------------

All configuration files are located in
``olympus/studies/simul/configs``.

Here's the ``logreg`` task for example:

.. code-block:: yaml

   function: 'olympus.baselines.logreg.main'

   objective: 'validation_error_rate'

   variables:
       random_state: 1
       bootstrap_seed: 1

   defaults: {}

   fidelity: null

   space:
       C: 'uniform(0, 1)'
       l1_ratio: 'uniform(0, 1)'

The ``function`` must be a string pointing 
to the function that will be used for training.
Make sure that it supports ``uid``,
``experiment_name`` and ``client``,
which are required to log the metrics in the database.

The ``objective`` is the metric that will be used for the hyperparameter optimization.

The ``variables`` are the sources of variation that will be 
investigated. The values passed will serve as the default values while
we vary another variable. For each variable seperately, we will execute
the training with n different seeds (sequential from 1 to n).

The ``fidelity`` is used only for task with stocastic gradient descent.
Leave it to ``null`` if you don't need it, otherwise set it like this

.. code-block:: yaml

   fidelity:
      min: 1
      max: 120
      base: 4
      name: 'epoch'

And adjust ``max`` to the maximum number of epochs you would like to use. This fidelity
config will be used by the hyperparameter optimisazation algorithm Hyperband.

The ``space`` is the search space used for the hyperparameter optimisazation.
Make sure to build it wide enough to avoid missing good values. We will only
support real hyperparameter, so for any discrete hyperparameter
you will need to cast the values within ``function``. For each hyperparameter
we will set a prior ``uniform(min, max)`` or ``loguniform(min, max)`` which 
will be used to guide the algorithms.

Execution
---------

The execution is divided between a master process and workers.

The master process can be started using the ``main.py`` script:

.. code-block:: bash

   $ python olympus/studies/simul/main.py \
       --uri 'mongodb://{username}:{password}@{host}/{db}?authSource={db}' \
       --database {db} \
       --config olympus/studies/simul/configs/logreg.yaml \
       --namespace logreg-simul \
       --num-replicates 100 \
       --num-simuls 20 \
       --save-dir olympus/studies/simul/results

This will register ``num_replicates`` random search algorithms in the database and wait for
the algorithms to complete. It will monitor the progress of the algorithms and 
when one is completed it will fetch the results and register the corresponding simulated
replicates. This means that a surrogate model will be fitted on the results of the 
hyper-parameter optimization and a simulation will be executed using the surrogate model.
Additionaly, biased replicates, that is, replicates using the best hyper-parameter found with 
random search but with other sources of variation randomized. 
The final results of all these experiments
will be a dictionary of ``ideal`` replicates, the actual random searches, ``biased`` 
replicates, the ones with fixed best hyper-parameters and randomized sources of variation,
``simul-free`` replicates, simulations with randomized sources of variation and 
``simul-fix`` replicates, simulations with fixed sources of variation.
All this will be saved in ``olympus/studies/simul/results/logreg.json``. You can run this
from your laptop, no need to run it on the cluster. The script is resumable
but it is better to keep it always running so that simulations can be started right away
when the random search is done. Note also that the simulation is relatively computationaly
expensive compared to all other operations in ``main.py`` scripts of studies.

To execute the trials you must start workers with:

.. code-block:: bash
   
   $ olympus-hpo-worker \ 
       --uri 'mongodb://{username}:{password}@{host}/{db}?authSource={db}' \
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
