~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hyperparameter Optimization Experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
``olympus/studies/hpo/configs``.

Here's the ``tiny`` task for example:

.. code-block:: yaml

   function: 'olympus.baselines.tiny.main'

   objective: 'validation_error_rate'

   variables:
       random_state: 332
       bootstrap_seed: 1

   defaults: {}

   fidelity: null

   space:
       max_depth: 'loguniform(2, 100)'
       min_samples_split: 'uniform(0, 1)'
       min_samples_leaf: 'loguniform(1, 100)'
       min_weight_fraction_leaf: 'uniform(0, 0.5)'

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

The ``defaults`` are default values to pass to all tasks that are not hyperparameters.

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

   $ python olympus/studies/hpo/main.py \
       --uri 'mongo://{username}:{password}@{host}/{db}?authSource={db}' \
       --database {db} \
       --config olympus/studies/hpo/configs/tiny.yaml \
       --namespace tiny-hpo \
       --num-experiments 20 \
       --budget 200 \
       --save-dir olympus/studies/hpo/results

This will register a random search algorithm in the database and wait for
the algorithm to complete before parsing the results and saving them
in ``olympus/studies/hpo/results/tiny.json``. You can run this
from your laptop, no need to run it on the cluster. The script is resumable.
Also, once it has registered the hyperparameter optimization, the worker are able to do most
of the work, so as long as the script completed registering the tasks it can be stopped
for a while and workers will do their job anyhow.

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

The results should be located at ``olympus/studies/hpo/results/{task_name}.json``

You can use the function at ``olympus/studies/hpo/main.load_results`` to load the data.

The data is divided in ``xarray.Dataset`` for each hyperparameter optimization algorithms.

.. code-block:: json

   {
      'grid_search': xarray.Dataset,
      'nudged_grid_search': xarray.Dataset,
      'noisy_grid_search': xarray.Dataset
      'random_search': xarray.Dataset,
      'bayesopt': xarray.Dataset
   }

Each dataset gives information for all hyperparameter optimization runs.

.. code-block:: python
   Dimensions:                   (epoch: 2, noise: 2, order: 200, params: 4, seed: 50)
   Coordinates:
     + params                    (params) <U24 'max_depth' ... 'min_weight_fraction_leaf'
     + noise                     (noise) <U14 'bootstrap_seed' 'random_state'
     + epoch                     (epoch) int64 0 1
     + order                     (order) int64 0 1 2 3 4 5 ... 195 196 197 198 199
       uid                       (seed, order) <U16 '4a925ecef813b1e1' ... '35726e4af2c0518c'
     + seed                      (seed) int64 3966912 44556670 ... 1067342868
       max_depth                 (order, seed) float64 2.83 7.901 ... 16.22 79.51
       min_samples_leaf          (order, seed) float64 32.91 2.298 ... 98.47 7.175
       min_samples_split         (order, seed) float64 0.125 0.8606 ... 0.218
       min_weight_fraction_leaf  (order, seed) float64 0.3137 0.4291 ... 0.3681
       bootstrap_seed            (order, seed) float64 1.0 1.0 1.0 ... 1.0 1.0 1.0
       random_state              (order, seed) float64 332.0 332.0 ... 332.0 332.0
       namespace                 (seed) <U30 'test-with-robo-2-bayesopt-s-0' ... 'test-with-robo-2-bayesopt-s-9'
   Data variables:
       elapsed_time              (epoch, order, seed) float64 3e-06 ... 0.05819
       sample_count              (epoch, order, seed) float64 0.0 0.0 ... 398.0
       test_accuracy             (epoch, order, seed) float64 nan nan ... 0.0314
       test_error_rate           (epoch, order, seed) float64 nan nan ... 0.9686
       test_loss                 (epoch, order, seed) float64 nan nan ... 0.0 0.0
       test_time                 (epoch, order, seed) float64 nan nan ... 0.000111
       validation_accuracy       (epoch, order, seed) float64 nan nan ... 0.0302
       validation_error_rate     (epoch, order, seed) float64 nan nan ... 0.9698
       validation_loss           (epoch, order, seed) float64 nan nan ... 0.0 0.0
       validation_time           (epoch, order, seed) float64 nan nan ... 0.0001545}


All source of variations are defined by ``dataset.noise`` and hyperparameters of the different
trials are defined by ``dataset.params``. The different optimization runs are indexed by 
``data.seed``. To select the i-th run, do ``dataset.sel(seed=0)``. Similarly, to get last epoch 
do ``dataset.sel(epoch=-1)``.
