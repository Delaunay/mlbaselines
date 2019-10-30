Olympus
-------

.. code-block:: bash

    $ pip install olympus
    $ olympus --devices 0 1 2 3 --name classification --batch-size 32 --epochs 10 --seed 0
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
