First Steps
===========

Olympus tries to stay true to pytorch as much as possible.
While learning to use olympus you will often find yourself writing code that looks like pytorch.
Indeed, on a high level Olympus is a pytorch wrapper made to automate repetitive tasks.

For example, the code below initialize ``resnet18`` a standard model that is included in olympus to
classify mnist images with mixed precision training!

.. code-block:: python

    import torch
    from olympus.models import Model

    model = Model(
        name='resnet18'                      # Name of the model to initialize
        input_size=(1, 28, 28),              # Input Size (some model are dynamic)
        output_size=(10,),                   # Output Size, number of classes in case of classifications
        initialization='glorot_uniform',     # weight initialization method
        seed=0,                              # seed used by the weight initialization method
        half=True                            # Enable Mixed Precision computation!
    )

    model = model.cuda()
    test_input = torch.randn((28, 28)).cuda()
    out = model(x)
    print(out)

You can also use your own models with Olympus'Model class, which enables you to get
mixed precision and custom initialization for free!

.. code-block:: python

    from olympus.models import Model

    class MyModel(nn.Module):
        def __init__(self, input_size, output_size):
            self.main = nn.Linear(input_size[0], output_size[0])

        def forward(self, x):
            return self.main(x)

    model = Model(
        model=MyModel,
        input_size=(290,),
        output_size=(10,)
    )

All other machine learning basic blocks such as ``Dataset``, ``Optimizer``, ``LRSchedule`` also have their Olympus variant!

Let's have a closer look at ``Optimizer`` as it will introduce you on how Olympus handles hyper parameters!

The code below creates a SGD optimizer and enables mixed precision.
As you can see none of its parameters are set, this is because Olympus classes are lazy and will try to delay
the creation of a block as much as possible.
This is to enable hyper parameter optimizer, such as Orion, to pick the optimizer's parameters if they have any.

.. code-block:: python

    from olympus.optimizers import Optimizer

    optimizer = Optimizer(
        name='sgd',
        half=True
    )

For example, I can check which hyper parameter are missing from my optimizer using ``get_space()`` and set
the parameters using ``init``, after ``init`` the optimizer is instantiated and ready to be used!

.. code-block:: python

    >>> optimizer.get_space()
    {'lr': 'loguniform(1e-5, 1)', 'momentum': 'uniform(0, 1)', 'weight_decay': 'loguniform(1e-10, 1e-3)'}

    >>> optimizer.init(
        lr=1e-5,
        momentum=0.99,
        weight_decay=1e-4
    )

Olympus comes with its own integrated hyper optimizer, Orion, so you will never have to set them yourself!
You can find below a fully functional example.
It is only 40 lines long and yet supports multi gpu training, mixed precision, hyper parameter search and more!

.. literalinclude:: ../../examples/hpo_simple.py
   :language: python
   :linenos:
