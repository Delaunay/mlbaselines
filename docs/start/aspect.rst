Observers and Aspects
=====================

The definition of Aspect Oriented Programming given by [wikipedia]_ is

.. [wikipedia] In computing, aspect-oriented programming (AOP) is a programming paradigm
    that aims to increase modularity by allowing the separation of cross-cutting concerns.
    It does so by adding additional behavior to existing code (an advice) without modifying the code itself,
    instead separately specifying which code is modified via a "pointcut" specification,
    such as "log all function calls when the function's name begins with 'set'".
    This allows behaviors that are not central to the business logic
    (such as logging) to be added to a program without cluttering the code, core to the functionality.


For machine learning, it means that all the functionality that is not tied to learning/training/inference should not be
mixed up together.

It is common to find online machine learning examples where all functionality are bundle together in a single main script.
While it might be a convenient for some things. Olympus is focused on making reusable components that you can reuse
across many models and tasks.


What not to do
~~~~~~~~~~~~~~

Below, you can find a bad example that you should NOT do.
It trains a network, compute validation accuracy and save checkpoints.
This is a lot of functionalities mixed together which makes the code cluttered, hard to understand and difficult to
reuse.
In the example below, the core logic is training everything else is noise.


.. code-block:: python

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


Observers
~~~~~~~~~

First of all, we can remove the clutter very easily by using Observers.
Instead of adding the additional code to the training loop, we will build a new observer object that will hold the logic.
The code will be executed once it receives the necessary event.
In our example the event is ``new_epoch``.

Now we see that we can simply add more functionalities through observers without modifying our original code.
The training code stays lean and simple.

.. code-block:: python

    # State of our Task
    task = Task()

    # List of features we want to enable
    obs = ObserverList()
    obs.append(ValidationAccuracy())
    obs.append(Checkpointer())

    for epoch in range(args.start_epoch, args.epochs):
        # evaluate on validation set
        train(train_loader, task.model, task.criterion, task.optimizer, epoch)

        # Send an event to all the observers so they can run
        # their logic
        obs.new_epoch(epoch, task)

You can find below an example of the implementation of an observer

.. code-block:: python

    class ValidationAccuracy:
        def on_new_epoch(epoch, task):
            task.prec1 = validate(val_loader, task.model, task.criterion)


Aspect
~~~~~~

TODO