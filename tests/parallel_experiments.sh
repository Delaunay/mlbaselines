#!/bin/bash

rm track_test.json track_test.json.lock | true
rm /tmp/track_test.json /tmp/track_test.json.lock | true

olympus classification --batch-size 32 --epochs 10 --dataset mnist --model logreg --optimizer sgd  &
olympus classification --batch-size 32 --epochs 10 --dataset mnist --model logreg --optimizer adam &

wait
