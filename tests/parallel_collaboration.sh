#!/bin/bash

rm track_test.json track_test.json.lock | true
rm /tmp/track_test.json /tmp/track_test.json.lock | true

for i in {0..1}; do
  echo $i
  olympus classification --batch-size 32 --epochs 10 --dataset mnist --model logreg &
done

wait
