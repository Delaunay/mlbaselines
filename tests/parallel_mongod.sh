#!/bin/bash


if [[ -z "${PORT}" ]]; then
  # find a free port to use
  export PORT=$(olympus-port)

  olympus-mongo --port $PORT --loc /tmp&
  MONGO_PID=$!
  export CLEAN_MONGO=True
fi

database_uri="mongodb://localhost:$PORT"

ARGUMENTS="--batch-size 32 --epochs 10 --dataset mnist --model logreg --optimizer sgd  --database $database_uri"

echo $database_uri
olympus classification $ARGUMENTS&
olympus classification $ARGUMENTS&

# wait for the two job to finish
wait %1 %2

if [[ -z "${CLEAN_MONGO}" ]]; then
  # kill mongodb
  kill $MONGO_PID
fi

