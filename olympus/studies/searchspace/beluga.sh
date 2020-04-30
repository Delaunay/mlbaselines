#!/usr/bin/env bash

#SBATCH --array=1-{num-jobs}
#SBATCH --cpus-per-task=10
#SBATCH --output=olympus.{task}.%A.%a.out
#SBATCH --error=olympus.{task}.%A.%a.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=olympus-{task}
#SBATCH --mem=40GB
#SBATCH --time=5:59:00
#SBATCH --account={account}

CODE_DIR=$PROJECT/repos/olympus
WORKING_DIR=$SCRATCH/olympus

# 1. Create env
echo -e "\n$(date) Loading modules\n\n"
module load python/3.6
module load scipy-stack
echo -e "\n$(date) Creating virtualenv\n\n"
virtualenv --no-download $SLURM_TMPDIR/env
echo -e "\n$(date) Source virtualenv\n\n"
source $SLURM_TMPDIR/env/bin/activate
echo -e "\n$(date) Install \n\n"
pip install --no-index -r ${CODE_DIR}/wheel_requirements.txt

# 2. Copy data
mkdir $SLURM_TMPDIR/data
echo -e "\n$(date) Copy dataset\n\n"
cp -r $OLYMPUS_DATA_PATH/cifar-10-batches-py $SLURM_TMPDIR/data

export OLYMPUS_DATA_PATH=$SLURM_TMPDIR/data

mkdir -p ${WORKING_DIR}
mkdir -p ${WORKING_DIR}/logs

# Opening up ssh tunnel with random port
export MONGODB_PORT=$(python -c "from socket import socket; s = socket(); s.bind((\"\", 0)); print(s.getsockname()[1])")
echo -e "\n$(date) Opening ssh tunnel for olympus at port ${MONGODB_PORT}\n\n"

echo "ssh -o StrictHostKeyChecking=no $GATEWAY -L $MONGODB_PORT:$MONGODB_HOST:27017 -n -N -f"
ssh -o StrictHostKeyChecking=no $GATEWAY -L $MONGODB_PORT:$MONGODB_HOST:27017 -n -N -f

# Avoid progress bar on cluster
export OLYMPUS_PROGRESS_FREQUENCY_EPOCH=0
export OLYMPUS_PROGRESS_FREQUENCY_BATCH=0

CREDENTIALS="${SIMUL_HPO_DB}:${SIMUL_HPO_DB_PASSWORD}"
MONGO_URI="mongo://$CREDENTIALS@localhost:$MONGODB_PORT/$SIMUL_HPO_DB?authSource=$SIMUL_HPO_DB"

echo -e "\n$(date) Starting olympus workers\n\n"
pids=
for WORKER_ID in {1..1}
do
  echo -e "\n$(date) Starting olympus worker $WORKER_ID\n"
  olympus-hpo-worker --uri $MONGO_URI --database $SIMUL_HPO_DB --rank $WORKER_ID &
  pids+=" $!"
done

sleep 1

echo $pids
wait $pids || { echo "\n$(date) there were errors\n" >&2; exit 1; }
