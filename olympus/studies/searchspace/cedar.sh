#!/usr/bin/env bash

#SBATCH --array=1-{NUM_JOBS}
#SBATCH --cpus-per-task=8
#SBATCH --output=olympus.{TASK}.%A.%a.out
#SBATCH --error=olympus.{TASK}.%A.%a.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=olympus-{TASK}
#SBATCH --mem=32GB
#SBATCH --time=5:59:00
#SBATCH --account={ACCOUNT}

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
echo -e "\n$(date) Cloning Olympus repo\n\n"
mkdir $SLURM_TMPDIR/code
cd $SLURM_TMPDIR/code
git clone git@github.com:mila-iqia/olympus.git
echo -e "\n$(date) Install \n\n"
pip install olympus/requirements.txt
pip install -e olympus

# 2. Copy data
mkdir $SLURM_TMPDIR/data
echo -e "\n$(date) Copy dataset\n\n"
cp -r $OLYMPUS_DATA_PATH/{DATA_FOLDER_NAME} $SLURM_TMPDIR/data

export OLYMPUS_DATA_PATH=$SLURM_TMPDIR/data

mkdir -p ${WORKING_DIR}
mkdir -p ${WORKING_DIR}/logs

# Avoid progress bar on cluster
export OLYMPUS_PROGRESS_FREQUENCY_EPOCH=0
export OLYMPUS_PROGRESS_FREQUENCY_BATCH=0

CREDENTIALS="${SIMUL_HPO_DB}:${SIMUL_HPO_DB_PASSWORD}"
MONGO_URI="mongo://$CREDENTIALS@$MONGODB_HOST:$MONGODB_PORT/$SIMUL_HPO_DB?authSource=$SIMUL_HPO_DB"

echo -e "\n$(date) Starting olympus workers\n\n"
pids=
for WORKER_ID in {1..{NUM_WORKER_PER_GPU}}
do
  echo -e "\n$(date) Starting olympus worker $WORKER_ID\n"
  olympus-hpo-worker --uri $MONGO_URI --database $SIMUL_HPO_DB --rank $WORKER_ID &
  pids+=" $!"
done

sleep 1

echo $pids
wait $pids || { echo "\n$(date) there were errors\n" >&2; exit 1; }
