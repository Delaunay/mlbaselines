export OLYMPUS_DATA_PATH=/tmp
export OLYMPUS_STATE_STORAGE_TIME=0
export CORES=6

travis: travis-doc travis-unit travis-custom travis-minimalist travis-hpo_simple travis-classification travis-classification-parallel travis-detection travis-classification-fp16 travis-a2c2 travis-a2c travis-end

travis-install:
	pip install -e .
	pip install -r requirements.txt
	pip install -r docs/requirements.txt
	pip install -r tests/requirements.txt

travis-doc: build-doc

travis-minimalist: clean
	COVERAGE_FILE=.coverage.min coverage run examples/minimalist.py

travis-hpo_simple: clean
	COVERAGE_FILE=.coverage.hpo_simple coverage run examples/hpo_simple.py

travis-classification: clean
	COVERAGE_FILE=.coverage.classify coverage run --parallel-mode olympus/baselines/launch.py classification -v 10 --batch-size 32 --min-epochs 0 --epochs 5 --dataset test-mnist --model logreg

travis-classification-hp: clean
	COVERAGE_FILE=.coverage.classify.hp coverage run --parallel-mode olympus/baselines/launch.py classification -v 10 --batch-size 32 --min-epochs 0 --epochs 5 --dataset test-mnist --model logreg --optimizer.lr 0.001

travis-classification-config: clean
	COVERAGE_FILE=.coverage.classify.conf coverage run --parallel-mode olympus/baselines/launch.py classification --arg-file ./examples/arguments.json --dataset test-mnist --min-epochs 0 --epochs 5

travis-classification-parallel: clean
	rm -rf /tmp/olympus/classification | true
	COVERAGE_FILE=.coverage.classify.parallel coverage run --parallel-mode olympus/baselines/launch.py --workers $${CORES} --device-sharing classification -v 10 --batch-size 4 --min-epochs 0 --epochs 5 --dataset test-mnist --model logreg

travis-classification-fp16: clean
	rm -rf /tmp/olympus/classification | true
	COVERAGE_FILE=.coverage.classify_fp16 coverage run --parallel-mode olympus/baselines/launch.py classification --batch-size 32 --min-epochs 0 --epochs 5 --dataset test-mnist --model logreg --half

travis-detection: clean
	COVERAGE_FILE=.coverage.dect_short coverage run --parallel-mode olympus/baselines/launch.py detection --batch-size 2 --min-epochs 0 --epochs 5 --dataset test_pennfudan --model fasterrcnn_resnet18_fpn -v 10

travis-detection-short: clean
	COVERAGE_FILE=.coverage.dect coverage run --parallel-mode examples/detection_simple.py

travis-unit:
	COVERAGE_FILE=.coverage.unit coverage run --parallel-mode -m pytest --cov=olympus tests/unit
	COVERAGE_FILE=.coverage.inte coverage run --parallel-mode -m pytest --cov-append --cov=olympus tests/integration

travis-custom:
	COVERAGE_FILE=.coverage.model coverage run examples/custom_model.py
	COVERAGE_FILE=.coverage.model_nas coverage run examples/custom_model_nas.py
	COVERAGE_FILE=.coverage.optim coverage run examples/custom_optimizer.py
	COVERAGE_FILE=.coverage.schedule coverage run examples/custom_schedule.py

travis-hpo_parallel:
	COVERAGE_FILE=.coverage.hpo_parallel coverage run examples/hpo_parallel.py

travis-end:
	coverage combine
	coverage report -m
	coverage xml
	codecov

travis-a2c: clean
	COVERAGE_FILE=.coverage.a2c coverage run olympus/baselines/launch.py a2c --verbose 10 --min-epochs 0 --epochs 5 --weight-init glorot_uniform --env-name SpaceInvaders-v0 --parallel-sim 4 --optimizer sgd --model toy_rl_convnet16 --num-steps 32

travis-a2c2: clean
	COVERAGE_FILE=.coverage.a2c2 coverage run olympus/baselines/launch.py a2c --verbose 10 --min-epochs 0 --epochs 10 --weight-init glorot_uniform --env-name chaser --parallel-sim 4 --optimizer sgd --model toy_rl_convnet16 --num-steps 32

test-parallel: clean
	olympus --workers 6 --device-sharing classification --batch-size 32 --min-epochs 0 --epochs 300 --dataset test-mnist --model logreg --orion-database legacy:pickleddb:rc_check.pkl

tests: clean
	python -m pytest --cov=olympus tests/unit
	python -m pytest --cov-append --cov=olympus tests/integration

check: clean
	olympus classification --batch-size 32 --min-epochs 0 --epochs 10 --dataset mnist --model logreg --orion-database legacy:pickleddb:test.pkl

run-hpo: clean
	python examples/hpo_simple.py

run-hpo-complete: clean
	python examples/hpo_complete.py  --dataset mnist --model logreg --batch-size 32 --min-epochs 0 --epochs 10

run-hpo-long: clean
	python examples/hpo_simple.py --min-epochs 0 --epochs 100

run-parallel-collab: clean
	./tests/parallel_collaboration.sh

run-parallel-experiments: clean
	./tests/parallel_experiments.sh

run-parallel-mongodb: clean
	./tests/parallel_mongod.sh

rm-doc:
	rm -rf docs/api
	rm -rf _build

build-doc:
	# sphinx-apidoc -e -o docs/api olympus
	sphinx-build -W --color -c docs/ -b html docs/ _build/html

serve-doc:
	sphinx-serve

update-doc: build-doc serve-doc

yolo: rm-doc build-doc serve-doc

kill-zombies:
	ps | grep olympus | awk '{print $$1}' | paste -s -d ' '
	bash -c "kill -9 $$(ps -e | grep olympus | awk '{print $$1}' | paste -s -d ' ')" | true
	ps | grep make | awk '{print $$1}' | paste -s -d ' '
	bash -c "kill -9 $$(ps -e | grep make | awk '{print $$1}' | paste -s -d ' ')" | true

clean:
	rm -rf /tmp/olympus/a2c | true
	rm -rf /tmp/olympus/classification | true
	rm -rf /tmp/olympus/detection | true
	rm -rf /tmp/olympus/*.json | true
	rm -rf /tmp/olympus/*.lock | true
	rm -rf /tmp/olympus/*.pkl | true
	rm -rf *.pkl *.lock | true
	mkdir -p /tmp/olympus/
