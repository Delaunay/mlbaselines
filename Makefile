export OLYMPUS_DATA_PATH=/tmp

travis: travis-doc travis-minimalist travis-hpo_simple travis-classification travis-classification-fp16 travis-unit travis-custom travis-end

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
	COVERAGE_FILE=.coverage.classify coverage run --parallel-mode olympus/baselines/launch.py classification --batch-size 32 --epochs 5 --dataset test-mnist --model logreg

travis-classification-fp16: clean
	COVERAGE_FILE=.coverage.classify_fp16 coverage run --parallel-mode olympus/baselines/launch.py classification --batch-size 32 --epochs 5 --dataset test-mnist --model logreg --half

travis-detection: clean
	COVERAGE_FILE=.coverage.dect coverage run --parallel-mode olympus/baselines/launch.py detection --batch-size 2 --epochs 5 --dataset pennfudan --model fasterrcnn_resnet18_fpn -vv

travis-unit:
	COVERAGE_FILE=.coverage.unit coverage run --parallel-mode -m pytest --cov=olympus tests/unit
	COVERAGE_FILE=.coverage.inte coverage run --parallel-mode -m pytest --cov-append --cov=olympus tests/integration

travis-custom:
	COVERAGE_FILE=.coverage.model coverage run examples/custom_model.py
	COVERAGE_FILE=.coverage.model_nas coverage run examples/custom_model_nas.py
	COVERAGE_FILE=.coverage.optim coverage run examples/custom_optimizer.py
	COVERAGE_FILE=.coverage.schedule coverage run examples/custom_schedule.py

travis-end:
	coverage combine
	coverage report -m
	coverage xml
	codecov

tests: clean
	python -m pytest --cov=olympus tests/unit
	python -m pytest --cov-append --cov=olympus tests/integration

check: clean
	olympus classification --batch-size 32 --epochs 10 --dataset mnist --model logreg --orion-database legacy:pickleddb:test.pkl

run-hpo: clean
	python examples/hpo_simple.py

run-hpo-complete: clean
	python examples/hpo_complete.py  --dataset mnist --model logreg --batch-size 32 --epochs 10

run-hpo-long: clean
	python examples/hpo_simple.py --epochs 100

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
	rm -rf /tmp/classification | true
	rm my_data.pkl my_data.pkl.lock | true
	rm test.pkl test.pkl.lock | true
	rm track_test.json track_test.json.lock | true
