export OLYMPUS_DATA_PATH=/tmp

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
	# bash -c "ps | grep olympus | awk '{print $$1}'"
	bash -c "kill -9 $$(ps | grep make | awk '{print $$1}' | paste -s -d ' ')"

travis-doc: build-doc

travis-minimalist:
	python examples/minimalist.py

travis-hpo_simple:
	rm test.pkl test.pkl.lock | echo ''
	python examples/hpo_simple.py

travis-classification:
	rm test.pkl test.pkl.lock | echo ''
	olympus classification --batch-size 32 --epochs 5 --dataset test-mnist --model logreg

travis-detection:
	rm test.pkl test.pkl.lock | echo ''
	olympus detection --batch-size 2 --epochs 5 --dataset pennfudan --model fasterrcnn_resnet18_fpn -vv

check:
	rm test.pkl test.pkl.lock | echo ''
	olympus classification --batch-size 32 --epochs 10 --dataset mnist --model logreg

run-hpo:
	rm test.pkl test.pkl.lock | echo ''
	python examples/hpo_simple.py

run-hpo-complete:
	rm -rf /tmp/classification
	rm test.pkl test.pkl.lock track_test.json track_test.json.lock | echo ''
	python examples/hpo_complete.py  --dataset mnist --model logreg --batch-size 32 --epochs 10

run-hpo-long:
	rm -rf /tmp/classification
	rm test.pkl test.pkl.lock | echo ''
	python examples/hpo_simple.py --epochs 100
