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

run-examples:
	python examples/minimalist.py

check:
	rm test.pkl test.pkl.lock | echo ''
	olympus classification --batch-size 32 --epochs 10 --dataset mnist --model logreg

run-hpo:
	rm test.pkl test.pkl.lock | echo ''
	python examples/hpo_simple.py

run-hpo-complete:
	rm test.pkl test.pkl.lock | echo ''
	python examples/hpo_complete.py  --dataset mnist --model logreg --batch-size 32 --epochs 10

run-hpo-long:
	rm -rf /tmp/classification
	rm test.pkl test.pkl.lock | echo ''
	python examples/hpo_simple.py --epochs 100
