

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

check:
	rm test.pkl test.pkl.lock | echo ''
	olympus classification --batch-size 32 --epochs 10 --dataset mnist --model resnet18
