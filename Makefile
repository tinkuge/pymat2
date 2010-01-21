# This makefile is only for mine conviniance

OS := $(shell eval 'uname | tr "[:upper:]" "[:lower:]" | tr -c -d "[:lower:]"')


all:
	echo $(OS)
ifeq ($(OS), cygwinnt)
	echo 1
	$(MAKE) windist
else
	echo 2
	$(MAKE) unixdist
endif

test: build
	cp -f test.py ./build/lib*/
	python ./build/lib*/test.py
	rm ./build/lib*/test.py

windist: test
	python setup.py bdist
	python setup.py bdist_wininst

unixdist: test
	python setup.py bdist

build:
	python setup.py build
