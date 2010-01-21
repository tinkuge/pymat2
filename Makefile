# This makefile is only for mine conviniance

OS := $(shell eval 'uname | tr "[:upper:]" "[:lower:]" | tr -c -d "[:lower:]"')


all:
ifeq ($(OS), cygwinnt)
	$(MAKE) windist
else
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

build: clean
	python setup.py build

clean:
	if [ -e ./dist ] ; then rm -r ./dist; fi
	if [ -e ./build ] ; then rm -r ./build; fi
