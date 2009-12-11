include env.mk

pymat2:
	$(MAKE) -C src pymat2

test: pymat2
	$(PYTHON) test.py


dist: clean
	$(MAKE) pymat2
	$(MAKE) test
	mkdir -p ./dist/pymat2
	cp ./src/*.{py,pyd} ./dist/pymat2
	cd ./dist && (zip -r ./pymat2_py26.zip ./pymat2)

clean:
	$(MAKE) -C src clean
	if [ -e ./dist ] ; then rm -rf ./dist; fi

all: pymat
