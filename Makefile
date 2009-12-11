include env.mk

pymat:
	$(MAKE) -C src pymat

test: pymat
	$(PYTHON) test.py


dist: clean
	$(MAKE) pymat
	$(MAKE) test
	mkdir -p ./dist/pymat2
	cp ./src/*.{py,pyd} ./dist/pymat2
	cd ./dist && (zip -r ./pymat2_py26.zip ./pymat2)

clean:
	$(MAKE) -C src clean
	if [ -e ./dist ] ; then rm -rf ./dist; fi

all: pymat
