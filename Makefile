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
	cp setup.py ./dist
	zip -r ./dist/pymat2_$(PYMAT2_VERSION)_win32_py26.zip \
		./dist/pymat2 ./dist/setup.py

windist: dist
	cd ./dist && (\
		$(PYTHON) setup.py bdist_wininst && \
		mv ./dist/*.exe . && \
		for fname in $$(ls *.exe) ; do \
			mv \
				$${fname} \
				$$(echo $${fname} | sed 's/\(.exe\)/_py26\1/g'); \
		done ; \
		rm -r ./dist \
	)

clean:
	$(MAKE) -C src clean
	if [ -e ./dist ] ; then rm -rf ./dist; fi

release_packages:
ifeq ($(OS),cygwinnt)
	$(MAKE) windist
else
	# TODO: add other OSes
endif

all: pymat2
