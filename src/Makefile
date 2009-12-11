include ../env.mk

_pymat2.pyd:
	$(MAKE) -C _pymat2 _pymat2.pyd
	cp ./_pymat2/_pymat2.pyd .

pymat2: _pymat2.pyd

clean:
	$(MAKE) -C _pymat2 clean
	$(RM) *.pyc *.pyd
