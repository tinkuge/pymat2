include ../env.mk

_pymat.pyd:
	$(MAKE) -C _pymat _pymat.pyd
	cp ./_pymat/_pymat.pyd .

pymat: _pymat.pyd

clean:
	$(MAKE) -C _pymat clean
	$(RM) *.pyc *.pyd

pymat: _pymat
	@echo "here"
