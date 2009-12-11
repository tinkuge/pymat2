include ../../env.mk

%.$(OBJ_EXT):	%.c
	$(CC) $<

objects:
	$(MAKE) $(patsubst %.c,%.$(OBJ_EXT),$(wildcard *.c)) 

_pymat.pyd: objects
	$(LINK) /DLL /IMPLIB:"_pymat.lib" /out:$@ *.$(OBJ_EXT)

clean:
	$(RM) *.$(OBJ_EXT) *.pyd *.idb *.exp *.lib
