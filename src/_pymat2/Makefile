include ../../env.mk

%.$(OBJ_EXT):	%.c
	$(CC) $<

objects:
	$(MAKE) $(patsubst %.c,%.$(OBJ_EXT),$(wildcard *.c)) 

_pymat2.pyd: objects
	$(LINK) /DLL /IMPLIB:"_pymat2.lib" /out:$@ *.$(OBJ_EXT)

clean:
	$(RM) *.$(OBJ_EXT) *.pyd *.idb *.exp *.lib
