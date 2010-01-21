include ../../env.mk

%.$(OBJ_EXT):	%.c
	$(CC) $<

fix_version:
	sed -e \
		's,\(#define PYMAT_VERSION\).*,\1 "$(PYMAT2_VERSION)",' \
		./pymat_head.h > ./pymat_head_tmp.h
	mv -f ./pymat_head_tmp.h ./pymat_head.h

objects: fix_version
	$(MAKE) $(patsubst %.c,%.$(OBJ_EXT),$(wildcard *.c)) 

_pymat2.pyd: objects
	$(LINK) /DLL /IMPLIB:"_pymat2.lib" /out:$@ *.$(OBJ_EXT)

clean:
	$(RM) *.$(OBJ_EXT) *.pyd *.idb *.exp *.lib
