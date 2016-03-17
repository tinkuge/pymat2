This page describes what I had to do to make pymat2 compile and run on current
Debian unstable.

It is likely that you will have to do something similar for other Linuxes

First of all, you need to register Matlab shared libraries on your system,
on Debian it is done via **executing as root**
```
$ echo "/usr/share/matlab/bin/glnxa64/" >> /etc/ld.so.conf.d/matlab.conf
$ ldconfig
```

calling `ldconfig` produced lots of output like
```
ldconfig: /usr/share/matlab/bin/glnxa64/libmwmclmcrrt.so.7.9.csf is not an ELF file - it has the wrong magic bytes at the start.

ldconfig: /usr/share/matlab/bin/glnxa64/libmwiqm.so.csf is not an ELF file - it has the wrong magic bytes at the start.
```

I've ignored that.

Please note that without given modifications in `ld`, the pymat2 cannot be executed at all.

Also, install `csh` shell on system, because this is what matlab library uses to start matlab engine when `engOpen` command is called (one that is called to start matlab engine on `.start()` call).

Please note that if you won't provide constructor args for `pymat2.Matlab` object, than
`engOpen` will simply execute `matlab` command using `csh`.