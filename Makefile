OBJS=RPIGears.o
BIN=RPIGears

CFLAGS    +=-DSTANDALONE -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -DTARGET_POSIX -D_LINUX -fPIC -DPIC -D_REENTRANT \
	     -DUSE_EXTERNAL_OMX -DHAVE_LIBBCM_HOST -DUSE_EXTERNAL_LIBBCM_HOST -DUSE_VCHIQ_ARM \
	     -DHAVE_LIBOPENMAX=2 -DOMX -DOMX_SKIP64BIT -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 \
	     -U_FORTIFY_SOURCE \
	     -g -Wall -Wno-psabi -ftree-vectorize -pipe

LDFLAGS   += -L$(SDKSTAGE)/opt/vc/lib/  \
	     -lpthread -lrt -lm -ldl \
	     -lbcm_host -lbrcmGLESv2 -lbrcmEGL -lvcos -lvchiq_arm -lopenmaxil

INCLUDES  += -I$(SDKSTAGE)/opt/vc/include/ \
	     -I$(SDKSTAGE)/opt/vc/include/interface/vcos/pthreads \
	     -I$(SDKSTAGE)/opt/vc/include/interface/vmcs_host/linux

all: $(BIN) $(LIB)

%.o: %.c
	@rm -f $@
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@ -Wno-deprecated-declarations

%.o: %.cpp
	@rm -f $@
	$(CXX) $(CFLAGS) $(INCLUDES) -std=c++0x -c $< -o $@ -Wno-deprecated-declarations

RPIgears: $(OBJS)
	$(CC) -o $@ -Wl,--whole-archive $(OBJS) $(LDFLAGS) -Wl,--no-whole-archive -rdynamic

.PHONY: clean rebuild

rebuild:
	make clean && make

clean:
	@rm -f *.o
	@rm -f RPIgears

ifndef LOGNAME
SDKSTAGE  = /SysGCC/Raspberry/arm-linux-gnueabihf/sysroot
endif
CC      := arm-linux-gnueabihf-gcc
CXX     := arm-linux-gnueabihf-g++
