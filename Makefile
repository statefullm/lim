LLAMA_ROOT = ../llama.cpp

CXX = g++
CXXFLAGS = -std=c++17 -O3 -I$(LLAMA_ROOT)/include -I$(LLAMA_ROOT)/common -I$(LLAMA_ROOT)/ggml/include -I$(LLAMA_ROOT)/vendor -I/usr/include/libxml2
LDFLAGS =  -L$(LLAMA_ROOT)/build/bin -L$(LLAMA_ROOT)/build/common -L/usr/local/cuda-13.0/targets/x86_64-linux/lib -lllama -lggml-base -lggml -lcuda -lcudart $(LLAMA_ROOT)/build/common/libcommon.a -lreadline -lcurl -lxml2
MAKEDEPEND = $(CXXFLAGS) -O0 -M -MG -DDEPEND

FILES = filesystem network parsers

TARGET = lllm

$(TARGET): main.o $(FILES:=.o) $(FILES:=.h)
	$(CXX) $(CXXFLAGS) main.o $(FILES:=.o) -o $(TARGET) $(LDFLAGS)

clean:	FORCE
	rm -f $(TARGET) *.o *.d

.SUFFIXES: .cc .o .d
%.o: %.cc $(FILES:=.cc)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

%.d: %.cc
	@echo Creating $@; \
	rm -f $@; \
	${CXX} $(MAKEDEPEND) $< > $@.$$$$ 2>/dev/null && \
	sed 's,\($*\)\.o[ :]*,\1.o \1.pic.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

ifeq (,$(findstring clean,${MAKECMDGOALS}))
-include $(FILES:=.d)
endif

FORCE:
