LLAMA_ROOT = ../llama.cpp

CXX = g++
CXXFLAGS = -std=c++17 -O3 -I$(LLAMA_ROOT)/include -I$(LLAMA_ROOT)/common -I$(LLAMA_ROOT)/ggml/include -I$(LLAMA_ROOT)/vendor -I/usr/include/libxml2
LDFLAGS =  -L$(LLAMA_ROOT)/build/bin -L/usr/local/cuda-13.0/targets/x86_64-linux/lib -Wl,-rpath,$(shell readlink -f $(LLAMA_ROOT)/build/bin) -Wl,-rpath,/usr/local/cuda-13.0/targets/x86_64-linux/lib -lllama -lggml-base -lggml -lggml-cpu -lggml-cuda -lcudart -lllama-common -lreadline -lcurl -lxml2 -lcrypto
MAKEDEPEND = $(CXXFLAGS) -O0 -M -MG -DDEPEND

FILES = output server tools filesystem network parsers signals model session token_generator tool_executor session_utils

TARGET = lllm

$(TARGET): main.o $(FILES:=.o) $(FILES:=.h)
	$(CXX) $(CXXFLAGS) main.o $(FILES:=.o) -o $(TARGET) $(LDFLAGS)

all: $(TARGET) vscode

vscode: FORCE
	cd vscode-extension && npm install && npx tsc -p ./ && npx @vscode/vsce package --allow-missing-repository

install: FORCE
	code --install-extension vscode-extension/vscode-extension-*.vsix

vscode-uninstall: FORCE
	code --uninstall-extension undefined_publisher.vscode-extension

uninstall: vscode-uninstall

clean:	FORCE
	rm -f $(TARGET) *.o *.d
	rm -rf vscode-extension/out vscode-extension/node_modules vscode-extension/*.vsix

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
