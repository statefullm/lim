LLAMA_DIR := llama
LLAMA_BUILD_DIR ?= $(LLAMA_DIR)/build

CXX = g++
# Auto-detect CUDA architecture if not overridden by environment
CUDA_ARCH_FLAGS ?=
ifeq ($(CUDA_ARCH_FLAGS),)
  # Try to detect GPU compute capability from nvidia-smi
  NV_CAPS := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ')

  ifneq ($(NV_CAPS),)
    # Build architecture list for all unique GPUs
    CUDA_ARCH_FLAGS := $(shell echo "$(NV_CAPS)" | while read cap; do \
      major=$$(echo $$cap | cut -d. -f1); \
      minor=$$(echo $$cap | cut -d. -f2); \
      arch=$${major}$${minor}; \
      if [ "$$minor" = "0" ]; then echo "$${arch}a"; else echo "$$arch"; fi; \
    done | sort -u | paste -sd';')
  endif
endif

ifeq ($(CUDA_ARCH_FLAGS),)
  # No GPU detected and no override — will be checked at build time if needed
endif

# Determine GGML backend flags
GGML_CUDA_FLAG ?=
GGML_HIPBLAS_FLAG ?=

ifeq ($(GGML_CUDA),off)
  GGML_CUDA_FLAG := -DGGML_CUDA=OFF
else ifneq ($(NV_CAPS),)
  GGML_CUDA_FLAG := -DGGML_CUDA=ON
endif

ifeq ($(GGML_HIPBLAS),on)
  GGML_HIPBLAS_FLAG := -DGGML_HIPBLAS=ON
endif

CXXFLAGS = -std=c++17 -O3 \
	-I$(LLAMA_DIR)/include \
	-I$(LLAMA_DIR)/common \
	-I$(LLAMA_DIR)/ggml/include \
	-I$(LLAMA_DIR)/vendor \
	-I/usr/include/libxml2

LDFLAGS = -L$(LLAMA_BUILD_DIR)/bin \
	-L/usr/local/cuda-13.0/targets/x86_64-linux/lib \
	-Wl,-rpath,$(shell readlink -f $(LLAMA_BUILD_DIR)/bin) \
	-Wl,-rpath,/usr/local/cuda-13.0/targets/x86_64-linux/lib \
	-lllama -lggml-base -lggml -lggml-cpu -lggml-cuda \
	-lcudart -l:libllama-common.so -lreadline -lcurl -lxml2 -lcrypto

MAKEDEPEND = $(CXXFLAGS) -O0 -M -MG -DDEPEND

FILES = output server tools filesystem network parsers signals model session token_generator tool_executor session_utils

TARGET = lllm

# Default target: build llama.cpp first, then lllm
all: $(TARGET) vscode

.PHONY: all clean llama-clean FORCE

$(LLAMA_BUILD_DIR):
	mkdir -p $@

ifeq ($(LLAMA_BUILD_DIR),$(LLAMA_DIR)/build)
# Auto-build llama.cpp from subrepo
$(LLAMA_BUILD_DIR)/bin/libllama.so $(LLAMA_BUILD_DIR)/bin/libllama-common.so: $(LLAMA_DIR)/CMakeLists.txt | $(LLAMA_BUILD_DIR)
	@if [ -z "$(CUDA_ARCH_FLAGS)" ]; then echo "Error: No GPU detected and CUDA_ARCH_FLAGS not set. Set it manually or build with GGML_CUDA=off for CPU-only." >&2; exit 1; fi
	@echo "[llama.cpp] Configuring with CUDA architectures: $(CUDA_ARCH_FLAGS)"
	cd $(LLAMA_DIR) && cmake -B $(abspath $(LLAMA_BUILD_DIR)) \
		-DCMAKE_CUDA_ARCHITECTURES="$(CUDA_ARCH_FLAGS)" \
		$(GGML_CUDA_FLAG) \
		$(GGML_HIPBLAS_FLAG) \
		$(LLAMA_CMAKE_FLAGS)
	@echo "[llama.cpp] Building..."
	cd $(LLAMA_DIR) && cmake --build $(abspath $(LLAMA_BUILD_DIR)) --target llama --target llama-common

$(TARGET): main.o $(FILES:=.o) $(FILES:=.h) | $(LLAMA_BUILD_DIR)/bin/libllama.so $(LLAMA_BUILD_DIR)/bin/libllama-common.so
	$(CXX) $(CXXFLAGS) main.o $(FILES:=.o) -o $(TARGET) $(LDFLAGS)
else
# Use pre-built llama.cpp from external directory
$(TARGET): main.o $(FILES:=.o) $(FILES:=.h)
	$(CXX) $(CXXFLAGS) main.o $(FILES:=.o) -o $(TARGET) $(LDFLAGS)
endif

VSIX = vscode-extension/vscode-extension-0.1.0.vsix

vscode: $(VSIX)

$(VSIX): vscode-extension/src/extension.ts vscode-extension/package.json vscode-extension/tsconfig.json
	cd vscode-extension && npm install --no-bin-links && node_modules/typescript/bin/tsc -p ./ && npx @vscode/vsce package

install: vscode
	code --install-extension $(VSIX)

vscode-uninstall: vscode FORCE
	code --uninstall-extension undefined_publisher.vscode-extension

uninstall: vscode-uninstall

clean:	FORCE
	rm -f $(TARGET) *.o *.d
	rm -rf vscode-extension/out vscode-extension/node_modules vscode-extension/*.vsix

llama-clean:
	rm -rf $(LLAMA_BUILD_DIR)

distclean: clean llama-clean

.SUFFIXES: .cc .o .d
%.o: %.cc $(FILES:=.cc)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

%.d: %.cc
	@echo Creating $@; \
	rm -f $@; \
	${CXX} $(MAKEDEPEND) $< > $@.$$$$ 2>/dev/null && \
	sed 's,\($*\)\.o[ :]*,\1.o \1.pic.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

ifneq ($(filter clean install,${MAKECMDGOALS}),)
SKIP_DEPS = 1
endif

ifndef SKIP_DEPS
-include $(FILES:=.d)
endif
