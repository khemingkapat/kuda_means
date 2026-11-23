# ==============================================================================
# kuda_means — Makefile
# ==============================================================================
# Targets:
#   make cpu                         Build CPU baseline (no CUDA required)
#   make gpu [ARCH=sm_89]            Build CUDA benchmark binary (default: RTX 4090 sm_89)
#   make gpu ARCH=sm_75              Build for Google Colab Tesla T4
#   make clean                       Remove build outputs
# ==============================================================================

CXX       ?= g++
NVCC      ?= nvcc
ARCH      ?= sm_89
NVCC_CCBIN ?=

CXXFLAGS  := -O3 -std=c++17 -Wall -Wextra -Iinclude
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -Iinclude --expt-relaxed-constexpr

ifneq ($(strip $(NVCC_CCBIN)),)
    NVCCFLAGS += -ccbin $(NVCC_CCBIN)
endif

BUILD_DIR := build
BIN_DIR   := bin

COMMON_SRCS := $(wildcard src/common/*.cpp)
CPU_SRCS    := $(wildcard src/cpu/*.cpp)
GPU_SRCS    := $(wildcard src/v1_naive/*.cu src/v2_memory/*.cu src/v3_privatized/*.cu src/v4_matmul/*.cu src/v5_streaming/*.cu src/gpu/*.cu)

.PHONY: all cpu gpu clean dirs info

all: cpu

dirs:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

info:
	@echo "CXX        = $(CXX)"
	@echo "NVCC       = $(NVCC)"
	@echo "ARCH       = $(ARCH)"
	@echo "NVCC_CCBIN = $(NVCC_CCBIN)"
	@echo "COMMON     = $(COMMON_SRCS)"
	@echo "CPU_SRCS   = $(CPU_SRCS)"
	@echo "GPU_SRCS   = $(GPU_SRCS)"

cpu: dirs
	@if [ -z "$(strip $(CPU_SRCS))" ]; then \
		echo "No CPU sources found in src/cpu/ yet."; \
	else \
		$(CXX) $(CXXFLAGS) $(COMMON_SRCS) $(CPU_SRCS) -o $(BIN_DIR)/kuda_means_cpu && \
		echo "Built $(BIN_DIR)/kuda_means_cpu"; \
	fi

gpu: dirs
	@if [ -z "$(strip $(GPU_SRCS))" ]; then \
		echo "No CUDA sources found yet."; \
	else \
		$(NVCC) $(NVCCFLAGS) $(COMMON_SRCS) $(CPU_SRCS) $(GPU_SRCS) -o $(BIN_DIR)/kuda_means_gpu && \
		echo "Built $(BIN_DIR)/kuda_means_gpu (arch=$(ARCH))"; \
	fi

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
