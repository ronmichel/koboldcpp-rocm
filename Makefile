# Add custom options to Makefile.local rather than editing this file.
-include $(abspath $(lastword ${MAKEFILE_LIST})).local

default: koboldcpp_default koboldcpp_hipblas koboldcpp_vulkan
tools: quantize_gpt2 quantize_gptj quantize_gguf quantize_neox quantize_mpt quantize_clip whispermain sdmain gguf-split llama-bench perplexity 
dev: koboldcpp_openblas
dev2: koboldcpp_clblast
dev3: koboldcpp_vulkan

ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

ifneq ($(shell grep -e "Arch Linux" -e "ID_LIKE=arch" /etc/os-release 2>/dev/null),)
ARCH_ADD = -lcblas
endif

#
# Compile flags
#

# keep standard at C11 and C++11
CFLAGS   = -I. -Iggml/include -Iggml/src -Iinclude -Isrc -I./include -I./include/CL -I./otherarch -I./otherarch/tools -I./otherarch/sdcpp -I./otherarch/sdcpp/thirdparty -I./include/vulkan -O3 -fno-finite-math-only -DNDEBUG -std=c11   -fPIC -DLOG_DISABLE_LOGS -D_GNU_SOURCE -DGGML_USE_LLAMAFILE
CXXFLAGS = -I. -Iggml/include -Iggml/src -Iinclude -Isrc -I./common -I./include -I./include/CL -I./otherarch -I./otherarch/tools -I./otherarch/sdcpp -I./otherarch/sdcpp/thirdparty -I./include/vulkan -O3 -fno-finite-math-only -DNDEBUG -std=c++11 -fPIC -DLOG_DISABLE_LOGS -D_GNU_SOURCE -DGGML_USE_LLAMAFILE
LDFLAGS  =
#CC         := gcc-13
#CXX        := g++-13
FASTCFLAGS = $(subst -O3,-Ofast,$(CFLAGS))
FASTCXXFLAGS = $(subst -O3,-Ofast,$(CXXFLAGS))

# these are used on windows, to build some libraries with extra old device compatibility
SIMPLECFLAGS =
FULLCFLAGS =
NONECFLAGS =

VULKAN_FLAGS = -DGGML_USE_VULKAN

OBJS_FULL += ggml-alloc.o ggml-aarch64.o ggml-quants.o unicode.o unicode-data.o sgemm.o common.o sampling.o grammar-parser.o
OBJS_SIMPLE += ggml-alloc.o ggml-aarch64.o ggml-quants_noavx2.o unicode.o unicode-data.o sgemm_noavx2.o common.o sampling.o grammar-parser.o
OBJS_FAILSAFE += ggml-alloc.o ggml-aarch64.o ggml-quants_failsafe.o unicode.o unicode-data.o sgemm_failsafe.o common.o sampling.o grammar-parser.o

#lets try enabling everything
CFLAGS   += -pthread -s -Wno-deprecated -Wno-deprecated-declarations
CXXFLAGS += -pthread -s -Wno-multichar -Wno-write-strings -Wno-deprecated -Wno-deprecated-declarations

# OS specific
ifeq ($(UNAME_S),Linux)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif

ifdef LLAMA_GPROF
	CFLAGS   += -pg
	CXXFLAGS += -pg
endif
ifdef LLAMA_PERF
	CFLAGS   += -DGGML_PERF
	CXXFLAGS += -DGGML_PERF
endif

CFLAGS += -mavx2 -msse3 -mfma -mf16c -mavx


OBJS_CUDA_TEMP_INST = $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/fattn-wmma*.cu))
OBJS_CUDA_TEMP_INST += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/mmq*.cu))
OBJS_CUDA_TEMP_INST += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu))
OBJS_CUDA_TEMP_INST += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu))
OBJS_CUDA_TEMP_INST += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/fattn-vec*f16-f16.cu))

ifdef LLAMA_HIPBLAS
	ifeq ($(wildcard /opt/rocm),)
		ROCM_PATH	?= /usr
		GPU_TARGETS ?= gfx803 gfx900 gfx906 gfx908 gfx90a gfx1010 gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102 $(shell $(shell which amdgpu-arch))
		HCC         := $(ROCM_PATH)/bin/hipcc
		HCXX        := $(ROCM_PATH)/bin/hipcc
	else
		ROCM_PATH	?= /opt/rocm
		HCC         := $(ROCM_PATH)/llvm/bin/clang
		HCXX        := $(ROCM_PATH)/llvm/bin/clang++
		GPU_TARGETS ?= gfx803 gfx900 gfx906 gfx908 gfx90a gfx1010 gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102 $(shell $(ROCM_PATH)/llvm/bin/amdgpu-arch)
	endif
	LLAMA_CUDA_DMMV_X ?= 32
	LLAMA_CUDA_MMV_Y ?= 1
	LLAMA_CUDA_KQUANTS_ITER ?= 2
	HIPFLAGS   += -DGGML_USE_HIPBLAS -DGGML_USE_CUDA -DSD_USE_CUBLAS $(shell $(ROCM_PATH)/bin/hipconfig -C)
	HIPLDFLAGS    += -L$(ROCM_PATH)/lib -Wl,-rpath=$(ROCM_PATH)/lib -lhipblas -lamdhip64 -lrocblas
	HIP_OBJS      += ggml-cuda.o ggml_v3-cuda.o ggml_v2-cuda.o ggml_v2-cuda-legacy.o
	HIP_OBJS      += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/*.cu))
	HIP_OBJS      += $(OBJS_CUDA_TEMP_INST)

	HIPFLAGS2    += $(addprefix --offload-arch=,$(GPU_TARGETS))
	HIPFLAGS2    += -DGGML_CUDA_DMMV_X=$(LLAMA_CUDA_DMMV_X)
	HIPFLAGS2    += -DGGML_CUDA_MMV_Y=$(LLAMA_CUDA_MMV_Y)
	HIPFLAGS2    += -DK_QUANTS_PER_ITERATION=$(LLAMA_CUDA_KQUANTS_ITER)

ggml/src/ggml-cuda/%.o: ggml/src/ggml-cuda/%.cu ggml/include/ggml.h ggml/src/ggml-common.h ggml/src/ggml-cuda/common.cuh
	$(HCXX) $(CXXFLAGS) $(HIPFLAGS) $(HIPFLAGS2) -x hip -c -o $@ $<
ggml-cuda.o: ggml/src/ggml-cuda.cu ggml/include/ggml-cuda.h ggml/include/ggml.h ggml/include/ggml-backend.h ggml/src/ggml-backend-impl.h ggml/src/ggml-common.h $(wildcard ggml/src/ggml-cuda/*.cuh)
	$(HCXX) $(CXXFLAGS) $(HIPFLAGS) $(HIPFLAGS2) -x hip -c -o $@ $<
ggml_v2-cuda.o: otherarch/ggml_v2-cuda.cu otherarch/ggml_v2-cuda.h
	$(HCXX) $(CXXFLAGS) $(HIPFLAGS) $(HIPFLAGS2) -x hip -c -o $@ $<
ggml_v2-cuda-legacy.o: otherarch/ggml_v2-cuda-legacy.cu otherarch/ggml_v2-cuda-legacy.h
	$(HCXX) $(CXXFLAGS) $(HIPFLAGS) $(HIPFLAGS2) -x hip -c -o $@ $<
ggml_v3-cuda.o: otherarch/ggml_v3-cuda.cu otherarch/ggml_v3-cuda.h
	$(HCXX) $(CXXFLAGS) $(HIPFLAGS) $(HIPFLAGS2) -x hip -c -o $@ $<
endif # LLAMA_HIPBLAS


DEFAULT_BUILD =
HIPBLAS_BUILD =
VULKAN_BUILD =

	DEFAULT_BUILD = $(CXX) $(CXXFLAGS) $^ -shared -o $@.so $(LDFLAGS)

	ifdef LLAMA_HIPBLAS
		HIPBLAS_BUILD = $(HCXX) $(CXXFLAGS) $(HIPFLAGS) $^ -shared -o $@.so $(HIPLDFLAGS) $(LDFLAGS)
	endif
	ifdef LLAMA_VULKAN
		VULKAN_BUILD = $(CXX) $(CXXFLAGS) $^ -lvulkan -shared -o $@.so $(LDFLAGS)
	endif

	ifndef LLAMA_HIPBLAS
	ifndef LLAMA_VULKAN
	OPENBLAS_BUILD = @echo 'Your OS $(OS) does not appear to be Windows. For faster speeds, install and link a BLAS library. Set LLAMA_OPENBLAS=1 to compile with OpenBLAS support or LLAMA_CLBLAST=1 to compile with ClBlast support. This is just a reminder, not an error.'
	endif
	endif

CCV := $(shell $(CC) --version | head -n 1)
CXXV := $(shell $(CXX) --version | head -n 1)

#
# Print build information
#

$(info I llama.cpp build info: )
$(info I UNAME_S:  $(UNAME_S))
$(info I UNAME_P:  $(UNAME_P))
$(info I UNAME_M:  $(UNAME_M))
$(info I CFLAGS:   $(CFLAGS))
$(info I CXXFLAGS: $(CXXFLAGS))
$(info I LDFLAGS:  $(LDFLAGS))
$(info I CC:       $(CCV))
$(info I CXX:      $(CXXV))
$(info )

#
# Build library
#

ggml.o: ggml/src/ggml.c ggml/include/ggml.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml_v4_cublas.o: ggml/src/ggml.c ggml/include/ggml.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) $(CUBLAS_FLAGS) $(HIPFLAGS) -c $< -o $@
ggml_v4_vulkan.o: ggml/src/ggml.c ggml/include/ggml.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) $(VULKAN_FLAGS) -c $< -o $@
ggml_v4_vulkan_noavx2.o: ggml/src/ggml.c ggml/include/ggml.h
	$(CC)  $(FASTCFLAGS) $(SIMPLECFLAGS) $(VULKAN_FLAGS) -c $< -o $@

#quants
ggml-quants.o: ggml/src/ggml-quants.c ggml/include/ggml.h ggml/src/ggml-quants.h ggml/src/ggml-common.h
	$(CC)  $(CFLAGS) $(FULLCFLAGS) -c $< -o $@

#sgemm
sgemm.o: ggml/src/llamafile/sgemm.cpp ggml/src/llamafile/sgemm.h ggml/include/ggml.h
	$(CXX) $(CXXFLAGS) $(FULLCFLAGS) -c $< -o $@

#there's no intrinsics or special gpu ops used here, so we can have a universal object
ggml-alloc.o: ggml/src/ggml-alloc.c ggml/include/ggml.h ggml/include/ggml-alloc.h
	$(CC)  $(CFLAGS) -c $< -o $@
llava.o: examples/llava/llava.cpp examples/llava/llava.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
unicode.o: src/unicode.cpp src/unicode.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
unicode-data.o: src/unicode-data.cpp src/unicode-data.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
ggml-aarch64.o: ggml/src/ggml-aarch64.c ggml/include/ggml.h ggml/src/ggml-aarch64.h ggml/src/ggml-common.h
	$(CC)  $(CFLAGS) -c $< -o $@

#these have special gpu defines
ggml-backend_default.o: ggml/src/ggml-backend.c ggml/include/ggml.h ggml/include/ggml-backend.h
	$(CC)  $(CFLAGS) -c $< -o $@
ggml-backend_vulkan.o: ggml/src/ggml-backend.c ggml/include/ggml.h ggml/include/ggml-backend.h
	$(CC)  $(CFLAGS) $(VULKAN_FLAGS) -c $< -o $@
ggml-backend_cublas.o: ggml/src/ggml-backend.c ggml/include/ggml.h ggml/include/ggml-backend.h
	$(CC)  $(CFLAGS) $(CUBLAS_FLAGS) -c $< -o $@
llavaclip_default.o: examples/llava/clip.cpp examples/llava/clip.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
llavaclip_cublas.o: examples/llava/clip.cpp examples/llava/clip.h
	$(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) -c $< -o $@

#this is only used for openblas and accelerate
ggml-blas.o: ggml/src/ggml-blas.cpp ggml/include/ggml-blas.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

#version 3 libs
ggml_v3.o: otherarch/ggml_v3.c otherarch/ggml_v3.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml_v3_cublas.o: otherarch/ggml_v3.c otherarch/ggml_v3.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) $(CUBLAS_FLAGS) $(HIPFLAGS) -c $< -o $@

#version 2 libs
ggml_v2.o: otherarch/ggml_v2.c otherarch/ggml_v2.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml_v2_cublas.o: otherarch/ggml_v2.c otherarch/ggml_v2.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) $(CUBLAS_FLAGS) $(HIPFLAGS) -c $< -o $@

#extreme old version compat
ggml_v1.o: otherarch/ggml_v1.c otherarch/ggml_v1.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) -c $< -o $@

#vulkan
ggml-vulkan.o: ggml/src/ggml-vulkan.cpp ggml/include/ggml-vulkan.h ggml/src/ggml-vulkan-shaders.hpp ggml/src/ggml-vulkan-shaders.cpp
	$(CXX) $(CXXFLAGS) $(VULKAN_FLAGS) -c $< -o $@

# intermediate objects
llama.o: src/llama.cpp ggml/include/ggml.h ggml/include/ggml-alloc.h ggml/include/ggml-backend.h ggml/include/ggml-cuda.h ggml/include/ggml-metal.h include/llama.h otherarch/llama-util.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
common.o: common/common.cpp common/common.h common/log.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
sampling.o: common/sampling.cpp common/common.h common/sampling.h common/log.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
console.o: common/console.cpp common/console.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
grammar-parser.o: common/grammar-parser.cpp common/grammar-parser.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
expose.o: expose.cpp expose.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# sd.cpp objects
sdcpp_default.o: otherarch/sdcpp/sdtype_adapter.cpp otherarch/sdcpp/stable-diffusion.h otherarch/sdcpp/stable-diffusion.cpp otherarch/sdcpp/util.cpp otherarch/sdcpp/upscaler.cpp otherarch/sdcpp/model.cpp otherarch/sdcpp/thirdparty/zip.c
	$(CXX) $(CXXFLAGS) -c $< -o $@
sdcpp_cublas.o: otherarch/sdcpp/sdtype_adapter.cpp otherarch/sdcpp/stable-diffusion.h otherarch/sdcpp/stable-diffusion.cpp otherarch/sdcpp/util.cpp otherarch/sdcpp/upscaler.cpp otherarch/sdcpp/model.cpp otherarch/sdcpp/thirdparty/zip.c
	$(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) $(HIPFLAGS) -c $< -o $@

#whisper objects
whispercpp_default.o: otherarch/whispercpp/whisper_adapter.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
whispercpp_cublas.o: otherarch/whispercpp/whisper_adapter.cpp
	$(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) $(HIPFLAGS) -c $< -o $@

# idiotic "for easier compilation"
GPTTYPE_ADAPTER = gpttype_adapter.cpp otherarch/llama_v2.cpp otherarch/llama_v3.cpp src/llama.cpp src/llama-grammar.cpp src/llama-sampling.cpp src/llama-vocab.cpp otherarch/utils.cpp otherarch/gptj_v1.cpp otherarch/gptj_v2.cpp otherarch/gptj_v3.cpp otherarch/gpt2_v1.cpp otherarch/gpt2_v2.cpp otherarch/gpt2_v3.cpp otherarch/rwkv_v2.cpp otherarch/rwkv_v3.cpp otherarch/neox_v2.cpp otherarch/neox_v3.cpp otherarch/mpt_v3.cpp ggml/include/ggml.h ggml/include/ggml-cuda.h include/llama.h otherarch/llama-util.h
gpttype_adapter.o: $(GPTTYPE_ADAPTER)
	$(CXX) $(CXXFLAGS) -c $< -o $@
gpttype_adapter_cublas.o: $(GPTTYPE_ADAPTER)
	$(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) $(HIPFLAGS) -c $< -o $@
gpttype_adapter_vulkan.o: $(GPTTYPE_ADAPTER)
	$(CXX) $(CXXFLAGS) $(VULKAN_FLAGS) -c $< -o $@

clean:
	rm -vf *.o main sdmain whispermain quantize_gguf quantize_clip quantize_gpt2 quantize_gptj quantize_neox quantize_mpt quantize-stats perplexity embedding benchmark-matmult save-load-state gguf imatrix vulkan-shaders-gen gguf-split gguf-split.exe vulkan-shaders-gen.exe imatrix.exe gguf.exe main.exe sdmain.exe whispermain.exe quantize_clip.exe quantize_gguf.exe quantize_gptj.exe quantize_gpt2.exe quantize_neox.exe quantize_mpt.exe koboldcpp_default.dll koboldcpp_openblas.dll koboldcpp_failsafe.dll koboldcpp_noavx2.dll koboldcpp_clblast.dll koboldcpp_clblast_noavx2.dll koboldcpp_cublas.dll koboldcpp_hipblas.dll koboldcpp_vulkan.dll koboldcpp_vulkan_noavx2.dll koboldcpp_default.so koboldcpp_openblas.so koboldcpp_failsafe.so koboldcpp_noavx2.so koboldcpp_clblast.so koboldcpp_clblast_noavx2.so koboldcpp_cublas.so koboldcpp_hipblas.so koboldcpp_vulkan.so koboldcpp_vulkan_noavx2.so
	rm -vrf ggml/src/ggml-cuda/*.o
	rm -vrf ggml/src/ggml-cuda/template-instances/*.o

# useful tools
main: examples/main/main.cpp build-info.h ggml.o llama.o console.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
	@echo '====  Run ./main -h for help.  ===='
sdmain: otherarch/sdcpp/util.cpp otherarch/sdcpp/main.cpp otherarch/sdcpp/stable-diffusion.cpp otherarch/sdcpp/upscaler.cpp otherarch/sdcpp/model.cpp otherarch/sdcpp/thirdparty/zip.c build-info.h ggml.o llama.o console.o ggml-backend_default.o $(OBJS_FULL) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
whispermain: otherarch/whispercpp/main.cpp otherarch/whispercpp/whisper.cpp build-info.h ggml.o llama.o console.o ggml-backend_default.o $(OBJS_FULL) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
imatrix: examples/imatrix/imatrix.cpp build-info.h ggml.o llama.o console.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
gguf: examples/gguf/gguf.cpp build-info.h ggml.o llama.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
gguf-split: examples/gguf-split/gguf-split.cpp ggml.o llama.o build-info.h llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

vulkan-shaders-gen: ggml/src/vulkan-shaders/vulkan-shaders-gen.cpp
	@echo 'This command can be MANUALLY run to regenerate vulkan shaders. Normally concedo will do it, so you do not have to.'
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
	@echo 'Now rebuilding vulkan shaders...'
	$(shell) vulkan-shaders-gen --glslc glslc --input-dir ggml/src/vulkan-shaders --target-hpp ggml/src/ggml-vulkan-shaders.hpp --target-cpp ggml/src/ggml-vulkan-shaders.cpp

#generated libraries
koboldcpp_default: ggml.o ggml_v3.o ggml_v2.o ggml_v1.o expose.o gpttype_adapter.o sdcpp_default.o whispercpp_default.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL) $(OBJS)
	$(DEFAULT_BUILD)

ifdef HIPBLAS_BUILD
koboldcpp_hipblas: ggml_v4_cublas.o ggml_v3_cublas.o ggml_v2_cublas.o ggml_v1.o expose.o gpttype_adapter_cublas.o sdcpp_cublas.o whispercpp_cublas.o llavaclip_cublas.o llava.o ggml-backend_cublas.o $(HIP_OBJS) $(OBJS_FULL) $(OBJS)
	$(HIPBLAS_BUILD)
else
koboldcpp_hipblas:
	$(DONOTHING)
endif

ifdef VULKAN_BUILD
koboldcpp_vulkan: ggml_v4_vulkan.o ggml_v3.o ggml_v2.o ggml_v1.o expose.o gpttype_adapter_vulkan.o ggml-vulkan.o sdcpp_default.o whispercpp_default.o llavaclip_default.o llava.o ggml-backend_vulkan.o $(OBJS_FULL) $(OBJS)
	$(VULKAN_BUILD)
else
koboldcpp_vulkan:
	$(DONOTHING)
endif

# tools
quantize_gguf: examples/quantize/quantize.cpp ggml.o llama.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
quantize_gptj: otherarch/tools/gptj_quantize.cpp otherarch/tools/common-ggml.cpp ggml_v3.o ggml.o llama.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
quantize_gpt2: otherarch/tools/gpt2_quantize.cpp otherarch/tools/common-ggml.cpp ggml_v3.o ggml.o llama.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
quantize_neox: otherarch/tools/neox_quantize.cpp otherarch/tools/common-ggml.cpp ggml_v3.o ggml.o llama.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
quantize_mpt: otherarch/tools/mpt_quantize.cpp otherarch/tools/common-ggml.cpp ggml_v3.o ggml.o llama.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
quantize_clip: examples/llava/clip.cpp examples/llava/clip.h examples/llava/quantclip.cpp ggml_v3.o ggml.o llama.o ggml-backend_default.o $(OBJS_FULL)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
perplexity: examples/perplexity/perplexity.cpp                build-info.h ggml_cublas.o ggml_v2_cublas.o ggml_v1.o expose.o common.o gpttype_adapter_cublas.o k_quants.o ggml-alloc.o $(CUBLAS_OBJS) $(HIP_OBJS) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(HIPLDFLAGS)
llama-bench: examples/llama-bench/llama-bench.cpp build-info.h ggml_cublas.o ggml_v2_cublas.o ggml_v1.o expose.o common.o gpttype_adapter_cublas.o k_quants.o ggml-alloc.o $(CUBLAS_OBJS) $(HIP_OBJS) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(HIPLDFLAGS)

#window simple clinfo
simpleclinfo: simpleclinfo.cpp
	$(CXX) $(CXXFLAGS) $^ lib/OpenCL.lib lib/clblast.lib -o $@ $(LDFLAGS)

build-info.h:
	$(DONOTHING)
