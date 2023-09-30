# koboldcpp-ROCM
This is mostly for Linux, but with the release of ROCm frameworks on Windows, it might eventually be possible to run this on Windows.

To install, either use the file "[easy_KCPP-ROCm_install.sh](https://github.com/YellowRoseCx/koboldcpp-rocm/blob/main/easy_KCPP-ROCm_install.sh)" or navigate to the folder you want to download to in Terminal then run
```
git clone https://github.com/YellowRoseCx/koboldcpp-rocm.git -b main --depth 1 && \
cd koboldcpp-rocm && \
make LLAMA_HIPBLAS=1 -j4 && \
./koboldcpp.py
```
When the KoboldCPP GUI appears, make sure to select "Use CuBLAS/hipBLAS" and set GPU layers 

Original [llama.cpp rocm port](https://github.com/ggerganov/llama.cpp/pull/1087) by SlyEcho, modified and ported to koboldcpp by YellowRoseCx

Comparison with OpenCL using 6800xt
| Model | Offloading Method | Time Taken - Processing 593 tokens| Time Taken - Generating 200 tokens| Total Time | Perf. Diff.
|-----------------|----------------------------|--------------------|--------------------|------------|---|
| Robin 7b q6_K |CLBLAST 6-t, All Layers on GPU | 6.8s (11ms/T) | 12.0s (60ms/T)  | 18.7s (10.7T/s) | 1x
| Robin 7b q6_K |ROCM 1-t, All Layers on GPU   | 1.4s (2ms/T) | 5.5s (28ms/T)   | 6.9s (29.1T/s)| **2.71x**
| Robin 13b q5_K_M |CLBLAST 6-t, All Layers on GPU | 10.9s (18ms/T) | 16.7s (83ms/T)  | 27.6s (7.3T/s) | 1x
| Robin 13b q5_K_M |ROCM 1-t, All Layers on GPU   | 2.4s (4ms/T) | 7.8s (39ms/T)   | 10.2s (19.6T/s)| **2.63x**
| Robin 33b q4_K_S |CLBLAST 6-t, 46/63 Layers on GPU | 23.2s (39ms/T) | 48.6s (243ms/T)  | 71.9s (2.8T/s) | 1x
| Robin 33b q4_K_S |CLBLAST 6-t, 50/63 Layers on GPU | 25.5s (43ms/T) | 44.6s (223ms/T)  | 70.0s (2.9T/s) | 1x
| Robin 33b q4_K_S |ROCM 6-t, 46/63 Layers on GPU   | 14.6s (25ms/T) | 44.1s (221ms/T)   | 58.7s (3.4T/s)| **1.19x**

--------
A self contained distributable from Concedo that exposes llama.cpp function bindings, allowing it to be used via a simulated Kobold API endpoint.

What does it mean? You get llama.cpp with a fancy UI, persistent stories, editing tools, save formats, memory, world info, author's note, characters, scenarios and everything Kobold and Kobold Lite have to offer. In a tiny package around 20 MB in size, excluding model weights.

![Preview](media/preview.png)

## Usage
- **[Download the latest .exe release here](https://github.com/LostRuins/koboldcpp/releases/latest)** or clone the git repo.
- Windows binaries are provided in the form of **koboldcpp.exe**, which is a pyinstaller wrapper for a few **.dll** files and **koboldcpp.py**. If you feel concerned, you may prefer to rebuild it yourself with the provided makefiles and scripts.
- Weights are not included, you can use the official llama.cpp `quantize.exe` to generate them from your official weight files (or download them from other places such as [TheBloke's Huggingface](https://huggingface.co/TheBloke).
- To run, execute **koboldcpp.exe** or drag and drop your quantized `ggml_model.bin` file onto the .exe, and then connect with Kobold or Kobold Lite. If you're not on windows, then run the script **KoboldCpp.py** after compiling the libraries.
- Launching with no command line arguments displays a GUI containing a subset of configurable settings. Generally you dont have to change much besides the `Presets` and `GPU Layers`. Read the `--help` for more info about each settings.
- By default, you can connect to http://localhost:5001
- You can also run it using the command line `koboldcpp.exe [ggml_model.bin] [port]`. For info, please check `koboldcpp.exe --help`
- Default context size to small? Try `--contextsize 3072` to 1.5x your context size! without much perplexity gain. Note that you'll have to increase the max context in the Kobold Lite UI as well (click and edit the number text field).
- Big context too slow? Try the `--smartcontext` flag to reduce prompt processing frequency. Also, you can try to run with your GPU using CLBlast, with `--useclblast` flag for a speedup
- Want even more speedup? Combine `--useclblast` with `--gpulayers` to offload entire layers to the GPU! **Much faster, but uses more VRAM**. Experiment to determine number of layers to offload, and reduce by a few if you run out of memory.
- If you are having crashes or issues, you can try turning off BLAS with the `--noblas` flag. You can also try running in a non-avx2 compatibility mode with `--noavx2`. Lastly, you can try turning off mmap with `--nommap`.

For more information, be sure to run the program with the `--help` flag.

## OSX and Linux
- You will have to compile your binaries from source. A makefile is provided, simply run `make`
- If you want you can also link your own install of OpenBLAS manually with `make LLAMA_OPENBLAS=1`
- Alternatively, if you want you can also link your own install of CLBlast manually with `make LLAMA_CLBLAST=1`, for this you will need to obtain and link OpenCL and CLBlast libraries.
  - For Arch Linux: Install `cblas` `openblas` and `clblast`.
  - For Debian: Install `libclblast-dev` and `libopenblas-dev`.
- For a full featured build, do `make LLAMA_OPENBLAS=1 LLAMA_CLBLAST=1 LLAMA_CUBLAS=1`
- After all binaries are built, you can run the python script with the command `koboldcpp.py [ggml_model.bin] [port]`
- Note: Many OSX users have found that the using Accelerate is actually faster than OpenBLAS. To try, you may wish to run with `--noblas` and compare speeds.

## Compiling on Windows
- You're encouraged to use the .exe released, but if you want to compile your binaries from source at Windows, the easiest way is:
  - Use the latest release of w64devkit (https://github.com/skeeto/w64devkit). Be sure to use the "vanilla one", not i686 or other different stuff. If you try they will conflit with the precompiled libs!
  - Make sure you are using the w64devkit integrated terminal, then run 'make' at the KoboldCpp source folder. This will create the .dll files.
  - If you want to generate the .exe file, make sure you have the python module PyInstaller installed with pip ('pip install PyInstaller').
  - Run the script make_pyinstaller.bat at a regular terminal (or Windows Explorer).
  - The koboldcpp.exe file will be at your dist folder.
- If you wish to use your own version of the additional Windows libraries (OpenCL, CLBlast and OpenBLAS), you can do it with:
  - OpenCL - tested with https://github.com/KhronosGroup/OpenCL-SDK . If you wish to compile it, follow the repository instructions. You will need vcpkg.
  - CLBlast - tested with https://github.com/CNugteren/CLBlast . If you wish to compile it you will need to reference the OpenCL files. It will only generate the ".lib" file if you compile using MSVC.
  - OpenBLAS - tested with https://github.com/xianyi/OpenBLAS .
  - Move the respectives .lib files to the /lib folder of your project, overwriting the older files.
  - Also, replace the existing versions of the corresponding .dll files located in the project directory root (e.g. libopenblas.dll).
  - Make the KoboldCPP project using the instructions above.

## Android (Termux) Alternative method
- See https://github.com/ggerganov/llama.cpp/pull/1828/files

## Using CuBLAS
- If you're on Windows with an Nvidia GPU you can get CUDA support out of the box using the `--usecublas` flag, make sure you select the correct .exe with CUDA support.
- You can attempt a CuBLAS build with `LLAMA_CUBLAS=1` or using the provided CMake file (best for visual studio users). If you use the CMake file to build, copy the `koboldcpp_cublas.dll` generated into the same directory as the `koboldcpp.py` file. If you are bundling executables, you may need to include CUDA dynamic libraries (such as `cublasLt64_11.dll` and `cublas64_11.dll`) in order for the executable to work correctly on a different PC.

## AMD
- Please check out https://github.com/YellowRoseCx/koboldcpp-rocm

## Questions and Help
- **First, please check out [The KoboldCpp FAQ and Knowledgebase](https://github.com/LostRuins/koboldcpp/wiki) which may already have answers to your questions! Also please search through past issues and discussions.**
- If you cannot find an answer, open an issue on this github, or find us on the [KoboldAI Discord](https://koboldai.org/discord).

## Considerations
- For Windows: No installation, single file executable, (It Just Works)
- Since v1.0.6, requires libopenblas, the prebuilt windows binaries are included in this repo. If not found, it will fall back to a mode without BLAS.
- Since v1.15, requires CLBlast if enabled, the prebuilt windows binaries are included in this repo. If not found, it will fall back to a mode without CLBlast.
- Since v1.33, you can set the context size to be above what the model supports officially. It does increases perplexity but should still work well below 4096 even on untuned models. (For GPT-NeoX, GPT-J, and LLAMA models) Customize this with `--ropeconfig`.
- **I plan to keep backwards compatibility with ALL past llama.cpp AND alpaca.cpp models**. But you are also encouraged to reconvert/update your models if possible for best results.

## License
- The original GGML library and llama.cpp by ggerganov are licensed under the MIT License
- However, Kobold Lite is licensed under the AGPL v3.0 License
- The other files are also under the AGPL v3.0 License unless otherwise stated

## Notes
- Generation delay scales linearly with original prompt length. If OpenBLAS is enabled then prompt ingestion becomes about 2-3x faster. This is automatic on windows, but will require linking on OSX and Linux. CLBlast speeds this up even further, and `--gpulayers` + `--useclblast` or `--usecublas` more so.
- I have heard of someone claiming a false AV positive report. The exe is a simple pyinstaller bundle that includes the necessary python scripts and dlls to run. If this still concerns you, you might wish to rebuild everything from source code using the makefile, and you can rebuild the exe yourself with pyinstaller by using `make_pyinstaller.bat`
- Supported GGML models (Includes backward compatibility for older versions/legacy GGML models, though some newer features might be unavailable):
  - LLAMA and LLAMA2 (LLaMA / Alpaca / GPT4All / Vicuna / Koala / Pygmalion 7B / Metharme 7B / WizardLM and many more)
  - GPT-2 / Cerebras
  - GPT-J
  - RWKV
  - GPT-NeoX / Pythia / StableLM / Dolly / RedPajama
  - MPT models
  - Falcon (GGUF only)

