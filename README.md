# <center>koboldcpp-ROCM MIXTRAL FanService Edition for AMD</center>
Quick Linux install:              
To install, either use the file "[easy_KCPP-ROCm_install.sh](https://github.com/YellowRoseCx/koboldcpp-rocm/blob/main/easy_KCPP-ROCm_install.sh)" or navigate to the folder you want to download to in Terminal then run
```        
git clone https://github.com/YellowRoseCx/koboldcpp-rocm.git -b main --depth 1 && \
cd koboldcpp-rocm && \
make LLAMA_HIPBLAS=1 -j4 && \
./koboldcpp.py
```
When the KoboldCPP GUI appears, make sure to select "Use hipBLAS (ROCm)" and set GPU layers 

Original [llama.cpp rocm port](https://github.com/ggerganov/llama.cpp/pull/1087) by SlyEcho et al., modified and ported to koboldcpp by YellowRoseCx

Comparison with OpenCL using 6800xt (old measurement)
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
KoboldCpp is an easy-to-use AI text-generation software for GGML and GGUF models. It's a single self contained distributable from Concedo, that builds off llama.cpp, and adds a versatile Kobold API endpoint, additional format support, backward compatibility, as well as a fancy UI with persistent stories, editing tools, save formats, memory, world info, author's note, characters, scenarios and everything Kobold and Kobold Lite have to offer.

![Preview](media/preview.png)
![Preview](media/preview2.png)
![Preview](media/preview3.png)
![Preview](media/preview4.png)

## Usage
- **[Download the latest .exe release here](https://github.com/YellowRoseCx/koboldcpp-rocm/releases/latest)** or clone the git repo.
- Windows binaries are provided in the form of **koboldcpp.exe**, which is a pyinstaller wrapper for a few **.dll** files and **koboldcpp.py**. You can also rebuild it yourself with the provided makefiles and scripts.
- Weights are not included, you can use the official llama.cpp `quantize.exe` to generate them from your official weight files (or download them from other places such as [TheBloke's Huggingface](https://huggingface.co/TheBloke).
- To run, execute **koboldcpp.exe** or drag and drop your quantized `ggml_model.bin` file onto the .exe, and then connect with Kobold or Kobold Lite. If you're not on windows, then run the script **KoboldCpp.py** after compiling the libraries.
- Launching with no command line arguments displays a GUI containing a subset of configurable settings. Generally you dont have to change much besides the `Presets` and `GPU Layers`. Read the `--help` for more info about each settings.
- By default, you can connect to http://localhost:5001
- You can also run it using the command line `koboldcpp.exe [ggml_model.bin] [port]`. For info, please check `koboldcpp.exe --help`

### Improving Performance
- **(Nivida Only) GPU Acceleration**: If you're on Windows with an Nvidia GPU you can get CUDA support out of the box using the `--usecublas` flag, make sure you select the correct .exe with CUDA support.
- **Any GPU Acceleration**: As a slightly slower alternative, try CLBlast with `--useclblast` flags for a slightly slower but more GPU compatible speedup.
- **GPU Layer Offloading**: Want even more speedup? Combine one of the above GPU flags with `--gpulayers` to offload entire layers to the GPU! **Much faster, but uses more VRAM**. Experiment to determine number of layers to offload, and reduce by a few if you run out of memory.
- **Increasing Context Size**: Try `--contextsize 4096` to 2x your context size! without much perplexity gain. Note that you'll have to increase the max context in the Kobold Lite UI as well (click and edit the number text field).
- **Reducing Prompt Processing**: Try the `--smartcontext` flag to reduce prompt processing frequency.
- If you are having crashes or issues, you can try turning off BLAS with the `--noblas` flag. You can also try running in a non-avx2 compatibility mode with `--noavx2`. Lastly, you can try turning off mmap with `--nommap`.

For more information, be sure to run the program with the `--help` flag, or [check the wiki](https://github.com/LostRuins/koboldcpp/wiki).

## Run on Colab
- KoboldCpp now has an **official Colab GPU Notebook**! This is an easy way to get started without installing anything in a minute or two. [Try it here!](https://colab.research.google.com/github/LostRuins/koboldcpp/blob/concedo/colab.ipynb).
- Note that KoboldCpp is not responsible for your usage of this Colab Notebook, you should ensure that your own usage complies with Google Colab's terms of use.

## OSX and Linux
- You will have to compile your binaries from source. A makefile is provided, simply run `make`.
- If you want you can also link your own install of OpenBLAS manually with `make LLAMA_OPENBLAS=1`
- Arch Linux users can install koboldcpp via the AUR package provided by @AlpinDale. Please see [below](#arch-linux) for more details.
- Alternatively, if you want you can also link your own install of CLBlast manually with `make LLAMA_CLBLAST=1`, for this you will need to obtain and link OpenCL and CLBlast libraries.
  - For Arch Linux: Install `cblas` `openblas` and `clblast`.
  - For Debian: Install `libclblast-dev` and `libopenblas-dev`.
- For an ROCm only build, do ``make LLAMA_HIPBLAS=1 -j4`` (-j4 means it will use 4 cores of your CPU; you can adjust accordingly or leave it off altogether)
- For a full featured build, do `make LLAMA_OPENBLAS=1 LLAMA_CLBLAST=1 LLAMA_HIPBLAS=1 -j4`
- After all binaries are built, you can run the python script with the command `koboldcpp.py [ggml_model.bin] [port]`

- Note: Many OSX users have found that the using Accelerate is actually faster than OpenBLAS. To try, you may wish to run with `--noblas` and compare speeds.

## Compiling for AMD on Windows
- You're encouraged to use the .exe released, but if you want to compile your binaries from source at Windows, the easiest way is:
  - Use the latest release of w64devkit (https://github.com/skeeto/w64devkit). Be sure to use the "vanilla one", not i686 or other different stuff. If you try they will conflit with the precompiled libs!
  - Make sure you are using the w64devkit integrated terminal, (powershell should work for the cmake hipblas part)
  - *This site may be useful, it has some patches for Windows ROCm to help it with compilation that I used, but I'm not sure if it's necessary.* https://streamhpc.com/blog/2023-08-01/how-to-get-full-cmake-support-for-amd-hip-sdk-on-windows-including-patches/ 
  - (ROCm Required): https://rocm.docs.amd.com/en/latest/deploy/windows/quick_start.html
    
   Build command used:         
  ``cd koboldcpp-rocm``        
  ``mkdir build && cd build``

  ```cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DLLAMA_HIPBLAS=ON -DCMAKE_C_COMPILER="C:/Program Files/AMD/ROCm/5.5/bin/clang.exe" -DCMAKE_CXX_COMPILER="C:/Program Files/AMD/ROCm/5.5/bin/clang++.exe" -DAMDGPU_TARGETS="gfx803;gfx900;gfx906;gfx908;gfx90a;gfx1010;gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1102"```

  ``cmake --build . -j 6`` (-j 6 means use 6 CPU cores, if you have more or less, feel free to change it to speed things up)

  That puts koboldcpp_hipblas.dll inside of .\koboldcpp-rocm\build\bin           
  copy koboldcpp_hipblas.dll to the main koboldcpp-rocm folder               
  (You can run koboldcpp.py like this right away) like this:             
  
  ``python koboldcpp.py --usecublas mmq --threads 1 --contextsize 4096 --gpulayers 45 C:\Users\YellowRose\llama-2-7b-chat.Q8_0.gguf``

  To make it into an exe, we use ``make_pyinstaller_exe_rocm_only.bat``          
  which will attempt to build the exe for you and place it in /koboldcpp-rocm/dists/        
  **kobold_rocm_only.exe is built!**
  
  -----
  If you'd like to do a full feature build with OPENBLAS and CLBLAST backends, you'll need [w64devkit](https://github.com/skeeto/w64devkit). Once downloaded, open w64devkit.exe and ``cd`` into the folder then run
  ``make LLAMA_OPENBLAS=1 LLAMA_CLBLAST=1 -j4`` then it will build the rest of the backend files.
  Once they're all built, you should be able to just run ``make_pyinst_rocm_hybrid_henk_yellow.bat`` and it'll bundle the files together into koboldcpp_rocm.exe in the \koboldcpp-rocm\dists folder
- If you wish to use your own version of the additional Windows libraries (OpenCL, CLBlast and OpenBLAS), you can do it with:
  - OpenCL - tested with https://github.com/KhronosGroup/OpenCL-SDK . If you wish to compile it, follow the repository instructions. You will need vcpkg.
  - CLBlast - tested with https://github.com/CNugteren/CLBlast . If you wish to compile it you will need to reference the OpenCL files. It will only generate the ".lib" file if you compile using MSVC.
  - OpenBLAS - tested with https://github.com/xianyi/OpenBLAS .
  - Move the respectives .lib files to the /lib folder of your project, overwriting the older files.
  - Also, replace the existing versions of the corresponding .dll files located in the project directory root (e.g. libopenblas.dll).
  - You can attempt a CuBLAS build with using the provided CMake file with visual studio. If you use the CMake file to build, copy the `koboldcpp_cublas.dll` generated into the same directory as the `koboldcpp.py` file. If you are bundling executables, you may need to include CUDA dynamic libraries (such as `cublasLt64_11.dll` and `cublas64_11.dll`) in order for the executable to work correctly on a different PC.
  - Make the KoboldCPP project using the instructions above.

## Android (Termux) Alternative method
- See https://github.com/ggerganov/llama.cpp/pull/1828/files

## AMD
- Please check out https://github.com/YellowRoseCx/koboldcpp-rocm (you're already here (:)

## Docker
- KoboldCpp has a few unofficial third-party community created docker images. Feel free to try them out, but do not expect up-to-date support:
  - https://github.com/korewaChino/koboldCppDocker
  - https://github.com/noneabove1182/koboldcpp-docker
  
## Arch Linux Packages
There are 4 AUR packages available: [CPU-only](https://aur.archlinux.org/packages/koboldcpp-cpu), [CLBlast](https://aur.archlinux.org/packages/koboldcpp-clblast), [CUBLAS](https://aur.archlinux.org/packages/koboldcpp-cuda), and [HIPBLAS](https://aur.archlinux.org/packages/koboldcpp-hipblas). They are, respectively, for users with no GPU, users with a GPU (vendor-agnostic), users with NVIDIA GPUs, and users with a supported AMD GPU.

The recommended installation method is through an AUR helper such as [paru](https://aur.archlinux.org/packages/paru) or [yay](https://aur.archlinux.org/packages/yay):

```sh
paru -S koboldcpp-cpu
```

Alternatively, you can manually install, though it's not recommended (since the build depends on [customtkinter](https://aur.archlinux.org/packages/customtkinter)):

```sh
git clone https://aur.archlinux.org/koboldcpp-cpu.git && cd koboldcpp-cpu

makepkg -si
```

You can then run koboldcpp anywhere from the terminal by running `koboldcpp` to spawn the GUI, or `koboldcpp --help` to view the list of commands for commandline execution (in case the GUI does not work).

## Questions and Help
- **First, please check out [The KoboldCpp FAQ and Knowledgebase](https://github.com/LostRuins/koboldcpp/wiki) which may already have answers to your questions! Also please search through past issues and discussions.**
- If you cannot find an answer, open an issue on this github, or find us on the [KoboldAI Discord](https://koboldai.org/discord).

## Considerations
- For Windows: No installation, single file executable, (It Just Works)
- Since v1.0.6, requires libopenblas, the prebuilt windows binaries are included in this repo. If not found, it will fall back to a mode without BLAS.
- Since v1.15, requires CLBlast if enabled, the prebuilt windows binaries are included in this repo. If not found, it will fall back to a mode without CLBlast.
- Since v1.33, you can set the context size to be above what the model supports officially. It does increases perplexity but should still work well below 4096 even on untuned models. (For GPT-NeoX, GPT-J, and LLAMA models) Customize this with `--ropeconfig`.
- Since v1.42, supports GGUF models for LLAMA and Falcon
- **I plan to keep backwards compatibility with ALL past llama.cpp AND alpaca.cpp models**. But you are also encouraged to reconvert/update your models if possible for best results.

## License
- The original GGML library and llama.cpp by ggerganov are licensed under the MIT License
- However, Kobold Lite is licensed under the AGPL v3.0 License
- The other files are also under the AGPL v3.0 License unless otherwise stated

## Notes
- Generation delay scales linearly with original prompt length. If OpenBLAS is enabled then prompt ingestion becomes about 2-3x faster. This is automatic on windows, but will require linking on OSX and Linux. CLBlast speeds this up even further, and `--gpulayers` + `--useclblast` or `--usecublas` more so.
- I have heard of someone claiming a false AV positive report. The exe is a simple pyinstaller bundle that includes the necessary python scripts and dlls to run. If this still concerns you, you might wish to rebuild everything from source code using the makefile, and you can rebuild the exe yourself with pyinstaller by using `make_pyinstaller.bat`
- API documentation available at `/api` and https://lite.koboldai.net/koboldcpp_api
- Supported GGML models (Includes backward compatibility for older versions/legacy GGML models, though some newer features might be unavailable):
  - LLAMA and LLAMA2 (LLaMA / Alpaca / GPT4All / Vicuna / Koala / Pygmalion 7B / Metharme 7B / WizardLM and many more)
  - GPT-2 / Cerebras
  - GPT-J
  - RWKV
  - GPT-NeoX / Pythia / StableLM / Dolly / RedPajama
  - MPT models
  - Falcon (GGUF only)

