cd /d "%~dp0"
copy "C:\Program Files\AMD\ROCm\5.5\bin\hipblas.dll" .\ /Y
copy "C:\Program Files\AMD\ROCm\5.5\bin\rocblas.dll" .\ /Y
xcopy /E /I "C:\Program Files\AMD\ROCm\5.5\bin\rocblas" .\rocblas\ /Y
pip install customtkinter
 
PyInstaller --noconfirm --onefile --collect-all customtkinter --clean --console --icon ".\niko.ico" --add-data "./klite.embd;." --add-data "./kcpp_docs.embd;." --add-data "./koboldcpp_hipblas.dll;." --add-data "./hipblas.dll;." --add-data "./rocblas.dll;." --add-data "./rwkv_vocab.embd;." --add-data "./rwkv_world_vocab.embd;." --add-data "./rocblas;." --add-data "C:/Windows/System32/msvcp140.dll;." --add-data "C:/Windows/System32/vcruntime140_1.dll;." "./koboldcpp.py" -n "koboldcpp_rocm_only.exe"
