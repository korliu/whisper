# Whisper


## Getting Started:


Whisper: https://huggingface.co/openai/whisper-large-v2/tree/main <br>
Dataset from: https://commonvoice.mozilla.org/en/dataset (Common Voice Delta Segment 13.0)

<br>
### Installation Requirements: (Windows 11, Python 3.10.11)
1. Pytorch: https://pytorch.org/get-started/locally/
     <br> ``` pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 ```
2. Chocolatey: https://chocolatey.org/install
     <br> Open Powershell as Administrator and run: ``` Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1')) ```
3. FFMPEG: Run in command line:
     <br> ```choco install ffmpeg```
4. Hugging Face Transformers: (To use for Whisper)
     <br> ``` pip install transformers ```
5. Hugging Face Evaluate: 
     <br> ``` pip install evaluate ```
   
