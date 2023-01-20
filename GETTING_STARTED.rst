# VGG 16 classification

Script uses pretrained VGG16 model which head can be trained on a dataset specified by the user.

## How to run the code?
1. Create virtual environment and install packages from requirements.txt

<p>python3 -m venv /path/to/new/virtual/environment</p>

<p>source /path/to/new/virtual/environment</p>

<p>pip install -r requirements.txt</p>

<p>py -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio===0.13.1 -f https://download.pytorch.org/whl/torch_stable.html</p>

<p>py /path/to/main.py --epochs --lr --seed --batch --workers</p>

2. To train model prepare a dataset in .zip format

`!` Can run on GPU or CPU