# ***SpikeNeRF***
Codes for CVPR 2024 paper "SpikeNeRF: Learning Neural Radiance Fields from Continuous Spike Stream"


# Contents
- [Demo](#Demo)
- [Code](#Code)
- [Requirements](#Requirements)
- [Data](#Data)
- [Usage](#Usage)
  - [trian](#train)
  - [test](#test)
- [Acknowledgement](#Acknowledgement)


# Demo
YouTube demo video link:

![IMAGE ALT TEXT](https://github.com/BIT-Vision/SpikeNeRF/blob/main/demo/SpikeNeRF.mp4)
<div><video autoplay loop controls src="https://user-images.githubusercontent.com/43722188/232816090-ced1fbbc-4246-45c6-a265-e7424e754c7b.mp4" muted="true"></video></div>

![IMAGE ALT TEXT](http://img.youtube.com/vi/AX07rpFppCc/0.jpg)

[Demo](https://www.youtube.com/watch?v=AX07rpFppCc "SpikeNeRF")


# Code
The code is based on [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch).


# Requirements

```
git clone https://github.com/BIT-Vision/SpikeNeRF.git
cd SpikeNeRF
pip install -r requirements.txt
```


# Data
You can download the training data we processed from [Google Drive](https://drive.google.com/drive/my-drive).
Place the downloaded dataset according to the following directory structure:
```
├── configs                                                                                                       
│   ├── ...                                                                                     
│                                                                                               
├── data                                                                                                                                                                                                       
│   ├── nerf_llff_data                                                                                                                                                                                                                              
│   │   └── toys  # downloaded llff dataset                                                                                  
|   |   └── ...
|   ├── nerf_synthetic
|   |   └── lego
|   |   └── chair # downloaded synthetic dataset
|   |   └── ...
```


# Usage

## train
To train SpikeNeRF on different datasets:
```
python run_spikenerf.py --config configs/{DataType}/{DataSet}.txt
```
replace ```{DataType}``` with ```nerf_synthetic``` or ```nerf_llff_data``` , and replace ```{DataSet}``` with ```toys``` | ```dolls``` | ```lego``` | ```chair``` | etc.

## test
To test SpikeNeRF trained on different datasets:
```
python run_spikenerf.py --config configs/{DataType}/{DataSet}.txt --render_only
```
replace ```{DataType}``` with ```nerf_synthetic``` or ```nerf_llff_data``` , and replace ```{DataSet}``` with ```toys``` | ```dolls``` | ```lego``` | ```chair``` | etc.


# Acknowledgement
This codebase is built upon [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch), and thanks to the above open source projects for their help and inspiration.
