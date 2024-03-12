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


# Demo
YouTube demo video link

![IMAGE ALT TEXT](http://img.youtube.com/vi/YwQRJuwtddc/0.jpg)

[Demo](https://www.youtube.com/watch?v=YwQRJuwtddc "SpikeNeRF")


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


#Usage

##trian

##test
