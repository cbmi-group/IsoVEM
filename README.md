# IsoVEM: Isotropic Reconstruction for Volume Electron Microscopy Based on Transformer

This repository `EMformer` is the official implementation of the bioRxiv paper(https://www.biorxiv.org/content/10.1101/2023.11.22.567807v3).

## Usage

#### 1.Install dependencies

Here's a summary of the key dependencies.

- python 3.7
- pytorch 1.8.1
- CUDA 11.1

We recommend the following demand to install all of the dependencies.

```
conda create -n emformer
conda activate emformer
pip install -r requirements.txt
```

#### 2.Model Training/Testing

The predefined config files are provided in`configs/EPFL.py` or  `configs/Cremi.py`. You can also define a new config file as needed.

Run the training code as follows.

```
python train.py 
```

Run the Testing code as follows.

```
python test.py 
```

#### 3.Attention Visualization

Visualizing the attention module in EMformer helps to better understand model behavior. We use the tool [Visualizer](https://github.com/luo3300612/Visualizer) to generate the attention map of intermediate layer of EMformer.

Run the visualization code as follows.

```
python models/attmap.py 
```

## Acknowledgement

The network code is based on [VRT(Video Restoration Transformer)](https://arxiv.org/abs/2201.12288) and its official [implementation](https://github.com/JingyunLiang/VRT/tree/main). We thank the authors for their work and for sharing the code.

## Citation

If you find this repository useful in your research, please cite our paper:

@article {He2023.11.22.567807,
	author = {Jia He and Yan Zhang and Wenhao Sun and Ge Yang and Fei Sun},
	title = {IsoVEM: Isotropic Reconstruction for Volume Electron Microscopy Based on Transformer},
	elocation-id = {2023.11.22.567807},
	year = {2023},
	doi = {10.1101/2023.11.22.567807},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/12/27/2023.11.22.567807},
	eprint = {https://www.biorxiv.org/content/early/2023/12/27/2023.11.22.567807.full.pdf},
	journal = {bioRxiv}
}