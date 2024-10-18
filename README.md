<div align="center">
    <h2>
        RSCaMa: Remote Sensing Image Change Captioning with State Space Model
    </h2>
</div>
<br>
<div align="center">
  <img src="resource/RSCaMa.png" width="800"/>
</div>

<div align="center">
  <a href="https://ieeexplore.ieee.org/abstract/document/10537177">
    <span style="font-size: 20px; ">Paper</span>
  </a>
</div>

[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

## Share us a :star: if you're interested in this repo
This repository contains the PyTorch implementation of "RSCaMa: Remote Sensing Image Change Captioning with State Space Model". 


### Installation and Dependencies
```python
git clone https://github.com/Chen-Yang-Liu/RSCaMa.git
cd RSCaMa
conda create -n RSCaMa_env python=3.9
conda activate RSCaMa_env
pip install -r requirements.txt
```

### Data Preparation
- Download the LEVIR_CC dataset: [LEVIR-CC](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset) .
- The data structure of LEVIR-CC is organized as follows:

```
├─/root/Data/LEVIR_CC/
        ├─LevirCCcaptions.json
        ├─images
             ├─train
             │  ├─A
             │  ├─B
             ├─val
             │  ├─A
             │  ├─B
             ├─test
             │  ├─A
             │  ├─B
```
where folder A contains images of pre-phase, folder B contains images of post-phase.

- Extract text files for the change descriptions of each image pair in LEVIR-CC:

```
python preprocess_data.py --input_captions_json /DATA_PATH/Levir-CC-dataset/LevirCCcaptions.json
```

!NOTE: When preparing the text token files, we suggest setting the word count threshold of LEVIR-CC to 5 and Dubai_CC to 0 for fair comparisons.
### NOTE 
- Please modify the source code of ``CLIP`` package, please modify ``CLIP.model.VisionTransformer.forward()`` as [[this](https://github.com/Chen-Yang-Liu/PromptCC/issues/3)].
- Mamba is only supported on Linux systems.

### Training
```
python train_CC.py --data_folder /DATA_PATH/Levir-CC-dataset/images
```

!NOTE: If the program encounters the error: "'Meteor' object has no attribute 'lock'," we recommend installing it with `sudo apt install openjdk-11-jdk` to resolve this issue.


### Evaluate
```python
python test.py --data_folder /DATA_PATH/Levir-CC-dataset/images --checkpoint xxxx.pth
```
Alternatively, you can download our pretrained model here: [[Hugging face](https://huggingface.co/lcybuaa/RSCaMa/tree/main)].


## Experiment: 
<br>
    <div align="center">
      <img src="resource/table1.png" width="800"/>
    </div>
<be>
<br>
    <div align="center">
      <img src="resource/table2.png" width="800"/>
    </div>
<br>
<br>
    <div align="center">
      <img src="resource/table3.png" width="400"/>
    </div>
<br>
    
## Citation: 
```
@ARTICLE{liu2024rscama,
  author={Liu, Chenyang and Chen, Keyan and Chen, Bowen and Zhang, Haotian and Zou, Zhengxia and Shi, Zhenwei},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={RSCaMa: Remote Sensing Image Change Captioning With State Space Model}, 
  year={2024},
  volume={21},
  number={},
  pages={1-5},
  keywords={Decoding;Visualization;Transformers;Task analysis;Solid modeling;Remote sensing;Feature extraction;Change captioning;Mamba;spatial difference-guided SSM;state space model (SSM);temporal traveling SSM},
  doi={10.1109/LGRS.2024.3404604}}

```
<details><summary></summary>
<p>
    
## Our other work: 
Our other Mamba-based work is as follows:
- RSMamba: Remote Sensing Image Classification with State Space Model [[Paper](https://arxiv.org/abs/2403.19654)] [[Code](https://github.com/KyanChen/RSMamba)]
- CDMamba: Remote Sensing Image Change Detection with Mamba [[Paper](https://arxiv.org/abs/2406.04207)] [[Code](https://github.com/zmoka-zht/CDMamba)]
</p></details>
