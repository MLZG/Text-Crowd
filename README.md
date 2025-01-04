# Text-Guided Synthesis of Crowd Animation (SIGGRAPH 2024)
Code for SIGGRAPH 2024 paper "Text-Guided Synthesis of Crowd Animation".

### Preparation

1. Install packages in `requirements.txt`. The implementation is based on Pytorch.
2. Find package `pyDeclutter` and `RVO2_Python` in `Libs`. For each library, run `python setup.py build` to build, and `python setup.py install` to install.
3. Visit [Diffusers](https://huggingface.co/docs/diffusers/en/index) to install the Diffusers, which is a modular library that contains most of the SOTA pre-trained diffusion models.

### Dataset

You can download the **already generated and post-processed dataset** from [this link](https://drive.google.com/file/d/1hFvB3DKTs5cxghOKCdO8YvCqc6hAdz_1/view?usp=sharing). Download `Dataset.zip` to get the dataset and unzip it into the  `Language_Crowd_Animation` folder for use.

Or you can **generate it by yourself**:

1. Run `Dataset_Generation.py` to generate the initial dataset. The generated dataset contains velocity fields without optimization, which would push the agents to be concentrated.
2. Run `Dataset_Postprocess.py` to post-process the initial dataset.

### Training

You can directly use the **pre-trained diffusion models** from [this link](https://drive.google.com/file/d/1rkJaLomTxqvR-YC7GGFVUYTkFG75FTPp/view?usp=sharing). Download `Models_Server_ForTest.zip` to get the pre-trained diffusion models and unzip it into the `Language_Crowd_Animation` folder for use.

Or you can **train the models by yourself**:

1. Run `Trainer_SgDistrDiffusion_Full_V1_Server.py` to train the start and goal diffusion model.
2. Run `Trainer_FieldDiffusion_Full_V2_Server.py` to train the velocity field diffusion model.

### Inference

Run `Quantitative_Exps.py` to evaluate the model using testing data from dataset.
