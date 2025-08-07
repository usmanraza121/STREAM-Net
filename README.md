<h1 align="left">
  <a href="https://doi.org/10.1016/j.knosys.2025.114080" target="_blank">
    STREAM-Net: Spatio-temporal feature fusion network for robust rPPG signal measurement in remote health monitoring
  </a>
</h1>


<div align="center" style="margin-top: 20px; margin-bottom: 20px;">

<br>

**Muhammad Usman<sup>1</sup> &nbsp;&nbsp;|&nbsp;&nbsp; Milena Sobotka<sup>1</sup> &nbsp;&nbsp;|&nbsp;&nbsp; Jacek Ruminski<sup>1</sup>**


<sup>1</sup>Gdańsk University of Technology

<br><br>

</div>


## 🚀 Overview
![STREAM-Net](figures/figure1.png)

Remote photoplethysmography (rPPG) is a popular, non-invasive, contactless technique for detecting physiological signals with promising applications in health monitoring. Recent advances leverage deep learning to overcome challenges such as motion artifacts, redundancy, and external noise in video-based rPPG signal extraction.

We propose **STREAM-Net**, a bilateral spatio-temporal network designed to estimate blood volume pulse (BVP) signals by analyzing human physiological processes in video frames. The network uses spatio-temporal branches with lateral attention and multi-scale feature integration to enhance rPPG signal extraction.

- The **spatio-temporal lateral attention module** integrates spatial-temporal features at multiple resolutions, preserving essential dependencies between spatial and temporal data.
- The **multi-scale feature enhancement module** encodes high-level features to refine spatial features with distinct local and global representations.

Extensive experiments on two benchmark datasets validate the superior performance and robustness of our method. Additionally, we quantify ***predictive uncertainty*** using **[Monte Carlo dropout](link)**, demonstrating the model’s repeatability and reliability.

---



## 🔥Installation
```bash
git clone https://github.com/usmanraza121/STREAM-Net.git
cd STREAM-Net

conda create -n rppg python=3.8 pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=10.2 -c pytorch -q -y

pip install -r requirements.txt
```

## <img width="40" height="40" alt="image" src="https://github.com/user-attachments/assets/096a7afc-0e6d-42fe-8017-c8f9b6f38dff" /> Training and Testing
### Training
```bash
python train.py

```



