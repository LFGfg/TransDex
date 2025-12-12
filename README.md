# TransDex: Pre-training Visuo-Tactile Policy with Point Cloud Reconstruction for Dexterous Manipulation of Transparent Objects
[🥛**Project Page**](https://transdex.github.io/) | [📄**Paper(Arxiv)**](https://arxiv.org/) 

Fengguan Li, Yifan Ma, Wentao Rao, Weiwei Shang, Chen Qian

<br/>

<img src="policy_ws\image\Method Overview.png" align="middle" width="100%"/> 

<br/>

We propose **TransDex**,  a 3D visuo-tactile fusion motor policy based on point cloud reconstruction pre-training.

---


## ⚙️ Installation
✅ This project is recommended to run in the following environment:
- **Linux**: Ubuntu 20.04.
- **CUDA version**: 12.9.1.
- **Python version**: 3.9.
- **PyTorch version**: the official build version adapted to CUDA 12.9.

⚡ **Important**: Please ensure CUDA 12.9.1 and the corresponding PyTorch version are **already installed**. This guide does **not** cover CUDA/PyTorch installation.


### Clone Repository

```bash
git clone https://github.com/LFGfg/TransDex.git
cd TransDex
```

### Setup Instructions

#### 1. Create and activate Conda environment
```bash
conda create -n vtfusion python=3.9 -y && conda activate VTFusion
```

#### 2. Install dependencies
```bash
sudo apt install libgl1 -y && sudo apt-get install g++ -y
conda install pinocchio xorg-libx11 -c conda-forge -y
pip install -r requirements.txt
```

#### 3. Build and install PointOps extension
Adjust the path according to your project layout：
```bash
cd /policy_ws/src/VTFusion/src/extensions/pointops
python setup.py install
```

---

## 🛠️ Usage

### 1. Pretrain
First enter the pre-training document directory：
```bash
cd /policy_ws/src/VTFusion/src/PretrainPoint
```
The code for the dataset processing and model in the pre-training stage can be found in `./models/Dataset_process_nor.py` and `./models/PretrainPoint.py`. Pre-trained data used in this project are generated in Pybullet simulator, and the dexterous hand used can be found in this [paper](https://ieeexplore.ieee.org/abstract/document/11031426). The sample dataset will be released at Google Drive later.

<!-- [Google Drive](https://drive.google.com/file/d/1r_ZTIlBnV1tvt5HuZ7LX18nZfiD_H1WX/view?usp=sharing)-->

😸 We strongly suggest that users generate corresponding point cloud datasets according to **your own dexterous hand systems** and process them in the format provided by the data processing codes. 

👉🏻 Before training, please make sure that `dataset.data_dir` in `./cfgs/pretrain_hand_object.yaml` should be changed to the storage location of your own dataset.
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pretrain_hand_object.yaml
```

The trained weight files can be found in `./experiments/pretrain_hand_object/`.

### 2. Policy
#### Hardware Setup
The real robotic system consists of a 16-DOF dexterous hand and a 7-DOF humanoid arm. The robot used in this project can also be found in this [paper](https://ieeexplore.ieee.org/abstract/document/11031426). The dexterous hand is equipped with **Paxini array tactile sensors**. Additionally, the system requires two Intel RealSense D435i depth cameras positioned at the wrists of the robotic arms and around the workbench respectively.

#### Policy Training
Enter the document directory：
```bash
cd /policy_ws/src/VTFusion/src/
```
The code for the dataset processing and model of the policy can be found in `./VTFusion_dataset.py` and `./FusionNetwork.py`. Users can collect manipulation dataset through **your own robotic system**.

👉🏻 Before training the policy, please ensure:
- The pre-trained encoder weight file is located under `pretrain_pointencoder/ckpt.pth`.
- Manipulation dataset is placed under `../data_record/`, and edit the `task_name` in the config file `config/config.yaml`.
- Put the URDF file of the robot in `/policy_ws/src/LeftArmHandURDFV3`.
- Adjust parameters such as `pos_mins/maxs`, `rpy_mins/maxs`, `joint_mins/maxs` in the config file according to robotic system and task. Relevant instructions are already commented in the sample config file `config/config.yaml`.

Use the following script for training：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --config config/config.yaml
```
The trained weight files can be found in `./ckpts/`.

⚡ **Note**: Code files such as `./pin_forward.py`,  `./tactile_sensor_calculator.py` are designed for the robotic systems used in the project.

#### Deploy
This project utilizes ROS and TwinCat communication for underlying motor control. Users can evaluate through your own robotic systems and corresponding trained networks.

#### 💡 Deployment Tips
- Dual RealSense cameras require hand-eye calibration and time synchronization.
- Point cloud fusion necessitates coordinate transformation and ICP registration.
- Calibrate robotic arm/dexterous hand joint zero position and set limits.
- Model inference requires the use of a graphics card to enhance control frequency.
- Ensure all hardware devices is supplied with stable power and properly connected.

---

## 📚 Citation
😄 If you find our work useful, please consider citing:
```bibtex
@article{TransDex2025Li,
  title={TransDex: Pre-training Visuo-Tactile Policy with Point Cloud Reconstruction for Dexterous Manipulation of Transparent Objects},
  author={Fengguan Li, Yifan Ma, Wentao Rao, Weiwei Shang, Chen Qian},
  journal={Conference/Journal Name},
  year={2025},
  url={https://your-domain.com/your-project-page}
}
```

❓ If you have any questions, please contact **lfguan@mail.ustc.edu.cn**.