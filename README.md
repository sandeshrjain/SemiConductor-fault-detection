# SemiConductor-fault-detection
Repo for fault detection and localization using DNN architectures and computer vision libraries https://github.com/sandeshrjain/Surface-Defect-Detection/tree/master/DeepPCB

Additional non-CV dataset at SemCom dataset (http://archive.ics.uci.edu/ml/datasets/SECOM)

## Annotation visualization:

<img src="https://github.com/sandeshrjain/SemiConductor-fault-detection/blob/main/Annotations/res_1.png">

### How to use?

Create the environment

```bash
source activate
conda env create -f environment.yml
conda activate fault_det
```

Clone this repository: 

```
git clone https://github.com/sandeshrjain/SemiConductor-fault-detection.git
cd SemiConductor-fault-detection
```

Install detectron2:

```
sh install-detectron2.sh
```

Clone the data set repo:

```
git clone https://github.com/Charmve/Surface-Defect-Detection.git
```
Run Detectron2_detection.py
```
python Detectron2_detection.py
```
