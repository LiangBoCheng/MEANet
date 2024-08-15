# MEANet
This project provides the code and results for 'MEANet: An Effective and Lightweight Solution for Salient Object Detection in Optical Remote Sensing Images', ESWA, 2023.
[Link](https://doi.org/10.1016/j.eswa.2023.121778)
# Network Architecture
![image text](https://github.com/LiangBoCheng/MEANet/blob/main/model/MEANet.png)
# Requirements
python 3.7 + pytorch 1.9.0 + imageio 2.22.2
# Saliency maps
We provide saliency maps of our MEANet on [ORSSD](https://pan.baidu.com/s/1E2NzJgVGeoeuhsGV1GSCSQ?pwd=MEAN) and [EORSSD](https://pan.baidu.com/s/1tPQXluovwlSliWhoWEepNw?pwd=MEAN) datasets and additional [ORSI-4199](https://pan.baidu.com/s/1Zbwtl8Fm25POB2McjEsT8Q?pwd=MEAN) datasets.  
We also provide saliency maps of our MEANet on [DUT-O](https://pan.baidu.com/s/1HC0nLaURFVOvUPetNu5_hA?pwd=MEAN) (code:MEAN), [DUTS-TE](https://pan.baidu.com/s/1etx9GEXYGxzFfjhYan_OfA?pwd=MEAN) (code:MEAN), [HKU-IS](https://pan.baidu.com/s/1iC3CWOlcOgJA1sgymwlF8w?pwd=MEAN) (code:MEAN), [ECSSD](https://pan.baidu.com/s/1J-2sWr7VQP3DFU89ZlMmWw?pwd=MEAN) (code:MEAN), [PASCALS](https://pan.baidu.com/s/1aMqhG_KA8ic7vHGegnyKjA?pwd=MEAN) (code:MEAN)
# Training
Run train_MEANet.py.
# Pre-trained model and testing
Download the following pre-trained model and put them in ./models/MEANet/, then run test_MEANet.py.  
[MEANet_EORSSD](https://pan.baidu.com/s/1uowO3bZHL45hZ875xhhTYA) (code:lbc0)  
[MEANet_ORSSD](https://pan.baidu.com/s/1I14LsveMLB-F08XCAsZqEg) (code:lbc1)  
[MEANet_ORSI-4199](https://pan.baidu.com/s/15cOeqJiFv5jeC0IaD4xD9w?pwd=MEAN) (code:MEAN)  
[MEANet_DUTS-TR](https://pan.baidu.com/s/174WPQjH9Dvl82NE-7vXaqQ?pwd=MEAN) (code:MEAN)
# Evaluation Tool
You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.
# Acknowledgement
We would like to thank the contributors to the [MCCNet](https://github.com/MathLee/MCCNet).
