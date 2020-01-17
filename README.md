# [Multitask-Learning-For-Facial-Expression-Analysis-FET-plus-FER](https://github.com/RickZ1010/Multitask-Learning-in-Facial-Expression-Analysis-FET-plus-FER)

## Intordction
This is the implementation of the paper, "DEEP MULTI-TASK LEARNING FOR FACIAL EXPRESSION RECOGNITION WITH EXPRESSION SYNTHESIS REGULARIZATION VIA SELECTIVE FEATURE SHARING". We propose a novel selective feature-sharing method, and establish a multi-task network for facial expression recognition and facial expression synthesis (FERSNet). The proposed method can effectively transfer beneficial features between different tasks, while filtering out useless and harmful information.

## Depandencies
Python >= 3.6.5, Pytorch >= 0.4.1, and cuda-9.2.

## Network architecture
The proposed multi-task network:
![](https://github.com/RickZ1010/Deep-Multitask-Learning-For-Facial-Expression-Analysis-FER-plus-FES/blob/master/figs/fig1.png?raw=true)
The proposed Convolutional Feature Leaky Unit:
<div align=left><img width="400" src="https://github.com/RickZ1010/Deep-Multitask-Learning-For-Facial-Expression-Analysis-FER-plus-FES/blob/master/figs/fig2.png?raw=true"/></div>

## Results
### Facial expression recognition on CK+ and Oulu-CASIA
We consider the first baseline model, denoted as Baseline, as the network without the bottom branch for FES. Thus, this baseline model is a single-task network. We further consider the second baseline model, denoted as Baseline*, as the hard parameter-sharing multi-task model, in which the first five convolutional blocks share the parameters for FER and FES, without ConvFLUs. In addition, we employ the FES branch to enlarge and balance the training dataset. We fine-tune the pre-trained FERSNet with 21K synthetic facial images from the FES branch to further enhance its generalization ability. The fine-tuned model is denoted as FERSNet* in the table.

![](https://github.com/RickZ1010/Deep-Multitask-Learning-For-Facial-Expression-Analysis-FER-plus-FES/blob/master/figs/table1.png?raw=true)

