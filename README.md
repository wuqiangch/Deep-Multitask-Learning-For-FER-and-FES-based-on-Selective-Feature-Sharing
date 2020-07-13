# [Deep-Multitask-Learning-for-FER-and-FES-based-on-Selective-Feature-Sharing](https://github.com/RickZ1010/Multitask-Learning-in-Facial-Expression-Analysis-FET-plus-FER)

## Intordction
This is the implementation of the following paper:

**DEEP MULTI-TASK LEARNING FOR FACIAL EXPRESSION RECOGNITION AND SYNTHESIS BASE ON SELECTIVE FEATURE SHARING**

*Rui Zhao, Tianshan Liu, Jun Xiao, Daniel P.K. Lun, and Kin-Man Lam*

Abstract: Multi-task learning is an effective learning strategy for deep-learning-based facial expression recognition tasks. However, most existing methods take into limited consideration the feature selection, when transferring information between different tasks, which may lead to task interference when training the multi-task networks. To address this problem, we propose a novel selective feature-sharing method, and establish a multi-task network for facial expression recognition and facial expression synthesis. The proposed method can effectively transfer beneficial features between different tasks, while filtering out useless and harmful information. Moreover, we employ the facial expression synthesis task to enlarge and balance the training dataset to further enhance the generalization ability of the proposed method. Experimental results show that the proposed method achieves state-of-the-art performance on those commonly used facial expression recognition benchmarks, which makes it a potential solution to real-world facial expression recognition problems.

arXiv: [https://arxiv.org/abs/2007.04514](https://arxiv.org/abs/2007.04514)

(The completed implementation will be released after the conference.)

## Depandencies
Python >= 3.6.5, PyTorch >= 0.4.1, and cuda-9.2.

## Network architecture
The proposed multi-task network:
![](https://github.com/RickZ1010/Deep-Multitask-Learning-For-Facial-Expression-Analysis-FER-plus-FES/blob/master/figs/fig1.png?raw=true)
The proposed Convolutional Feature Leaky Unit (ConvFLU):
Structure                  | Procedure
:-------------------------:|:-------------------------:
![](https://github.com/RickZ1010/Deep-Multitask-Learning-For-FER-and-FES-based-on-Selective-Feature-Sharing/blob/master/figs/fig2.png)  |  ![](https://github.com/RickZ1010/Deep-Multitask-Learning-For-FER-and-FES-based-on-Selective-Feature-Sharing/blob/master/figs/ConvFLU_eqs.png)

## Results
### Facial expression recognition on CK+, Oulu-CASIA, and MMI
CK+                        |  Oulu-CASIA               | MMI
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/RickZ1010/Deep-Multitask-Learning-For-FER-and-FES-based-on-Selective-Feature-Sharing/blob/master/figs/table1.png)  |  ![](https://github.com/RickZ1010/Deep-Multitask-Learning-For-FER-and-FES-based-on-Selective-Feature-Sharing/blob/master/figs/table2.png)|  ![](https://github.com/RickZ1010/Deep-Multitask-Learning-For-FER-and-FES-based-on-Selective-Feature-Sharing/blob/master/figs/table3.png)

### Facial expression synthesis on CK+ and Oulu-CASIA
CK+                        |  Oulu-CASIA
:-------------------------:|:-------------------------:
![](https://github.com/RickZ1010/Deep-Multitask-Learning-For-FER-and-FES-based-on-Selective-Feature-Sharing/blob/master/figs/fig3a.png)  |  ![](https://github.com/RickZ1010/Deep-Multitask-Learning-For-FER-and-FES-based-on-Selective-Feature-Sharing/blob/master/figs/fig3b.png)

## Citation

    @INPROCEEDINGS{Zhao2020, 
        author={R. {Zhao} and T. S. {Liu} and J. {Xiao} and D. P. K. {Lun} and K. {Lam}}, 
        booktitle={International Conference on Pattern Recognition (ICPR)}, 
        title={Deep Multi-task Learning For Facial Expression Recognition and Synthesis based on Selective Feature Sharing}, 
        year={2020}}
