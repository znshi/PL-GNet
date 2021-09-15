# PL-GNet-image-forgery-detection
Code for "Pixel Level Global Network for Detion and Localization of Image Forgeries". The paper is under review.

# Overview
**PL-GNet** is an end-to-end image manipulation detection model, which takes a manipulated image as input, and predicts manipulation techniques classes and pixel-level manipulation localization simultaneously. Different from most Image Forgery Detection and Localization (IFDL) methods that classify the tampered regions by local patch, the features from the whole image in the spatial and frequency domains are leveraged in this paper to classify each pixel in the image. To combat real-life image forgery which commonly involves different types and visually looks particularly realistic, this paper proposes a high-confidence pixel level global network called PL-GNet without extra pre-processing and post-processing operations. In brief, our main contributions of PL-GNet can be summarized as follows.

1.	 A **pixel level global network** that takes different kinds of inputs into consideration is proposed to localize manipulated regions in the image. 
2.	 In the **newly proposed Encoding Net**, a well-designed first layer and the new backbone network architecture based on atrous convolutions are utilized to extract efficient and abundant features. In addition, the sub-network LSTM Network which introduces co-occurrence based features as input is designed in the frequency domain;
3.	 Our PL-GNet is **not limited to one specific manipulation type** and shows superior performance in localizing manipulated region at pixel level as demonstrated on six challenging datasets. 


There are three building blocks in our end-to-end PL-GNet framework as shown in following figure: 

![Image](https://github.com/znshi/PL-GNet-image-forgery-detection/blob/main/architecture.png)

(1) An Encoding net allows us to extract the global features and generate the high-quality feature maps which indicate possible tampered regions. The new designed first layer and backbone network architecture based on atrous convolutions in Encoding net are adopted to capture the changes of pixel relationships and extract rich multi-scale spatial information. 

(2) A Long Short Term Memory (LSTM) network based on co-occurrence matrix is designed to capture the tampering traces and the discriminative features between manipulated and non-manipulated regions. 

(3) A Decoding net which incorporates the output of Encoding net and LSTM network learns the mapping from low-resolution feature maps to pixel-wise prediction masks. Furthermore, a series of ablation experiments are conducted to systematically optimize the design of the Encoding network. Extensive experiments on the six challenging datasets demonstrate that our PL-GNet outperforms each individual subnetwork, and consistently achieves state-of-the-art performance compared to alternative methods over three evaluation metrics. 

# Dependency
PL-GNet is written in Tensorflow. Some packages need to be installed.


# Test
To run the code on the NC2016 dataset,type:

> python test.py
 
 
# Contact
For any paper related questions, please contact shizn17@mails.jlu.edu.cn
