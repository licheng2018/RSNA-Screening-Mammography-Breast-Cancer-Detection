<a name="br1"></a> 

This is the introduction of the 94th solution for [RSNA](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview)[ ](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview)[Screening](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview)[ ](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview)[Mammography](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview)[ ](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview)[Breast](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview)[ ](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview)[Cancer](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview)

[Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview). The objective of this competition is to detect breast cancer in screening

mammograms acquired during routine screenings. The dataset utilized for training, validation,

and testing consists of mammograms in DICOM format, along with accompanying information

such as laterality, age, biopsy results, and more. The submissions are assessed based on the

[probabilistic](https://aclanthology.org/2020.eval4nlp-1.9.pdf)[ ](https://aclanthology.org/2020.eval4nlp-1.9.pdf)[F1](https://aclanthology.org/2020.eval4nlp-1.9.pdf)[ ](https://aclanthology.org/2020.eval4nlp-1.9.pdf)[score](https://aclanthology.org/2020.eval4nlp-1.9.pdf)[ ](https://aclanthology.org/2020.eval4nlp-1.9.pdf)(pF1). To attain the final outcome, we combine the [SEResNeXt](https://arxiv.org/abs/1709.01507v4)[ ](https://arxiv.org/abs/1709.01507v4)and

[ConvNeXtV2](https://arxiv.org/abs/2301.00808)[ ](https://arxiv.org/abs/2301.00808)models.

**Summary:**

**●**

**●**

**●**

Images: Kaggle train data

Preprocess: ROI cropping in CV Canny Detection cv2.drawContours

Model: EfficientNetV2S (and EfficientNet B5) with GeM pooling (p=3) and ConvNext v2

**Extracting Row Data:**

In order to minimize data extracting times, we use DALI for decoding docom using GPU. The

details can be obtained from this [website](https://developer.nvidia.com/blog/rapid-data-pre-processing-with-nvidia-dali/). DALI can be used to accelerate and save the space

of training and test data. The procedure is like the following Fig. 1:

[Fig.](https://developer-blogs.nvidia.com/wp-content/uploads/2021/10/RAPIDSData_Pic2.png)[ ](https://developer-blogs.nvidia.com/wp-content/uploads/2021/10/RAPIDSData_Pic2.png)[1](https://developer-blogs.nvidia.com/wp-content/uploads/2021/10/RAPIDSData_Pic2.png)[ ](https://developer-blogs.nvidia.com/wp-content/uploads/2021/10/RAPIDSData_Pic2.png)Processing of DALI

**Crop Data**



<a name="br2"></a> 

Next, we will crop data to augment the image. The crop can be based on the offset of pixels as

shown in Fig.2. The center points of the cropped image are overlapping areas of horizontal and

vertical peaks. Fig.3 presents the comparison of original and cropped images. The reference of

this section is [RSNA:](https://www.kaggle.com/code/vslaykovsky/rsna-cut-off-empty-space-from-images)[ ](https://www.kaggle.com/code/vslaykovsky/rsna-cut-off-empty-space-from-images)[Cut](https://www.kaggle.com/code/vslaykovsky/rsna-cut-off-empty-space-from-images)[ ](https://www.kaggle.com/code/vslaykovsky/rsna-cut-off-empty-space-from-images)[Off](https://www.kaggle.com/code/vslaykovsky/rsna-cut-off-empty-space-from-images)[ ](https://www.kaggle.com/code/vslaykovsky/rsna-cut-off-empty-space-from-images)[Empty](https://www.kaggle.com/code/vslaykovsky/rsna-cut-off-empty-space-from-images)[ ](https://www.kaggle.com/code/vslaykovsky/rsna-cut-off-empty-space-from-images)[Space](https://www.kaggle.com/code/vslaykovsky/rsna-cut-off-empty-space-from-images)[ ](https://www.kaggle.com/code/vslaykovsky/rsna-cut-off-empty-space-from-images)[from](https://www.kaggle.com/code/vslaykovsky/rsna-cut-off-empty-space-from-images)[ ](https://www.kaggle.com/code/vslaykovsky/rsna-cut-off-empty-space-from-images)[Images](https://www.kaggle.com/code/vslaykovsky/rsna-cut-off-empty-space-from-images). YOLO can also be a choice to detect

the object, namely the breast in images. The reference of this section is [Breast](https://www.kaggle.com/code/remekkinas/breast-cancer-roi-brest-extractor)[ ](https://www.kaggle.com/code/remekkinas/breast-cancer-roi-brest-extractor)[Cancer](https://www.kaggle.com/code/remekkinas/breast-cancer-roi-brest-extractor)[ ](https://www.kaggle.com/code/remekkinas/breast-cancer-roi-brest-extractor)[-](https://www.kaggle.com/code/remekkinas/breast-cancer-roi-brest-extractor)[ ](https://www.kaggle.com/code/remekkinas/breast-cancer-roi-brest-extractor)[ROI](https://www.kaggle.com/code/remekkinas/breast-cancer-roi-brest-extractor)

[(brest)](https://www.kaggle.com/code/remekkinas/breast-cancer-roi-brest-extractor)[ ](https://www.kaggle.com/code/remekkinas/breast-cancer-roi-brest-extractor)[extractor](https://www.kaggle.com/code/remekkinas/breast-cancer-roi-brest-extractor). Finally, Open CV has its library to get the edge of objects. The reference of this

section is [NextVIT](https://www.kaggle.com/code/programmaticart/nextvit-tensorrt-inference-pytorch/notebook)[ ](https://www.kaggle.com/code/programmaticart/nextvit-tensorrt-inference-pytorch/notebook)[TensorRT](https://www.kaggle.com/code/programmaticart/nextvit-tensorrt-inference-pytorch/notebook)[ ](https://www.kaggle.com/code/programmaticart/nextvit-tensorrt-inference-pytorch/notebook)[Inference](https://www.kaggle.com/code/programmaticart/nextvit-tensorrt-inference-pytorch/notebook)[ ](https://www.kaggle.com/code/programmaticart/nextvit-tensorrt-inference-pytorch/notebook)[|](https://www.kaggle.com/code/programmaticart/nextvit-tensorrt-inference-pytorch/notebook)[ ](https://www.kaggle.com/code/programmaticart/nextvit-tensorrt-inference-pytorch/notebook)[Pytorch](https://www.kaggle.com/code/programmaticart/nextvit-tensorrt-inference-pytorch/notebook). We apply the third method to crop the objects.

Fig 2. Offset of each image

Fig 3. Comparison between original and cropped images



<a name="br3"></a> 

**Sample Submission**

Due to the imbalanced nature of the data, training the model on an unaltered training set may

result in prediction biases. Fig. 4 illustrates the distribution disparity between positive and

negative outcomes

Fig.4 distribution of targets

To mitigate this bias, we performed sampling on the negative data. Specifically, we selected only

10% of the negative data for training purposes.

**Model1**

In the case of model 1, we utilize ConvNext v2 as our chosen approach. In the following

sections, we will delve into the novel aspects and advancements introduced in both ConvNext

and its updated version, ConvNext v2.

**ConvNext**

ConvNext is an innovative computer vision algorithm developed by Facebook AI Research

(FAIR). It builds upon the foundation of ResNet50 or ResNet200 models. Several modifications

have been introduced to enhance the ResNet architecture, such as Macro Design, ResNeXt-ify,

inverted bottleneck, large kernel, and other specialized design elements. The optimized

structure of ConvNext, as shown in Fig. 5, incorporates these modifications to improve its

performance and efficiency.



<a name="br4"></a> 

Fig.5 Improvement of ConvNext



<a name="br5"></a> 

Macro Design:

The ResNet architecture consists of stages with the following distribution: (3, 4, 6, 3). In the

Swin-Transformer, the ratio of stages is either 1:1:3:1 or 1:1:9:1, with the latter being employed

in larger models. On the other hand, ConvNeXt utilizes a stage configuration of 3:3:9:3.

ConvNext introduces a modification to the initial processing of input images by replacing the

stem with Patchify. The stem cell design focuses on the initial processing of input images within

the network. In traditional ConvNets and vision Transformers, a typical stem cell aggressively

reduces the size of input images to generate an appropriate feature map size, taking advantage

of the inherent redundancy present in natural images.

ResNeXt-ify:

The fundamental principle of ResNeXt is to utilize grouped convolution, which divides the

convolutional filters into multiple groups. By increasing the number of groups and expanding the

width of the network, ResNeXt achieves a higher computational speed while significantly

reducing the number of floating-point operations (FLOPs) required.

Inverted Bottleneck:

The hidden dimension of the MLP block in ConvNext is four times larger than the input

dimension. The design of ConvNext is influenced by the inverted bottleneck design commonly

used in ConvNets, which incorporates an expansion ratio of 4.

Large Kernel:

The kernel size used in ConvNext is 7×7. In theory, a larger kernel size leads to a broader

receptive field. By increasing the kernel size from 3×3 to 7×7, the network's performance

improves from 79.9% to 80.6%, while the computational operations (FLOPs) required by the

network remain relatively unchanged.

Specific Optimization:

1\. Replacing ReLU with GELU

2\. Fewer activation functions:

In a Transformer block, there are several components, including key/query/value linear

embedding layers, a projection layer, and an MLP block consisting of two linear layers.

Notably, there is only a single activation function used within the MLP block. In contrast,

it is customary to apply an activation function to each convolutional layer, including the

1×1 convolutions, in other contexts.

3\. Fewer normalization layers

4\. Replacing BN with LN

5\. Separate downsampling layers:

In ResNet, the spatial downsampling is achieved by the residual block at the start of

each stage, using 3×3 conv with stride 2 (and 1×1 conv with stride 2 at the shortcut

connection)

Performance:



<a name="br6"></a> 

Fig.6 Performance of ConvNext

**ConvNext v2**

The update of ConvNext v2 are Fully Convolutional Masked Autoencoder and Global Response

Normalization.

Fully Convolutional Masked Autoencoder:

Our approach is conceptually simple and runs in a fully convolutional manner. The learning

signals are generated by randomly masking the raw input visuals with a high masking ratio and

letting the model predict the missing parts given the remaining context. Fig. 7 provides a visual

representation of this process.

Fig.7 Fully Convolutional Masked Autoencoder



<a name="br7"></a> 

Global Response Normalization:

To enhance the effectiveness of FCMAE (Fully Convolutional Masked Autoencoder) pretraining

in conjunction with the ConvNeXt architecture, we incorporate the Global Response

Normalization (GRN) technique. This technique aims to optimize the performance of FCMAE by

applying normalization across the entire network response.

The performance is shown in Fig.8.

Fig.8 CovNext v2 performance

**Model2**

**Model3**

**Merge Two Result**

