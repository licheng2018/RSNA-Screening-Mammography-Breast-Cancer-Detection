This is the introduction of the 94th solution for  [RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview). The objective of this competition is to detect breast cancer in screening mammograms acquired during routine screenings. The dataset utilized for training, validation, and testing consists of mammograms in DICOM format, along with accompanying information such as laterality, age, biopsy results, and more. The submissions are assessed based on the [probabilistic F1 score (pF1)](https://aclanthology.org/2020.eval4nlp-1.9.pdf). To attain the final outcome, we combine the [SEResNeXt](https://arxiv.org/abs/1709.01507v4) and [ConvNeXtV2](https://arxiv.org/abs/2301.00808) models.

# Summary:
* Images: Kaggle train data 
* Preprocess: ROI cropping in CV Canny Detection cv2.drawContours
* Model: EfficientNetV2S (and EfficientNet B5) with GeM pooling (p=3) and ConvNext v2