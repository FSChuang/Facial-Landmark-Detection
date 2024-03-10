# <p align = "center"> Facial Landmark Detection </p>
## Table of contents
- [Data Set](#data-set)
- [Data Preprocessing](#data-preprocessing)
- [Model](#model)
- [Short Demo Video](#short-demo)


## <a name = "data-set"> 300W data set 
<p align = "center"> paper:
  <a href="https://arxiv.org/abs/2401.13601](https://ibug.doc.ic.ac.uk/media/uploads/documents/sagonas_iccv_2013_300_w.pdf">300W</a> </p>
<p align = "center"><img width = "600px" src="https://github.com/FSChuang/Facial-Landmark-Detection/assets/124766162/355c434e-a391-4f97-95fb-2c1e94061695"/></p>
The 300-W is a face dataset that consists of 300 Indoor and 300 Outdoor in-the-wild images. It covers a large variation of identity, expression, illumination conditions, pose, occlusion and face size. Images were annotated with the 68-point mark-up. <br/>

## <a name = "data-preprocessing"> Data Preprocessing
To increase the variety of data set and improve the performance of the model, apply augmentation on the input data including:
  - Random Brightness
  - Random Contrast
  - Random Saturation
  - Random Hue
  - Random Rotation

## <a name = "model"> Model
<p align = "center">Architecture: 
  <a href = "https://arxiv.org/abs/1610.02357">[Xception Net]</a></p>
<p align = "center"><img width = "600px" src="https://github.com/FSChuang/Facial-Landmark-Detection/assets/124766162/6b516384-38ad-4322-93e6-40e30dbf0231"/></p>
<p align = "center">$$\textsf{\color{red}TODO: Apply Efficient Net!!!!!!!!!}$$</p>

<details><summary>Training Hyperparameter</summary>
  <pre>
  1.  Objective loss: MSELoss</br>
  2.  Optimizer: Adam</br>
  3.  Learning Rate: 0.0016</br>
  4.  Epoch: 30
  </pre>
</details>


## <a name = "short-demo"> Short demo here~ ⬇ ⬇ ⬇ ⬇
https://github.com/FSChuang/Facial-Landmark-Detection/assets/124766162/8bb4922a-d0f2-4a87-898d-cfc11ce86ae3





