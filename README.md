# SegFormer_Segmentation
The code uses SegFormer for Semantic Segmentation on Drone Dataset. The details for the SegFormer can be obtained from the following cited paper and the drone dataset can be downloaded from the link below. Alternatively, you can also download the dataset from Kaggle, the link is mentioned below. Clone the repository and install all the packages mentioned in the requirement.txt file. If you just want to infer the semantic segmentation, open the segformer_inf.py, change the image file name you want to test and run the code. Make sure the trained model is in the model folder. You can download the model at https://drive.google.com/file/d/1zsHyMlGJCpPZrDB0v3ZeaogTcUULmUVB/view?usp=sharing. Alternatively, you can train the model and save the model, locally.

If you want to train the SegFormer on the drone dataset. Make sure that the directory structure is as follows:  
root  
| drone_dataset  
|---images  
|----|---test  
|----|---train  
|---mask  
|----|---test  
|----|---train  
|---class_dict_seg.csv  

## Citations and References
```
SegFormer
@article{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={arXiv preprint arXiv:2105.15203},
  year={2021}
}

Drone Dataset
http://dronedataset.icg.tugraz.at/

https://www.kaggle.com/bulentsiyah/semantic-drone-dataset

```
