# CCSC
Cross-Compatible Embedding and Semantic Consistent Feature Construction for Sketch Re-identification

### Usage

- This project is based on TransReID[1] ([paper](https://openaccess.thecvf.com/content/ICCV2021/papers/He_TransReID_Transformer-Based_Object_Re-Identification_ICCV_2021_paper.pdf) and [official code](https://github.com/heshuting555/TransReID))

- Usage of this code is free for research purposes only. 

### Installation

```bash
pip install -r requirements.txt
we use /torch 1.7 /timm 0.3.2 for training and evaluation.
```

### Prepare Datasets

Preparing the dataset(Sketch Re-ID dataset[2] ([paper](https://dl.acm.org/doi/pdf/10.1145/3240508.3240606)) and QMUL-Shoe-v2[3] ([paper](https://link.springer.com/content/pdf/10.1007/s11263-020-01382-3.pdf))). and QMUL-Chair-v2[3] ([paper](https://link.springer.com/content/pdf/10.1007/s11263-020-01382-3.pdf))).  
 

### Prepare Transformer Pre-trained Models

You need to download the ImageNet pretrained transformer model : [pre-train](https://drive.google.com/file/d/1HREqNZtlZbX5DJqKkjLu0D2tN_dOyLjA/view?usp=sharing)

## Train

* We utilize 1  GPU for training.
* train, please replace dataset-path with your own path
* To begin training.(See the code and our paper for more details) 

```bash
# Sketch Re-ID dataset 
python train.py --config_file configs/configs/transformerPKU.yml 
# QMUL-Shoe-v2
python train.py --config_file configs/configs/transformer_ShoeV2.yml
# QMUL-Chair-v2
python train.py --config_file configs/configs/transformer_ChairV2.yml
```

## Test

* Downloading the parameter files trained in this paper.( Using to verify the effectiveness of the proposed method).[Sketch Re-ID](https://drive.google.com/file/d/1Z3f6KlJk6txnYgpOq04Rsiz3vYwfPKBD/view?usp=sharing), [QMUL-Shoe-v2](https://drive.google.com/file/d/1YFN16c_nlLk2MN_2oKWqVhohxwSq7CUL/view?usp=sharing), [QMUL-Chair-v2](https://drive.google.com/file/d/1ZLTsUGDb1BjIKML1Tre1xOIjk_HclXUY/view?usp=sharing).
* To begin testing.(See the code for more details)    

```bash
# Sketch Re-ID dataset 
python test.py --config_file configs/configs/transformerPKU.yml   TEST.WEIGHT 'PKU_logs/transformer_100.pth'
# QMUL-Shoe-v2
python test.py --config_file configs/MSMT17/vit_transreid_stride.yml  TEST.WEIGHT 'shoe_logs/transformer_100.pth'
# QMUL-Chair-v2
python test.py --config_file configs/OCC_Duke/vit_transreid_stride.yml  TEST.WEIGHT 'chair_logs/transformer_100.pth'
```

## Contact

If you have any question, please feel free to contact me. E-mail: [wangyongzeng@stu.kust.edu.cn](wangyongzeng@stu.kust.edu.cn),[shuangli936@gmail.com](shuangli936@gmail.com)

## Reference
```
[1]He S, Luo H, Wang P, et al. Transreid: Transformer-based object re-identification[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2021: 15013-15022.
[2]Pang L, Wang Y, Song Y Z, et al. Cross-domain adversarial feature learning for sketch re-identification[C]//Proceedings of the 26th ACM international conference on Multimedia. 2018: 609-617.
[3]Yu Q, Song J, Song Y Z, et al. Fine-grained instance-level sketch-based image retrieval[J]. International Journal of Computer Vision, 2021, 129(2): 484-500.
```
