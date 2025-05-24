# DCDMM: Behavior Condition Diffusion Model for Multi-Modal Recommendation


## 📝 Environment

We develop our codes in the following environment:

- python==3.9.13
- numpy==1.23.1
- torch==1.11.0
- scipy==1.9.1


## 🚀 How to run the codes

We provide the image and text features of the TMALL dataset.


The Sports and Baby datasets are too large to include here, but you can download them from [MMRec](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing).


- Baby

```python
python Main.py --data baby1 --reg 1e-5 --ssl_reg 1e-1 --behavior_layers 3 --lambda_diff 0.15 --lambda_differ 0.01 --temp 0.3 
```

- Sports

```python
python Main.py --data sports1 --reg 1e-6 --ssl_reg 1e-2 --temp 0.2 --keepRate 1 --trans 1 --d_emb_size 10 --alpha 1.0
```

- TMALL

```python
python Main.py --data TMALL_CLIP --reg 1e-5 --ssl_reg 1e-1 --behavior_layers 2 --lambda_diff 0.15 --denoise_embeds 0.5 --temp 0.2 --denoise_drop 0.4 --lambda_differ 1e-1 --image_knn_k 5 --d_emb_size 20 
```




