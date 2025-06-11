# antibody-antigen-prediction
![Model structure](https://github.com/BioInfrookie/antibody-antigen-prediction/blob/main/Model.png)

```Python
这是一个简易版的深度学习检测抗原抗体互作平台。目的：服务于更多有数据集但无法公开发表的人员及机构，可以让其更快的训练互作模型，更好的向前推进项目。
Author：Qiao Zhenghao
Email：qzhrookie@163.com
```

```Python
# 安装显卡驱动
NVDIA Tookie
NVDIA cuda

# 安装依赖
pip install transformers pandas scikit-learn scipy
# 注意 torch必须与显卡驱动配套兼容 5070--cu128
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 一般国内下载不了，网速限制，建议阿里云下载pip安装
https://mirrors.aliyun.com/pytorch-wheels/cu128/

```

```Python
1. 数据处理
python src/data_processing.py --input interaction_data.csv --test_size 0.15 --val_size 0.15

2. 模型训练
python src/model_training.py --data_path processed_data/processed.pt \
               --hidden_dim 256 \
               --dropout 0.4 \
               --lr 0.0005 \
               --batch_size 128 \
               --num_epochs 200

3. 预测交互
python src/prediction.py --model_path saved_models/best_model.pt \
                 --antigen "PIVQNLQGQMVHQCISPRTLNAWVKVVEEKAFSPEVIPMFSALSCGATPQDLNTMLNTVGGHQAAMQMLKETINEEAAEWDRLHPVHAAPGQMREPRGSDIAGTTSTLQEQIGWMTHNPPIPVGEIYKRWIILGLNKIVRMYSPTSILDIRQGPKEPFRDYVDRFYKTLRAEQASQEVKNAATETLLVQNANPDCKTILKALGPGATLEEMMTACQGVGP" \
                 --antibody "DVQLQESGGGLVQAGGSLRLSCAASGSISRFNAMGWWRQAPGKEREFVARIVKGGYAVLADSVKGRFTISIDSAENTLALQMNRLKPEDTAVYYCFAALDTAYWGQGTQVTVS" \
                 --hidden_dim 256

```

