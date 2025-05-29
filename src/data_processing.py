import pandas as pd
import torch
from esm import pretrained
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """加载并处理原始CSV数据"""
    df = pd.read_csv(file_path)
    print(f"Loaded data: {len(df)} samples")
    # 检查数据有效性
    assert {'antigen', 'antibody', 'label'}.issubset(df.columns), "Missing required columns"
    assert df['label'].isin([0, 1]).all(), "Labels must be 0 or 1"
    return df

def extract_esm_embeddings(sequences, model, alphabet, batch_size=8):
    """使用ESM模型提取序列嵌入"""
    model.eval()
    embeddings = []
    batch_converter = alphabet.get_batch_converter()
    
    # 分批处理
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        data = [(f"seq{j}", seq) for j, seq in enumerate(batch_seqs)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.cuda()
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        batch_embeddings = results["representations"][33].mean(dim=1).cpu()  # 平均池化
        embeddings.append(batch_embeddings)
    
    return torch.cat(embeddings)

def process_data(input_csv, test_size=0.15, val_size=0.15, output_dir="processed_data", seed=42):
    """主数据处理函数，加入测试集划分"""
    os.makedirs(output_dir, exist_ok=True)
    df = load_data(input_csv)
    
    # 加载ESM模型
    model, alphabet = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    model = model.cuda()
    
    # 划分训练+验证集和测试集
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df['label'])
    
    # 从训练+验证集中划分验证集
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size/(1-test_size), 
        random_state=seed, stratify=train_val_df['label'])
    
    print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # 提取训练集嵌入
    print("Processing training set...")
    train_antigen_emb = extract_esm_embeddings(train_df['antigen'].tolist(), model, alphabet)
    train_antibody_emb = extract_esm_embeddings(train_df['antibody'].tolist(), model, alphabet)
    
    # 提取验证集嵌入
    print("Processing validation set...")
    val_antigen_emb = extract_esm_embeddings(val_df['antigen'].tolist(), model, alphabet)
    val_antibody_emb = extract_esm_embeddings(val_df['antibody'].tolist(), model, alphabet)
    
    # 提取测试集嵌入
    print("Processing test set...")
    test_antigen_emb = extract_esm_embeddings(test_df['antigen'].tolist(), model, alphabet)
    test_antibody_emb = extract_esm_embeddings(test_df['antibody'].tolist(), model, alphabet)
    
    # 保存处理后的数据
    processed_data = {
        'train': {
            'antigen_embeddings': train_antigen_emb,
            'antibody_embeddings': train_antibody_emb,
            'labels': torch.tensor(train_df['label'].values, dtype=torch.float)
        },
        'val': {
            'antigen_embeddings': val_antigen_emb,
            'antibody_embeddings': val_antibody_emb,
            'labels': torch.tensor(val_df['label'].values, dtype=torch.float)
        },
        'test': {
            'antigen_embeddings': test_antigen_emb,
            'antibody_embeddings': test_antibody_emb,
            'labels': torch.tensor(test_df['label'].values, dtype=torch.float)
        }
    }
    
    save_path = os.path.join(output_dir, "processed.pt")
    torch.save(processed_data, save_path)
    print(f"Processed data saved to {save_path}")
    return save_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process antibody-antigen interaction data')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--test_size', type=float, default=0.15, help='Test set proportion')
    parser.add_argument('--val_size', type=float, default=0.15, help='Validation set proportion')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default="processed_data", help='Output directory')
    args = parser.parse_args()
    
    process_data(
        args.input, 
        test_size=args.test_size,
        val_size=args.val_size,
        output_dir=args.output_dir,
        seed=args.seed
    )