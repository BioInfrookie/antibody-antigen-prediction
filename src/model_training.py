import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

class InteractionPredictor(nn.Module):
    """抗体-抗原互作预测模型（带可配置参数）"""
    def __init__(self, input_dim=1280, hidden_dim=128, dropout=0.3):  # ESM-2每序列输出1280维
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dim*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim*4),
            
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, antigen, antibody):
        combined = torch.cat((antigen, antibody), dim=1)
        return self.classifier(combined)

def load_processed_data(data_path):
    data = torch.load(data_path)
    return data

def train_model(data_path, config):
    # 加载数据
    data = load_processed_data(data_path)
    train_data = data['train']
    val_data = data['val']
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(
        train_data['antigen_embeddings'], 
        train_data['antibody_embeddings'],
        train_data['labels']
    )
    val_dataset = torch.utils.data.TensorDataset(
        val_data['antigen_embeddings'], 
        val_data['antibody_embeddings'],
        val_data['labels']
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size*2, shuffle=False, pin_memory=True)
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InteractionPredictor(
        input_dim=1280, 
        hidden_dim=config.hidden_dim,
        dropout=config.dropout
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, min_lr=1e-6)
    
    # 训练参数
    best_auc = 0
    no_improve = 0
    log_file = open(f"training_log_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", "w")
    metrics_history = {'train_loss': [], 'val_auc': [], 'val_f1': []}
    
    print(f"Starting training with config: {config}")
    log_file.write(f"Training config: {vars(config)}\n")
    
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for antigen, antibody, labels in train_loader:
            antigen, antibody, labels = antigen.to(device), antibody.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(antigen, antibody)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * antigen.size(0)
        
        train_loss /= len(train_loader.dataset)
        metrics_history['train_loss'].append(train_loss)
        
        # 验证阶段
        model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for antigen, antibody, labels in val_loader:
                antigen, antibody, labels = antigen.to(device), antibody.to(device), labels.cpu().numpy()
                
                outputs = model(antigen, antibody).cpu().numpy().flatten()
                preds = (outputs > 0.5).astype(int)
                
                all_labels.extend(labels.tolist())
                all_preds.extend(preds.tolist())
                all_probs.extend(outputs.tolist())
        
        # 计算指标
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_probs)
        
        metrics_history['val_auc'].append(auc)
        metrics_history['val_f1'].append(f1)
        
        # 学习率调整
        scheduler.step(auc)
        
        # 保存最佳模型
        if auc > best_auc:
            best_auc = auc
            no_improve = 0
            os.makedirs(config.model_save_path, exist_ok=True)
            best_model_path = f"{config.model_save_path}/best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {auc:.4f}")
        else:
            no_improve += 1
        
        # 早停检查
        if no_improve >= config.patience:
            print(f"Early stopping at epoch {epoch+1}, no improvement for {config.patience} epochs")
            log_file.write(f"Early stopping at epoch {epoch+1}\n")
            break
        
        # 记录日志
        log_str = (f"Epoch {epoch+1}/{config.num_epochs} | Train Loss: {train_loss:.4f} | "
                   f"Val Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | "
                   f"F1: {f1:.4f} | AUC: {auc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(log_str)
        log_file.write(log_str + "\n")
    
    # 最终测试评估
    if 'test' in data:
        print("\nEvaluating on test set...")
        test_data = data['test']
        test_antigen = test_data['antigen_embeddings'].to(device)
        test_antibody = test_data['antibody_embeddings'].to(device)
        test_labels = test_data['labels'].cpu().numpy()
        
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        with torch.no_grad():
            test_probs = model(test_antigen, test_antibody).cpu().numpy().flatten()
            test_preds = (test_probs > 0.5).astype(int)
        
        test_acc = accuracy_score(test_labels, test_preds)
        test_prec = precision_score(test_labels, test_preds, zero_division=0)
        test_rec = recall_score(test_labels, test_preds, zero_division=0)
        test_f1 = f1_score(test_labels, test_preds, zero_division=0)
        test_auc = roc_auc_score(test_labels, test_probs)
        
        test_log = (f"\nTest Results: Acc={test_acc:.4f}, Prec={test_prec:.4f}, Rec={test_rec:.4f}, "
                    f"F1={test_f1:.4f}, AUC={test_auc:.4f}")
        print(test_log)
        log_file.write(test_log + "\n")
    
    # 保存训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(metrics_history['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics_history['val_auc'], label='Val AUC')
    plt.plot(metrics_history['val_f1'], label='Val F1')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.savefig(f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
    plt.close()
    
    log_file.close()
    print(f"Training complete. Best model saved to {best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train antibody-antigen interaction model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to processed data')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model_save_path', type=str, default="saved_models", help='Model save directory')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    train_model(args.data_path, args)