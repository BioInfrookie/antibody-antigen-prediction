import torch
import torch.nn as nn  # 添加这行导入
from esm import pretrained
import argparse
import time

# 定义与训练脚本相同的模型类
class InteractionPredictor(nn.Module):
    """与训练脚本完全一致的模型定义"""
    def __init__(self, input_dim=1280, hidden_dim=128, dropout=0.3):
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

class InteractionPredictorSystem:
    def __init__(self, model_path, esm_model_name="esm2_t33_650M_UR50D", 
                 hidden_dim=128, dropout=0.3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 加载ESM
        self.esm_model, self.alphabet = pretrained.load_model_and_alphabet(esm_model_name)
        self.esm_model = self.esm_model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm_model.eval()
        
        # 加载预测模型 - 使用与训练相同的模型类
        self.model = InteractionPredictor(
            hidden_dim=hidden_dim, 
            dropout=dropout
        ).to(self.device)
        
        # 加载状态字典
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Loaded model from {model_path}")
    
    def get_embedding(self, sequence):
        """获取单个序列的ESM嵌入"""
        data = [("seq", sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        return results["representations"][33].mean(dim=1).cpu()
    
    def predict_interaction(self, antigen_seq, antibody_seq):
        """预测抗原-抗体互作"""
        start_time = time.time()
        
        # 获取嵌入
        antigen_emb = self.get_embedding(antigen_seq)
        antibody_emb = self.get_embedding(antibody_seq)
        
        # 预测
        with torch.no_grad():
            prediction = self.model(
                antigen_emb.to(self.device), 
                antibody_emb.to(self.device)
            )
        
        interaction_prob = prediction.item()
        interaction_score = interaction_prob * 100  # 转换为0-100的分数
        
        elapsed = time.time() - start_time
        return {
            "interaction_probability": interaction_prob,
            "interaction_score": interaction_score,
            "predicted_class": 1 if interaction_prob > 0.5 else 0,
            "inference_time": elapsed
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict antibody-antigen interaction')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--antigen', type=str, required=True, help='Antigen sequence')
    parser.add_argument('--antibody', type=str, required=True, help='Antibody sequence')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension (must match training)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (must match training)')
    
    args = parser.parse_args()
    
    predictor = InteractionPredictorSystem(
        args.model_path, 
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )
    
    result = predictor.predict_interaction(args.antigen, args.antibody)
    print("\nPrediction Result:")
    print(f"- Interaction Probability: {result['interaction_probability']:.4f}")
    print(f"- Interaction Score: {result['interaction_score']:.2f}")
    print(f"- Predicted Interaction: {'Yes' if result['predicted_class'] else 'No'}")
    print(f"- Inference Time: {result['inference_time']:.2f} seconds")