# test.py

import torch
from models.model import get_model
from utils.dataset import get_dataloaders
from utils.test_utils import evaluate
from config import *
import os

def main():
    # 设备配置
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 获取数据加载器
    _, test_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

    # 初始化模型
    model = get_model(NUM_CLASSES).to(device)

    # 加载最佳模型权重
    checkpoint_path = os.path.join(MODEL_DIR, 'best_model.pth')
    if os.path.isfile(checkpoint_path):
        print(f"加载模型权重 '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载模型权重 (epoch {checkpoint['epoch']})")
    else:
        print(f"未找到模型权重 '{checkpoint_path}'")
        return

    # 评估模型
    accuracy = evaluate(model, test_loader, device)
    print(f"测试集准确率: {accuracy:.2f}%")

if __name__ == '__main__':
    main()
