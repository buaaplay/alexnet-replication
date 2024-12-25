# train.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models.model import get_model
from utils.dataset import get_dataloaders
from utils.train_utils import set_seed, save_checkpoint
import config
import os
from utils.test_utils import evaluate

# ====== 新增: 命令行超参数解析 ======
def parse_args():
    parser = argparse.ArgumentParser(description="Train AlexNet on CIFAR-10 with minimal changes")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer type: sgd or adam")
    parser.add_argument("--momentum", type=float, default=None, help="Momentum for SGD")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader workers")
    parser.add_argument("--print-freq", type=int, default=None, help="Frequency of print")
    parser.add_argument("--exp-name", type=str, default="default_exp", help="Experiment name for logging")
    args = parser.parse_args()
    return args
# =================================

def main():
    # ====== 新增: 解析命令行参数 ======
    args = parse_args()
    # 更新 config.py 中的默认值（如果命令行传了参数，就覆盖）
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.optimizer is not None:
        config.OPTIMIZER = args.optimizer.lower()  # 在 config 中新加一个 OPTIMIZER
    if args.momentum is not None:
        config.MOMENTUM = args.momentum
    if args.weight_decay is not None:
        config.WEIGHT_DECAY = args.weight_decay
    if args.num_workers is not None:
        config.NUM_WORKERS = args.num_workers
    if args.print_freq is not None:
        config.PRINT_FREQ = args.print_freq
    exp_name = args.exp_name
    # =================================

    # 设置随机种子
    set_seed(config.SEED)

    # 设备配置
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 获取数据加载器
    train_loader, test_loader = get_dataloaders(
        config.DATA_DIR, 
        batch_size=config.BATCH_SIZE, 
        num_workers=config.NUM_WORKERS
    )

    # 初始化模型
    model = get_model(config.NUM_CLASSES).to(device)

    # ====== 新增: 根据 config.OPTIMIZER 判断用哪个优化器 ======
    criterion = nn.CrossEntropyLoss()
    if getattr(config, "OPTIMIZER", "sgd") == "adam":
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        print("使用 Adam 优化器")
    else:
        optimizer = optim.SGD(
            model.parameters(), 
            lr=config.LEARNING_RATE, 
            momentum=config.MOMENTUM, 
            weight_decay=config.WEIGHT_DECAY
        )
        print("使用 SGD 优化器")
    # =================================

    best_accuracy = 0.0

    # ====== 新增: TensorBoard SummaryWriter ======
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f"runs/{exp_name}")
    # =================================

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % config.PRINT_FREQ == 0:
                print(f"Epoch [{epoch}/{config.EPOCHS}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{config.EPOCHS}] 训练损失: {epoch_loss:.4f}")

        # 在 TensorBoard 中记录训练损失
        writer.add_scalar("Train/Loss_Epoch", epoch_loss, epoch)

        # 评估模型
        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch [{epoch}/{config.EPOCHS}] 测试准确率: {accuracy:.2f}%")

        # 在 TensorBoard 中记录测试准确率
        writer.add_scalar("Test/Accuracy", accuracy, epoch)

        # 保存模型
        is_best = accuracy > best_accuracy
        if is_best:
            best_accuracy = accuracy
            print(f"新最佳准确率: {best_accuracy:.2f}%，保存模型.")
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, is_best, config.MODEL_DIR)

    writer.close()  # 关闭 SummaryWriter
    print("训练完成.")

if __name__ == '__main__':
    main()
