import torch

def get_inner_iou(box1, box2, ratio=1.0, eps=1e-7):
    """
    计算 Inner-IoU
    Args:
        box1: 预测框 [x1, y1, x2, y2]
        box2: 真实框 [x1, y1, x2, y2]
        ratio: 比例因子
    """
    # 1. 解析坐标 (x1, y1, x2, y2)
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    
    # 2. 计算原始框的宽和高
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    
    # 3. 计算中心点坐标
    b1_x_center, b1_y_center = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    b2_x_center, b2_y_center = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
    
    # 4. 生成辅助框 (Inner/Outer Box)
    # 核心逻辑：基于中心点，按照 ratio 缩放宽高
    inner_b1_x1 = b1_x_center - (w1 * ratio) / 2
    inner_b1_x2 = b1_x_center + (w1 * ratio) / 2
    inner_b1_y1 = b1_y_center - (h1 * ratio) / 2
    inner_b1_y2 = b1_y_center + (h1 * ratio) / 2

    inner_b2_x1 = b2_x_center - (w2 * ratio) / 2
    inner_b2_x2 = b2_x_center + (w2 * ratio) / 2
    inner_b2_y1 = b2_y_center - (h2 * ratio) / 2
    inner_b2_y2 = b2_y_center + (h2 * ratio) / 2
    
    # 5. 计算辅助框的 IoU
    inter = (torch.min(inner_b1_x2, inner_b2_x2) - torch.max(inner_b1_x1, inner_b2_x1)).clamp(0) * \
            (torch.min(inner_b1_y2, inner_b2_y2) - torch.max(inner_b1_y1, inner_b2_y1)).clamp(0)
            
    union = (w1 * ratio) * (h1 * ratio) + (w2 * ratio) * (h2 * ratio) - inter + eps
    
    return inter / union