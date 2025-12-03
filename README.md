# 番茄花果识别 Web

上传图片，使用 `model/` 中的 YOLOv8 模型识别番茄的花与果实，并生成带有框的预测图和类别统计。

## 快速开始
1) 创建虚拟环境并安装依赖
```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
```
2) 运行服务
```bash
python app.py
# 浏览器打开 http://localhost:5000
```

## 目录
- `app.py`：Flask 后端，加载 YOLO 模型、处理上传、返回统计。
- `templates/`：前端页面。
- `static/uploads/`：用户上传的原图（运行时自动创建）。
- `static/predictions/`：模型输出的预测图（运行时自动创建）。
- `model/`：YOLO 权重文件（已提供）。

## 使用提示
- 支持图片格式：jpg、jpeg、png、bmp、gif、webp，大小默认限制 10MB。
- 模型预测失败时会在页面显示错误信息；请确认权重路径与版本匹配。
- 新增模型或脚本时请更新 `requirements.txt` 并在页面或 README 补充示例。
