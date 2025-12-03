# 番茄花果识别 Web

上传图片，使用 `model/` 中的 YOLOv8 模型识别番茄的花与果实，并生成带有框的预测图和类别统计。

## 快速开始（本地）
1) 创建虚拟环境并安装依赖
```bash
python -m venv .venv
source .venv/bin/activate  # Windows 用 .venv\\Scripts\\activate
pip install -r requirements.txt
```
2) 运行服务（开发模式）
```bash
python app.py
# 浏览器打开 http://localhost:5000
```

## 使用 Docker 部署
1) 构建镜像
```bash
docker build -t yolo-web .
```
2) 运行容器（挂载模型目录、开放端口）
```bash
docker run -d --name yolo-web -p 8000:8000 \\
  -v $(pwd)/model:/app/model \\
  yolo-web
# 访问 http://<服务器IP>:8000
```
如需持久化上传/预测结果，可额外挂载：
```bash
  -v $(pwd)/static/uploads:/app/static/uploads \\
  -v $(pwd)/static/predictions:/app/static/predictions \\
```

```
docker run -d --name yolo-web -p 8000:8000 \\ -v $(pwd)/model:/app/model \\ -v $(pwd)/static/uploads:/app/static/uploads \\ -v $(pwd)/static/predictions:/app/static/predictions \\ yolo-web
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
