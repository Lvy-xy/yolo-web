
# 番茄花果识别 Web

上传图片，使用 `model/` 中的 YOLO 模型识别番茄花与果实，生成带框的预测图和类别统计。

## 快速开始（本地）
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# 或 source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
python app.py  # 浏览器访问 http://localhost:5000
```

## 使用自定义 ultralytics 源码
- 将修改后的 ultralytics 源码放在项目根目录 `ultralytics-main/`
  （与 `app.py` 同级，对应 `-e ./ultralytics-main`）。
- 本地运行：激活虚拟环境后执行 `pip install -r requirements.txt`
  （会使用本地源码），再运行 `python app.py`。
- Docker 构建：直接 `docker build -t yolo-web .`，`Dockerfile` 会复制 `ultralytics-main/` 并安装。

## 使用 Docker 部署
```bash
# 构建镜像
docker build -t yolo-web .

# 运行容器（挂载模型和上传/输出目录）
docker run -d --name yolo-web -p 8000:8000 \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/static/uploads:/app/static/uploads \
  -v $(pwd)/static/predictions:/app/static/predictions \
  yolo-web
```

## 目录
- app.py：Flask 后端，加载 YOLO 模型并处理上传/预测
- templates/：前端页面
- static/uploads/：用户上传的原图（运行时自动创建）
- static/predictions/：模型输出的预测图（运行时自动创建）
- model/：YOLO 权重文件
- ultralytics-main/：自定义的 YOLO 源码（可选，配合 `-e ./ultralytics-main`）

## 使用提示
- 支持图片格式：jpg、jpeg、png、bmp、gif、webp；默认上传大小限制 20MB。
- 推理失败时页面会显示错误；请检查权重路径和模型版本是否匹配。
- 新增模型或脚本后，记得更新 `requirements.txt` 和本 README 的示例。
