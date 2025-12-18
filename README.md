# 番茄花果识别 Web

上传图片，使用 `model/` 中的 YOLO 模型识别番茄花与果实，生成带框的预测图和类别统计。

## 快速开始（本地）
1) python -m venv .venv
2) .venv\Scripts\activate（Windows）或 source .venv/bin/activate（Linux/macOS）
3) pip install -r requirements.txt
4) python app.py
   浏览器打开 http://localhost:5000

## 使用自定义 ultralytics 源码
- 将你修改后的 ultralytics 仓库放在项目根目录，路径为 ./ultralytics（与 app.py 同级）。
- 本地运行：激活虚拟环境后执行 pip install -r requirements.txt（包含 -e ./ultralytics，会使用本地源码），然后运行 python app.py。
- Docker 构建：直接 docker build -t yolo-web .，Dockerfile 会复制 ultralytics/ 并用 -e ./ultralytics 安装；构建前确保该目录存在且为你的修改版本。

## 使用 Docker 部署
1) docker build -t yolo-web .
2) docker run -d --name yolo-web -p 8000:8000 -v $(pwd)/model:/app/model -v $(pwd)/static/uploads:/app/static/uploads -v $(pwd)/static/predictions:/app/static/predictions yolo-web
   访问 http://<服务器IP>:8000

## 目录
- app.py：Flask 后端，加载 YOLO 模型并处理上传/预测
- templates/：前端页面
- static/uploads/：用户上传的原图（运行时自动创建）
- static/predictions/：模型输出的预测图（运行时自动创建）
- model/：YOLO 权重文件
- ultralytics/：自定义 YOLO 源码（可选，配合 -e ./ultralytics）

## 提示
- 支持图片格式：jpg、jpeg、png、bmp、gif、webp；默认上传大小限制 20MB。
- 如果推理失败，页面会显示错误；
  请检查权重路径和模型版本是否匹配。
- 新增模型或脚本后，记得更新 requirements.txt 和本 README 的示例。
