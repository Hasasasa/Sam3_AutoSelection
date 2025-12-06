# SAM 3 自动选取（Web 演示）

[English](README.md) | 中文说明

---

本项目是一个基于 **SAM3** 的交互式分割 Web Demo，提供三种模式：

- **悬停选取**：在浏览器中使用 **decoder-only ONNX 模型** 实现毫秒级悬停分割。
- **点击分割**：后端 PyTorch 在点击点附近进行精细分割。
- **文字分割**：通过文本提示（例如 `car, wheel`）进行分割。

整体架构：后端使用 **SAM3 encoder（PyTorch）** 做一次整图特征提取，前端使用 **decoder-only ONNX 模型** 做快速交互，在速度和效果之间取得平衡。

更详细的技术拆解（架构、接口、ONNX 导出、坐标映射等）参见：

- `SAM3_auto_selection_summary.md`

---

## 1. 目录结构

- `web_server.py` – FastAPI 后端：图片上传、点 / 文本分割、特征导出、ONNX 模型下载。
- `web_UI.html` – 前端 UI（HTML + CSS + JS），使用 `onnxruntime-web` 做悬停解码。
- `sam3_decoder_onnx/export_sam3_decoder_onnx.py` – 导出 **仅 Decoder** 的 SAM3 ONNX（支持量化）。
- `sam3_decoder_onnx/verify_sam3_decoder_onnx.py` – 校验 ONNX Decoder 与 PyTorch Decoder 的误差。
- `SAM3_auto_selection_summary.md` – 详细技术说明文档。

---

## 2. 导出 Decoder ONNX

在仓库根目录（`sam3-demo`）执行：

```bash
python sam3_decoder_onnx/export_sam3_decoder_onnx.py ^
  --model-path "D:\HF_DATA\sam3" ^
  --out sam3_decoder_onnx/sam3_decoder.onnx ^
  --quant-out sam3_decoder_onnx/sam3_decoder_quant.onnx
```

如果脚本中的默认参数与你的路径一致，也可以直接：

```bash
python sam3_decoder_onnx/export_sam3_decoder_onnx.py
```

---

## 3. 校验 Decoder ONNX（可选）

```bash
python sam3_decoder_onnx/verify_sam3_decoder_onnx.py ^
  --model-path "D:\HF_DATA\sam3" ^
  --onnx-path "sam3_decoder_onnx/sam3_decoder.onnx"

python sam3_decoder_onnx/verify_sam3_decoder_onnx.py ^
  --model-path "D:\HF_DATA\sam3" ^
  --onnx-path "sam3_decoder_onnx/sam3_decoder_quant.onnx"
```

脚本会打印 PyTorch 与 ONNX 掩膜之间的误差，数值很小即可放心使用。

---

## 4. 启动后端

```bash
python web_server.py
# 默认监听 http://0.0.0.0:8000
```

前端默认认为后端地址为：

- `http://localhost:8000`

可以在 `web_UI.html` 顶部的 **“后端模型地址”** 输入框中修改。

---

## 5. 使用 Web UI

1. 在浏览器中打开 `web_UI.html`。
2. 顶部配置：
   - **后端模型地址**：例如 `http://localhost:8000`。
   - **Sam3 模型地址**：例如 `D:/HF_DATA/sam3`。
3. 点击 **“上传图片”**：
   - 将图片上传到后端。
   - 使用 SAM3 encoder 跑一次整图特征。
   - 下载 decoder-only ONNX 模型（如尚未加载）。
   - 调用 `/get_embeddings` 预计算悬停用特征。
   - 如果过程失败，右侧的 **“重试预处理”** 按钮会变为可用，可在修复后端 / 路径后重新预处理。
4. 使用三种模式：
   - **悬停选取**：移动鼠标即可看到实时掩膜；点击可将当前 hover 结果“固定”，并在边缘画白线。
   - **点击分割**：在“点击分割”模式下点击目标区域，后端返回高质量掩膜并描白边。
   - **文字分割**：输入英文提示（逗号分隔，如 `window,wheel`），点击“运行文字分割”。

---

## 6. 许可证

请阅读仓库中的 `LICENSE` 文件，了解并遵守本项目的使用许可条款。

