# SAM 3 自动选取（Web 演示）

[English](README.md) | 中文说明

---

本项目是一个基于 **SAM3** 的交互式分割 Web Demo，提供三种模式：

- **悬停选取**：在浏览器中使用 **decoder‑only ONNX 模型** 实现毫秒级悬停分割。
- **点击分割**：后端 PyTorch 在点击点附近进行精细分割。
- **文字分割**：通过文本提示（例如 `car, wheel`）进行分割。

整体思路：后端用 **SAM3 encoder（PyTorch）** 对整张图片做一次特征提取，前端用 **decoder‑only ONNX 模型** 在浏览器里做快速悬停解码，在速度和效果之间取得平衡。

更多技术细节（架构、接口、ONNX 导出、坐标映射等）可参考：

- `SAM3_auto_selection_summary.md`

---

## 1. 环境与使用步骤

### 第一步：下载 SAM3 模型

1. **官方模型（优先推荐）**

   - Hugging Face 模型页：  
     <https://huggingface.co/facebook/sam3>

   - 申请访问权限后，将模型文件下载到本地目录，例如：

     ```text
     D:\HF_DATA\sam3
     ```

2. **如果 Hugging Face 仓库申请被拒绝**

   - 可以在 ModelScope 上下载同一模型：  
     <https://www.modelscope.cn/models/facebook/sam3/>
   - 同样放到本地某个目录（例如仍然使用 `D:\HF_DATA\sam3`）。

稍后在 Web 页面中，需要将这个本地路径填入 **“Sam3 模型地址”** 输入框。

### 第二步：克隆本仓库

```bash
git clone https://github.com/Hasasasa/Sam3_AutoSelection.git
cd Sam3_AutoSelection
```

### 第三步：创建并安装虚拟环境

使用仓库提供的一键脚本创建虚拟环境并安装依赖：

```bat
setup_venv.bat
```

脚本会：

- 在当前目录创建 `.\venv` 虚拟环境；
- 自动激活该虚拟环境；
- 根据 `requirements.txt` 安装所有依赖包。

默认情况下，`requirements.txt` 中安装的 `torch` / `torchvision` 为 **CPU 版本**，可以在没有 GPU 的环境下正常运行。  
如果你希望使用 **GPU（CUDA）** 加速推理，请在虚拟环境中根据 PyTorch 官方说明手动安装对应 CUDA 版本的 `torch` 和 `torchvision`：<https://pytorch.org/get-started/>。

之后每次使用项目，只需先手动激活虚拟环境：

```bat
cd Sam3_AutoSelection
venv\Scripts\activate
```

### 第四步：启动后端服务

在已经激活的虚拟环境中运行：

```bat
python server.py
```

默认 API 监听在：

- `http://0.0.0.0:8000`  
  前端通常使用 `http://localhost:8000` 作为后端地址。

### 第五步：在浏览器中使用 Web UI

1. 在浏览器中打开 `web.html`（例如在资源管理器中双击，或者在编辑器里右键“在浏览器中打开”）。
2. 在页面顶部配置：
   - **后端模型地址**：一般为 `http://localhost:8000`。
   - **Sam3 模型地址**：填写你刚才下载好的 SAM3 模型所在路径，例如：

     ```text
     D:/HF_DATA/sam3
     ```
3. 点击 **“上传图片”**：
   - 将图片上传到后端。
   - 使用 SAM3 encoder 跑一次整图特征。
   - 下载 decoder‑only ONNX 模型（如果此前尚未加载）。
   - 调用 `/get_embeddings` 预计算悬停用特征。
   - 如果上传或预处理失败，右侧的 **“重试预处理”** 按钮会变为可用；在修复后端或路径后，点击即可重新预处理，无需重新选择图片。
4. 使用三种模式：
   - **悬停选取**：移动鼠标即可看到实时掩膜；点击可将当前 hover 结果“固定”，并在边缘画 2 像素左右的白线，仅保留一个固定选区。
   - **点击分割**：在“点击分割”模式下点击目标区域，后端返回高质量掩膜，并在边缘描白。
   - **文字分割**：在文本框中输入英文提示（逗号分隔，如 `window,wheel`），点击“运行文字分割”，后端根据文本生成掩膜。

---

## 2. [功能文档](Design_Document.md)

---

## 3. 许可证

请阅读仓库中的 `LICENSE` 文件，了解并遵守本项目的使用许可条款。
