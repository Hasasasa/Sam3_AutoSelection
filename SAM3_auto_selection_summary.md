# SAM 3 Auto Selection 功能总结与实现说明

本文总结当前项目的整体功能设计与实现细节，方便后续回顾和扩展。

---

## 1. 整体架构

- **后端：** `web_server.py`
  - 基于 FastAPI。
  - 负责加载 SAM3 模型、管理图片、计算分割结果、提供 HTTP 接口。
- **前端：** `web_UI.html`
  - 纯 HTML + CSS + 原生 JS。
  - 负责页面布局、模式切换、鼠标交互、调用后端接口。
  - 使用 `onnxruntime-web` 在浏览器中加载 *Decoder-only* ONNX 模型，实现毫秒级悬停分割。
- **ONNX 导出与验证：**
  - `export_sam3_decoder_onnx.py`：导出 SAM3 的 **decoder-only ONNX 模型**，并支持量化。
  - `verify_sam3_decoder_onnx.py`：对比 PyTorch decoder 与 ONNX decoder 的输出，验证误差。

**运行流程（高层）：**

1. 前端上传图片 → 后端 `POST /set_image` 存储图片，返回 `image_id`。
2. 上传完成后，前端自动调用 `POST /get_embeddings`：
   - 后端用 PyTorch encoder 计算整张图的特征（3 个多尺度 feature maps）。
   - 返回 `original_sizes`、`target_size` 和 embeddings。
3. 前端将 embeddings 加载到 `onnxruntime-web`，构造 decoder 的输入张量。
4. 悬停选取时，仅在浏览器端用 ONNX decoder 做预测（encoder 不再重复计算）。
5. 点击分割和文字分割仍由后端 PyTorch 完成，并返回带白边的遮罩 PNG。

---

## 2. 后端实现（`web_server.py`）

### 2.1 模型与处理器

- 使用 HuggingFace 的：
  - `Sam3TrackerModel`：完整模型（视觉 encoder + prompt encoder + mask decoder）。
  - `Sam3TrackerProcessor`：预处理与后处理工具。
- 启动时加载模型到 `cuda` 或 `cpu`：
  - `get_default_device()` 判断是否有 GPU。

### 2.2 图片存储结构

后端维护一个字典 `stored_images[image_id]`，典型内容包括：

- `image`: `PIL.Image` 原图。
- `image_embeddings`: encoder 预计算的多尺度特征列表 `[e0, e1, e2]`。
- `original_sizes`: 处理前原始图像尺寸（例如 `[[H, W]]`）。

### 2.3 关键 API 接口

#### 2.3.1 `POST /set_image`

- 入参：
  - `image_data`: base64 的 `data:image/png;base64,...`。
  - `model_path`: 可选的本地权重路径（例如 `D:/HF_DATA/sam3`）。
- 逻辑：
  - 解码成 `PIL.Image`。
  - 生成新的 `image_id`，保存到 `stored_images`。
  - 将该 `image_id` 之前的 `image_embeddings`、`original_sizes` 重置。
- 返回：

```json
{ "image_id": "xxxx" }
```

#### 2.3.2 `POST /segment_point`

- 入参：`image_id, x, y`（原图像素坐标）。
- 路径选择：
  - **已预处理**（推荐，快速）：
    - 使用 `processor_pvs(images=None, input_points, input_labels, original_sizes)` 构造 decoder 输入。
    - 调用 `model_pvs` 仅运行 decoder，`image_embeddings` 直接复用预处理结果。
  - **未预处理**（慢速）：
    - 每次完整调用 `processor_pvs(images=image, input_points, input_labels)`。
    - 运行 encoder + decoder 一整套推理。
- 解码输出：
  - `outputs.pred_masks` → `processor_pvs.post_process_masks(..., binarize=False)` → 概率 mask（非二值）。
  - 调用 `apply_mask_overlay(image, mask)` 生成可视化遮罩。

`apply_mask_overlay` 的核心逻辑：

- 将 mask 归一化为 `[0,1]`。
- 用指定颜色（默认 `RGB=(30,144,255)`）和透明度 `alpha` 混合原图生成半透明区域。
- 使用 OpenCV：
  - `cv2.MORPH_GRADIENT` 计算 mask 的边缘。
  - 适度膨胀，使边缘约 2px 宽。
  - 在这些边缘像素上直接画成白色（`(255,255,255)`），得到干净的轮廓线。
- 最终编码为 PNG，返回 base64 字符串。

返回结果：

```json
{ "image": "data:image/png;base64,..." }
```

#### 2.3.3 `POST /segment_text`

- 入参：`image_id, text`（例如 `"window,wheel"`）。
- 使用 `Sam3TrackerProcessor` 文本接口构造 prompts。
- 推理和 `segment_point` 类似，最后同样用 `apply_mask_overlay` 生成遮罩。
- 返回同样是带白边的 PNG。

#### 2.3.4 `POST /get_embeddings`

- 入参：`image_id`。
- 如果当前 `image_embeddings` 或 `original_sizes` 为空：
  - 调用：
    ```python
    encoding = processor_pvs(images=stored.image, return_tensors="pt")
    pixel_values = encoding["pixel_values"].to(device)
    original_sizes = encoding["original_sizes"].tolist()
    image_embeddings = model_pvs.get_image_embeddings(pixel_values)
    ```
  - 将 `image_embeddings` 和 `original_sizes` 存到 `stored_images[image_id]`。
- 返回：

```json
{
  "original_sizes": [[H, W]],
  "target_size": 1008,
  "embeddings": [e0_list, e1_list, e2_list]
}
```

前端会将这些 list 转成 `Float32Array`，作为 ONNX decoder 的输入。

---

## 3. Decoder-only ONNX 模型

### 3.1 导出脚本：`export_sam3_decoder_onnx.py`

核心类 `Sam3DecoderOnly`：

- 在 `__init__` 中：
  - 保存 `model.prompt_encoder` 与 `model.mask_decoder`。
  - 调用 `model.get_image_wide_positional_embeddings()`，作为 buffer 复用。
- `forward` 输入：
  - `image_embedding_s0/s1/s2`: 三个多尺度 encoder 特征。
  - `input_points`: `(B, point_batch, num_points, 2)`。
  - `input_labels`: `(B, point_batch, num_points)`。
  - `input_masks`: 可选。
  - `multimask_output`: bool，浏览器端固定为 False。
- 内部调用：
  1. `prompt_encoder`：将点击点 / 文本等提示编码为 sparse / dense embeddings。
  2. `mask_decoder`：使用图片特征 + prompt embeddings + positional embeddings 输出低分辨率 mask。

导出步骤：

1. 构造一张 dummy 图片和一个 dummy 点击点。
2. 使用原模型计算一次 `image_embeddings`，得到真实特征形状。
3. 使用 `torch.onnx.export` 导出：
   - 指定 `input_names`、`output_names`。
   - 设置 `dynamic_axes` 使 batch 和点数可变。
   - `opset_version=17`。
4. 尝试用 `onnx.save_model(..., save_as_external_data=False)` 合并权重为单文件（避免 `.onnx.data`）。

### 3.2 验证脚本：`verify_sam3_decoder_onnx.py`

- 目的：保证 ONNX decoder 的行为与 PyTorch decoder 足够接近，误差小。
- 步骤：
  1. 用 PyTorch 跑一次 encoder+decoder 得到 `pred_masks_pt`。
  2. 用 encoder 得到的 `image_embeddings` 作为输入，调用 ONNX runtime 的 decoder 得到 `pred_masks_onnx`。
  3. 计算差异：
     - `diff_mean = (pt - onnx).abs().mean()`
     - `diff_max = (pt - onnx).abs().max()`
  4. 验证非量化模型 diff 极小；量化模型 diff 稍大但仍可接受。

---

## 4. 前端 UI 结构（`web_UI.html`）

### 4.1 顶部 Header

- 左侧 Logo 区：
  - Icon：`icon/icon.png`。
  - 文本：`SAM 3 Auto Selection`。

- 中间工具条 `toolbar`：
  - `serverInput`：
    - 标签：**后端模型地址**。
    - 默认值：`http://localhost:8000`。
    - 右侧有 `serverStatus` 圆点显示在线 / 离线。
  - `modelPathInput`：
    - 标签：**Sam3 模型地址**。
    - 默认值：`D:/HF_DATA/sam3`。
    - 用于 POST `/set_image` 时告诉后端使用哪个模型路径。

- 右侧上传按钮：
  - `上传图片` 按钮 + 隐藏的 `fileInput`（`accept="image/*"`）。

### 4.2 主体区域

- 左侧画布：
  - `.canvas-container`：整体背景（深色渐变）。
  - 初始显示：
    - `#placeholder`：鼠标手势 emoji + `UPLOAD TO START` 文案。
  - 上传成功后：
    - 显示 `#canvasWrapper`（固定 16:9）：
      - `#baseImage`：原图（object-fit: contain）。
      - `#overlayImage`：当前遮罩（悬停 / 点击 / 文本）。
      - `#fixedOverlay`：悬停模式下被点击“固定”的遮罩（只保留一个）。

- 右侧侧边栏 `.side-panel`：
  - 标题 `Mode`。
  - 三个模式按钮：
    - `悬停选取`（默认激活）。
    - `点击分割`。
    - `文字分割`。
  - `文字分割` 控制区：
    - 文本提示标签：**文本提示（英文逗号分隔）**。
    - `textarea#promptInput`：输入如 `car, wheel`。
    - `button#runTextButton`：运行文字分割。
  - 状态显示：
    - 标签：`状态`。
    - `div#status`：显示当前状态（如“等待上传图片…”、“ONNX 悬停分割坐标: (x, y)”）。

### 4.3 全屏 Loading

- `#loadingOverlay`：
  - 固定定位，覆盖全屏，带高斯模糊背景。
  - 内含：
    - `img.loading-icon`：使用 `icon/icon.png`，64x64，做上下浮动动画。
    - `div.loading-text`：显示加载提示文本。
  - 上传 + 预处理阶段显示，完成后自动关闭，期间不允许操作。

---

## 5. 前端逻辑：上传与自动预处理

### 5.1 上传流程

1. 点击 `上传图片` → 触发 `fileInput.click()`。
2. `fileInput.change`：
   - 读取文件为 data URL：`reader.readAsDataURL(file)`。
   - 将 `baseImage.src = dataUrl`。
   - 隐藏 placeholder，显示 canvasWrapper。
   - 显示全屏 loading（“正在上传并预处理图片，请稍候...”）。

### 5.2 调用后端

在 `reader.onload` 的异步回调中：

1. `POST /set_image`：
   - Body：
     ```json
     {
       "image_data": dataUrl,
       "model_path": "D:/HF_DATA/sam3" // 来自 modelPathInput
     }
     ```
   - 成功后保存 `imageId = json.image_id`。
   - 重置 ONNX 状态：`emb0/1/2`、`shape0/1/2`、`onnxReady`、`hoverUseOnnx` 等。

2. 预处理（自动）：
   - `await ensureDecoderSession()`：
     - 若 `decoderSession` 为空，从 `${serverInput}/sam3_decoder.onnx` 下载 ONNX 并创建 Session。
   - `POST /get_embeddings`：
     - Body：`{ "image_id": imageId }`。
     - 拿到 `embeddings`、`original_sizes`、`target_size`。
   - 将 embeddings 展平为 `Float32Array`：
     ```js
     emb0 = new Float32Array(flatten(e0));
     emb1 = new Float32Array(flatten(e1));
     emb2 = new Float32Array(flatten(e2));
     ```
   - 记录：
     ```js
     origH = original_sizes[0][0];
     origW = original_sizes[0][1];
     decoderTargetSize = target_size; // 通常为 1008
     ```
   - 标记预处理完成：
     ```js
     onnxReady = true;
     hoverUseOnnx = true;
     ```

3. 结束：
   - 关闭 loading 覆盖层。
   - 状态改为“图片上传并预处理完成，可以开始悬停选取。”。

> 注意：原来存在的“预处理（加速悬停）”按钮已经移除，预处理完全自动触发。

---

## 6. 坐标映射与遮罩对齐

### 6.1 鼠标 → 原图坐标（`getRelativeCoords`）

因为 `baseImage` 使用 `object-fit: contain`，固定在 16:9 画布中居中显示，会产生黑边；因此需要先计算显示区域，再反推原图坐标：

```js
function getRelativeCoords(event) {
  const rect = baseImage.getBoundingClientRect();
  const naturalW = baseImage.naturalWidth;
  const naturalH = baseImage.naturalHeight;
  if (!naturalW || !naturalH) {
    return { x: 0, y: 0 };
  }

  const scale = Math.min(rect.width / naturalW, rect.height / naturalH);
  const displayW = naturalW * scale;
  const displayH = naturalH * scale;
  const offsetX = (rect.width - displayW) / 2;
  const offsetY = (rect.height - displayH) / 2;

  let x = event.clientX - rect.left - offsetX;
  let y = event.clientY - rect.top - offsetY;

  x = Math.max(0, Math.min(displayW, x));
  y = Math.max(0, Math.min(displayH, y));

  const origX = x / scale;
  const origY = y / scale;

  return { x: Math.round(origX), y: Math.round(origY) };
}
```

这样，鼠标指向的位置与原图像素坐标一一对应，不受黑边影响。

### 6.2 原图坐标 → Decoder 坐标（悬停 ONNX）

ONNX decoder 的输入点坐标是基于 `target_size`（例如 1008）的空间：

```js
const sx = decoderTargetSize / origW;
const sy = decoderTargetSize / origH;
const nx = x * sx;
const ny = y * sy;

const normPoints = [[[[nx, ny]]]];
const normLabels = [[[1]]];
```

这些坐标连同 embeddings 一起喂给 ONNX Session。

---

## 7. 三种分割模式

### 7.1 悬停选取（Hover Selection）

- 激活条件：`currentMode === "hover" && onnxReady && hoverUseOnnx`。
- `baseImage` 的 `mousemove` 处理：
  - 调用 `getRelativeCoords(event)` 得到 `(x, y)`。
  - 用 `HOVER_MIN_DISTANCE` 限制最小移动距离，防止频繁请求。
  - 调用 `handleHoverOnnx(x, y)`。

#### 7.1.1 `handleHoverOnnx`

1. 检查 ONNX 状态和 embeddings 是否准备好。
2. 归一化坐标到 decoder：
   ```js
   const sx = decoderTargetSize / origW;
   const sy = decoderTargetSize / origH;
   const nx = x * sx;
   const ny = y * sy;
   ```
3. 调用 `decodePointONNX(normPoints, normLabels)`：
   - 构造 Tensor：
     - `image_embedding_s0/s1/s2` → `ort.Tensor("float32", emb0/1/2, shape0/1/2)`。
     - `input_points` → `ort.Tensor("float32", pts, [...])`。
     - `input_labels` → `ort.Tensor("int64", lbl, [...])`（使用 `BigInt64Array`）。
4. 得到 `pred_masks`（形状 `[1,1,1,H,W]`），取 logits 做二值化：
   - `bin[i] = data[i] > 0 ? 1 : 0;`
   - 做一次 3x3 膨胀，再做一次 3x3 腐蚀（闭运算）平滑边缘并填小洞。
5. 在小画布 `maskCanvas` 上绘制低分辨率 mask（蓝色 + alpha）。
6. 再将其拉伸到原图分辨率的 `fullCanvas`（`fullW = naturalWidth`，`fullH = naturalHeight`）。
7. 设置：
   ```js
   overlayImage.src = fullCanvas.toDataURL("image/png");
   overlayImage.style.display = "block";
   statusEl.textContent = `ONNX 悬停分割坐标: (${x}, ${y})`;
   ```

#### 7.1.2 悬停结果“固定”

- 在 Hover 模式下，点击 `baseImage`：
  - 如果当前没有悬停遮罩 → 忽略。
  - 否则：
    - 新建 `fixedCanvas`（大小 = 原图分辨率）。
    - 将当前 `overlayImage` 绘制到该画布上。
    - 从 alpha 通道构造二值 mask，并计算边缘：
      - 周围存在 0 的像素视为边缘。
      - 对边缘再做一层膨胀，形成 ~2px 的描边带。
    - 将边缘区域像素改为白色（RGBA=255）。
    - `fixedOverlay.src = fixedCanvas.toDataURL("image/png");`
    - 清空 `overlayImage`，只保留 `fixedOverlay`。
    - 文案：`已固定选区，坐标 (x, y)`。
  - 每次点击都会覆盖 `fixedOverlay`，因此 Hover 模式只保留一个固定选区。

### 7.2 点击分割（Click Segmentation）

- `baseImage.click` 下，当 `currentMode === "click"`：
  - 使用 `getRelativeCoords` 取 `(x, y)`。
  - 调用 `requestSegmentPoint(x, y)`。

简化后的 `requestSegmentPoint`：

```js
async function requestSegmentPoint(x, y) {
  const currentRequestId = ++lastRequestId;
  statusEl.textContent = `点击坐标: (${x}, ${y})，请求中...`;

  try {
    const res = await fetch(`${apiBase()}/segment_point`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: imageId, x, y }),
    });
    if (!res.ok) {
      statusEl.textContent = "当前点未检测到目标。";
      clearOverlay();
      return;
    }
    const json = await res.json();

    if (currentRequestId !== lastRequestId) {
      return; // 新请求已经发出，丢弃旧结果
    }

    overlayImage.src = json.image;
    overlayImage.style.display = "block";
    statusEl.textContent = `已更新选区，坐标 (${x}, ${y})`;
  } catch (err) {
    console.error(err);
    setServerOnline(false);
    statusEl.textContent = "请求失败，请检查后端。";
  }
}
```

### 7.3 文字分割（Text Segmentation）

- 模式切换到 text 时：
  - 显示 `textControls`。
  - 清空当前遮罩。
- 点击 “运行文字分割”：
  - 校验 `imageId` 和文本。
  - 调用：

```js
const res = await fetch(`${apiBase()}/segment_text`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ image_id: imageId, text }),
});
```

成功后：

```js
const json = await res.json();
overlayImage.src = json.image;
overlayImage.style.display = "block";
statusEl.textContent = "文字分割完成。";
```
---

## 9. 小结

当前版本的 SAM 3 Auto Selection 具备以下特性：

- 三种交互模式：
  - 悬停选取：基于浏览器端 ONNX decoder，实现毫秒级、流畅的掩膜预览，并支持点击固定选区。
  - 点击分割：精确的后端分割，带平滑白边。
  - 文字分割：基于文本提示的多目标分割，同样带白边。
- 自动预处理：
  - 上传图片后自动下载 decoder ONNX（若未加载）并提取 encoder 特征，用户无需额外操作。
- 高质量掩膜：
  - 后端使用形态学操作与白边描绘，消除小洞，边缘平滑。
  - 前端 Hover 路径也做闭运算与 2px 白边固定，实现与后端视觉一致的效果。
- 坐标严格对齐：
  - 通过对 `object-fit: contain` 的 letterbox 情况做精确的坐标反推，保证遮罩与鼠标落点一致，不随画布拉伸而偏移。

这份文档可以作为后续维护和扩展（例如增加多选区、橡皮擦、形状编辑、导出 mask 等高级功能）的基础说明。若后续有新的改动，可以在此基础上继续追加章节。

