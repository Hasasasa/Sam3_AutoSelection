# SAM 3 Auto Selection 功能总结与技术亮点

这份文档用「偏白话」的方式，总结当前 SAM3 Auto Selection 项目的整体设计和关键技术点，方便以后自己回顾和继续扩展。

---

## 1. 整体架构：前后端分工很清晰

- **后端：`server.py`**
  - 基于 **FastAPI**。
  - 负责：
    - 加载 SAM3 系列模型；
    - 管理图片和缓存的特征；
    - 计算点击 / 文本分割结果；
    - 对外提供 HTTP 接口。

- **前端：`web.html` + `css/web.css`**
  - 纯 HTML + CSS + 原生 JS，无框架依赖。
  - 负责：
    - 页面布局（画布 + 侧边栏）；
    - 模式切换（悬停 / 点击 / 文本）；
    - 监听鼠标 / 输入交互；
    - 调用后端接口，并在 `<img>` 上叠加遮罩。
  - 使用 `onnxruntime-web` 在浏览器里直接跑 **Decoder-only ONNX 模型**，实现「悬停时本地快速解码」。

- **ONNX 导出与验证：`sam3_decoder_onnx` 目录**
  - `export_sam3_decoder_onnx.py`：从原始 PyTorch 模型中导出 **只包含 decoder 的 ONNX 模型**，可选量化。
  - `verify_sam3_decoder_onnx.py`：对比 PyTorch decoder 和 ONNX decoder 的输出误差，确保行为一致。

**高层运行流程（简化版）：**

1. 前端上传图片 → 调用后端 `POST /set_image`，后端保存图片并返回 `image_id`。
2. 前端再调用 `POST /get_embeddings`，让后端一次性算好多尺度图像特征（encoder），返回给前端。
3. 前端把这些特征喂给浏览器里的 ONNX decoder，用于 Hover（悬停）时快速解码。
4. **悬停分割**：完全在浏览器端用 ONNX decoder 完成，只复用后端算好的特征，不再重复跑 encoder。
5. **点击分割 / 文本分割**：仍然由后端用 PyTorch 完整执行，并做更重的后处理，返回带白边的 PNG 遮罩。

---

## 2. 后端实现（`server.py`）：专注「高质量分割」

### 2.1 使用的模型与设备选择

- 使用 HuggingFace 的：
  - `Sam3Model` + `Sam3Processor`：用于 **文本分割（PCS）**。
  - `Sam3TrackerModel` + `Sam3TrackerProcessor`：用于 **点分割 / 跟踪（PVS）**。
- 设备选择逻辑：
  - 如果有 `CUDA`：优先用 GPU。
  - 其次尝试 `MPS`（Apple）。
  - 否则退回 `CPU`。

加载完成后，会把模型和处理器，以及实际使用的 `device` 都保存起来，全局复用。

### 2.2 图片缓存结构：避免重复算 encoder

后端维护一个字典 `stored_images[image_id]`，典型内容包括：

- `image`: 原始 `PIL.Image` 对象。
- `image_embeddings`: 已预计算好的多尺度图像特征列表 `[e0, e1, e2]`，供 PVS 使用。
- `original_sizes`: 模型预处理前的原始高宽信息，用于把 mask 映射回原图大小。

有了这个缓存：

- 第一次上传时会真正算一遍 encoder 并缓存特征；
- 之后的点击、文本分割可以直接复用特征，明显减少延迟。

### 2.3 核心 API 接口

#### 2.3.1 `POST /set_image`

- 入参：
  - `image_data`: base64 的 `data:image/png;base64,...`。
  - `model_path`: 可选的本地权重路径，比如 `D:/HF_DATA/sam3`。
- 逻辑：
  - 解码为 `PIL.Image`；
  - 生成新的 `image_id` 并存入 `stored_images`；
  - 清空该 `image_id` 下旧的 `image_embeddings` / `original_sizes`（防止脏数据）。
- 返回：

```json
{ "image_id": "xxxx" }
```

#### 2.3.2 `POST /get_embeddings`

- 入参：`image_id`
- 逻辑：
  - 如果该 `image_id` 下尚未有 `image_embeddings`，则：
    - 使用 `Sam3TrackerProcessor` 做预处理；
    - 送入 `Sam3TrackerModel` 的 encoder，得到 3 个尺度的特征；
    - 同时记录 `original_sizes` 与 `target_size`；
    - 缓存到 `stored_images`。
  - 若已有缓存，则直接复用。
- 返回给前端：

```json
{
  "original_sizes": [[H, W]],
  "target_size": 1008,
  "embeddings": [e0_list, e1_list, e2_list]
}
```

前端会把这些 list 转成 `Float32Array`，作为浏览器端 ONNX decoder 的输入。

#### 2.3.3 `POST /segment_point`（点击分割）

- 入参：`image_id, x, y`（原图坐标）
- 两种路径：
  - **已有 encoder 结果（推荐）**：
    - 使用 `processor_pvs(images=None, input_points, input_labels, original_sizes)` 构造 decoder 输入；
    - 调用 `model_pvs` 只跑 decoder；
    - encoder 特征来自之前缓存的 `image_embeddings`，无需重复计算。
  - **无缓存（备用）**：
    - 每次传入 `images=image`；
    - 由处理器同时跑 encoder + decoder，开销较大。
- 解码输出：
  - 得到 `outputs.pred_masks`；
  - 通过 `post_process_masks(..., binarize=False)` 拿到 **logits** 形式的 mask（非单纯 0/1）。
  - 使用 `refine_mask_from_logits(mask_logits, target_size=(orig_h, orig_w))` 做精细后处理（见 2.4）。
  - 使用 `apply_mask_overlay(image, mask_np, color=(30,144,255), alpha=0.65)` 生成带白边的叠加图。
- 返回：

```json
{ "image": "data:image/png;base64,..." }
```

#### 2.3.4 `POST /segment_text`（文本分割）

- 入参：`image_id, text`，例如 `"window, wheel"`。
- 逻辑：
  - 将文本按逗号切分并去空，得到多个 prompt。
  - 对每个 prompt：
    - 用 `Sam3Processor` 构造文本 + 图像的输入；
    - 模型输出后，通过 `post_process_instance_segmentation` 得到多个实例的 mask；
    - 合并该 prompt 的所有 mask，随机生成一个颜色；
    - 调用 `apply_mask_overlay` 把这一类的区域叠加到同一张图片上。
  - 最终返回一张叠加了所有文本类别遮罩的图片。

---

### 2.4 关键后处理函数：高质量遮罩与白色描边

#### 2.4.1 `refine_mask_from_logits`

这个函数负责把模型输出的 **低分辨率 mask logits** 变成质量较高、干净的二值 mask：

1. 如果指定了目标尺寸 `target_size`，先用 `cv2.resize` 把 logits 上采样到原图大小。
2. 对 logits 施加 Sigmoid，得到 0～1 的概率图。
3. 用一个阈值（0.5 以上，可再加一个 `prob_threshold` ）做二值化，得到初始 mask。
4. 做一次「开运算」（小卷积核 + `cv2.MORPH_OPEN`）：
   - 断开细小粘连；
   - 去掉孤立小噪点。
5. 只保留最大的连通区域，进一步去除杂点。

整体目标是：**让 mask 尽量干净、整体连贯，而不是一堆噪点和毛刺**。

#### 2.4.2 `apply_mask_overlay`

这个函数负责把二值 mask 漂亮地画在原图上，带有半透明区域 + 白色轮廓线：

1. 把 `mask` 归一化为 0 和 1，保证不会出现 0～255 的奇怪值。
2. 若遮罩尺寸和原图不同，则用 `cv2.resize` 调整到一致。
3. 生成一个与图像同尺寸的纯色叠加层（默认蓝色 `RGB=(30,144,255)`）。
4. 用 `alpha` 混合得到半透明的遮罩区域。
5. 使用形态学梯度 `cv2.MORPH_GRADIENT` 找出 mask 边缘，并略微膨胀，得到约 2px 宽的边缘带。
6. 在这些边缘像素上直接画纯白色 `(255,255,255)`，形成非常清晰的轮廓线。

这一套后处理让点击 / 文本分割的结果在视觉上「干净、边界清楚、没有小洞」。

---

## 3. 前端实现（`web.html`）：三种模式 + 本地 ONNX

### 3.1 页面结构

- 左侧是画布区域：
  - `baseImage`: 原始图片；
  - `overlayImage`: 用于显示当前点击 / 文本分割的遮罩 PNG；
  - `fixedOverlay`: （可选）用于固定选区或其它特定用途。

- 右侧是侧边栏：
  - 服务器地址与模型路径输入；
  - 模式切换按钮（悬停 / 点击 / 文本）；
  - 文本输入框（在文本模式下显示）；
  - 状态提示区域。

### 3.2 模式管理

核心状态变量：

- `currentMode`：`"hover" | "click" | "text"`，控制当前交互模式。
- `imageId`：当前图片在后端的标识。
- `lastRequestId`：用于取消过时请求，避免返回结果乱序。

模式切换时会：

- 更新按钮的 `active` 样式；
- 重置 `lastRequestId`，停止旧的悬停请求；
- 清空当前遮罩；
- 文本模式下显示文本控制区。

---

## 4. 三种交互模式：Hover / Click / Text

### 4.1 悬停选取（Hover）

**技术亮点：浏览器端 ONNX decoder，本地毫秒级响应。**

- 在 `baseImage` 上监听 `mousemove`：
  - 若当前模式不是 `"hover"`，或图片尚未准备好，直接返回；
  - 若 ONNX decoder 或后端 embeddings 尚未就绪，也不触发。
- 使用 `getRelativeCoords(event)` 把鼠标相对画布的坐标统一到「原图空间」。
- 控制请求频率：
  - 通过 `lastHoverX / lastHoverY` 记录上一次悬停点；
  - 若两次距离小于 `HOVER_MIN_DISTANCE`（例如 10 像素），则不发新请求；
  - 使用 `hoverInFlight` 防止并发悬停请求。

处理流程（`handleHoverOnnx`）：

1. 根据原图尺寸和 decoder 的 `target_size` 计算缩放比例，把 `(x, y)` 映射到 decoder 空间。
2. 构造 ONNX 输入的 `normPoints` 和 `normLabels`。
3. 调用 `decodePointONNX`，用浏览器中的 `ort.InferenceSession` 进行解码，得到 `pred_masks`（logits）。
4. 对 logits 做深度后处理（关键亮点）：
   - 二值化得到粗 mask；
   - 只保留最大连通区域；
   - 调用一系列函数：`solidifyMask`、`fillHoles`、`erodeMask`、`dilateMask` 等，让目标变成实心、边缘平滑；
   - 生成 `internalCore`（绝对内部）和 `edgeZone`（边缘带），构造 `mixedLogits`：
     - 核心区域 logits 强行设为 `10.0`，确保是「100% 概率」；
     - 边缘区域保留原始 logits，如有负值则提到 2.0，避免断裂；
     - 外部背景设为 `-10.0`。
5. 把 `mixedLogits` 双线性插值放大到更高分辨率（比如最长边 800），并根据图像长宽比做等比例缩放。
6. 把放大后的 logits 转成 RGBA 图片：
   - 固定遮罩颜色 `(30,144,255)`；
   - 使用 Sigmoid 得到概率；
   - `prob < 0.4` 完全透明；
   - `prob ≥ 0.4` 按 `(prob - 0.4) / 0.4` 映射到 Alpha，最大不超过 165；
   - 形成柔和的「透明度渐变边缘」，不额外画白边。
7. 最后把这一张遮罩绘制到与原图一致大小的临时 canvas 上，再赋值给 `overlayImage.src`。

**效果：**

- 悬停时几乎实时响应；
- 遮罩中心颜色稳定，边缘平滑渐变；
- 不会因为 logits 噪声导致「锯齿感很强」。

### 4.2 点击分割（Click）

- 在 `baseImage` 上监听 `click` 事件：
  - 使用 `getRelativeCoords` 得到原图坐标 `(x, y)`；
  - 如果当前模式是 `"click"`，则调用 `requestSegmentPoint(x, y)`。

`requestSegmentPoint`：

1. 递增 `lastRequestId`，记录本次请求的 id，用于之后丢弃过时结果。
2. 调用后端 `POST /segment_point`，传入 `image_id, x, y`。
3. 如果返回 404，说明该点附近没有检测到目标，前端会清空遮罩并给出提示。
4. 成功时：
   - 如果发现 `currentRequestId !== lastRequestId`，说明有更新的请求已经发出，则丢弃旧结果；
   - 否则将 `json.image` 赋给 `overlayImage.src` 并显示。

**与 Hover 的区别：**

- 点击分割用的是后端的 PyTorch 模型 + `refine_mask_from_logits` + `apply_mask_overlay`。
- 遮罩边缘带有非常清晰的 **白色描边**，整体区域是固定 alpha 的半透明蓝色。
- Mask 质量更「严谨」一些，可以视为结果版，而 Hover 更偏预览版。

### 4.3 文本分割（Text）

- 在侧边栏选择「文字分割」模式时：
  - 展开文本输入区域；
  - 清空当前遮罩。
- 点击「运行文字分割」按钮：
  - 校验已上传图片且文本不为空；
  - 调用 `POST /segment_text`，传入 `image_id` 和文本内容（英文逗号分隔）。
- 后端返回叠加好的 PNG，前端用 `overlayImage` 显示。

文本模式的遮罩绘制逻辑与点击模式类似，也使用 `apply_mask_overlay` 保证边缘有白色描边、内部半透明。

---

## 5. 遮罩质量与边缘处理：Hover 和 Click 的区别

这一块是本项目的一个核心细节，也是之前重点优化的地方。

- **悬停 Hover：**
  - 完全在前端基于 ONNX decoder 的 logits 做处理；
  - 通过连通域筛选、填洞、腐蚀 / 膨胀，构造出平滑的 `mixedLogits`；
  - 依靠「透明度渐变」来表现边缘，**没有额外绘制白色描边**；
  - 更加适合做连续预览，边缘柔和、不生硬。

- **点击 / 文本分割：**
  - 在后端使用 PyTorch 输出 logits，并用 `refine_mask_from_logits` + `apply_mask_overlay` 做强力去噪；
  - 遮罩内部是固定 alpha 的纯蓝色；
  - 边缘使用形态学梯度提取，再画一圈 2px 左右的 **纯白轮廓线**；
  - 更适合作为「确定结果」展示，轮廓非常清晰。

简单理解：

- Hover 是「高刷新率的柔边预览」；
- Click / Text 是「低频调用但质量更高、轮廓更硬朗的最终结果」。

---

## 6. 性能与 ONNX 优化思路

项目在性能上的几个关键点：

- 图像 encoder 仅在上传后由后端计算一次，然后通过 `/get_embeddings` 把特征传给前端。
- 悬停时只在浏览器里跑 decoder-on-ONNX：
  - 避免频繁走网络；
  - 避免重复 encoder 的大算力消耗；
  - 保证悬停可以做到「几乎实时」。
- 使用固定 `decoderTargetSize`（例如 1024）控制 decoder 的输入大小，避免输入过大导致 ONNX 推理变慢。
- 前端对悬停点做距离阈值控制，防止鼠标微小抖动时触发太多解码请求。

---

## 7. 坐标与缩放：对齐鼠标和遮罩

前端需要在多种「显示比例」下保持坐标准确：

- 图片显示时通常会使用 `object-fit: contain`，会产生上下或左右的黑边（letterbox）。
- `getRelativeCoords` 会：
  - 考虑图片在容器中的实际 offset；
  - 把鼠标相对于 `<img>` 的坐标转换到原图坐标；
  - 确保同一个 `(x, y)` 在后端和前端看到的位置是一致的。

遮罩绘制阶段：

- Hover 路径会把 decoder 空间的 logits 通过 canvas 放大到「原图尺寸」，再叠加到 `baseImage` 上。
- 点击 / 文本分割由后端已经在原图空间完成，前端只需要把得到的 PNG 简单叠加即可。

---

## 8. 使用流程（面向最终用户）

1. 启动后端（`server.py`），浏览器自动打开 `web.html`。
2. 确认右上角服务器地址正确（默认 `http://localhost:8000`）。
3. 如需指定模型路径，可在「Sam3 模型地址」中填写本地路径。
4. 点击「上传图片」，选择一张需要分割的图片。
5. 等待状态显示「图片上传并预处理完成，可以开始悬停选取」：
   - 说明 encoder 已算完，ONNX decoder 也准备好了。
6. 选择不同模式进行交互：
   - **悬停选取**：移动鼠标查看实时预览遮罩；
   - **点击分割**：点击目标位置获取高质量结果遮罩；
   - **文字分割**：输入英文提示词（逗号分隔）后点击运行，获取多目标遮罩。

---

## 9. 小结与后续扩展方向

当前版本的 SAM 3 Auto Selection 具备以下特点：

- **三种交互模式：**
  - 悬停选取：基于浏览器端 ONNX decoder，实现毫秒级、连续的遮罩预览；
  - 点击分割：后端高质量分割，带平滑白边，适合作为最终结果；
  - 文字分割：支持多文本提示，多目标叠加展示。

- **自动预处理与缓存：**
  - 上传图片后自动下载 / 加载 ONNX 模型（如未加载）；
  - 自动预计算 encoder 特征并缓存，后续交互无需重复计算。

- **高质量遮罩：**
  - 后端通过形态学操作和只保留最大连通域，去掉噪点和小洞；
  - 前端 Hover 路径通过 logits「重混合」和透明度渐变，提供柔和而稳定的可视化效果。

- **坐标精确对齐：**
  - 考虑 `object-fit: contain` 的 letterbox 情况，保证鼠标位置与遮罩位置始终对齐。

这份文档可以作为后续维护和扩展（例如多选区、橡皮擦、形状编辑、导出 mask 文件等高级功能）的基础说明。若之后引入新的模式或改变遮罩风格，可以在对应章节继续追加说明。 

