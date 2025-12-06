import base64
import io
import uuid
import uvicorn
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import webbrowser
import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from fastapi.responses import FileResponse
from transformers import (
    Sam3Processor,
    Sam3Model,
    Sam3TrackerProcessor,
    Sam3TrackerModel,
)


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(model_path: str, device_choice: str):
    """
    加载 SAM 3 的 PCS（文本）和 PVS（点/视觉）两套模型，
    """
    device = device_choice or get_default_device()
    device = torch.device(device)

    print(f"[Server] Loading SAM3 models from {model_path} to {device}...")

    # PCS：用于文字分割
    model_pcs = Sam3Model.from_pretrained(model_path).to(device)
    processor_pcs = Sam3Processor.from_pretrained(model_path)

    # PVS：用于点分割 / 悬停 / 点击
    model_pvs = Sam3TrackerModel.from_pretrained(model_path).to(device)
    processor_pvs = Sam3TrackerProcessor.from_pretrained(model_path)

    status = f"Models loaded successfully on {device}."
    print("[Server]", status)
    return model_pcs, processor_pcs, model_pvs, processor_pvs, device, status


def apply_mask_overlay(
    image: Image.Image | np.ndarray,
    mask: np.ndarray,
    color=(30, 144, 255),
    alpha: float = 0.65,
    border_color=(255, 255, 255),
    border_width: int = 2,
) -> Image.Image:
    """
    在原图上叠加半透明遮罩，并在选区边缘绘制一圈白色描边。
    image: PIL.Image 或 np.ndarray (H, W, 3)
    mask : np.ndarray (H, W), 值为 0/1
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image).astype(np.float32)
        pil_out = True
    else:
        img_np = image.astype(np.float32)
        pil_out = False

    if mask.dtype != np.uint8:
        mask_np = mask.astype(np.uint8)
    else:
        mask_np = mask

    if mask_np.ndim == 3:
        mask_np = mask_np[0]

    h, w = mask_np.shape
    if img_np.shape[:2] != (h, w):
        # 尽量对齐尺寸
        img_np = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_LINEAR)

    # 半透明颜色遮罩
    overlay = np.zeros_like(img_np)
    overlay[..., 0] = color[2]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[0]

    mask_3c = mask_np[..., None].astype(np.float32)
    out = img_np * (1.0 - alpha * mask_3c) + overlay * (alpha * mask_3c)
    out = np.clip(out, 0, 255).astype(np.uint8)

    # 用形态学梯度在 mask 上取一圈边缘，然后画白线
    if border_width > 0:
        k = border_width * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        grad = cv2.morphologyEx(mask_np, cv2.MORPH_GRADIENT, kernel)
        edge = grad.astype(bool)
        if edge.any():
            out[edge] = np.array(border_color, dtype=np.uint8)

    if pil_out:
        return Image.fromarray(out)
    return Image.fromarray(out)



def refine_mask_from_logits(mask_logits: np.ndarray, prob_threshold: float = 0.45) -> np.ndarray:
    """
    从概率 / logit mask 生成平滑、无小洞的二值 mask。
    mask_logits: (H, W) float32, 值约在 [0,1] 或 logits。
    """
    mask = mask_logits.astype(np.float32)

    # 如果是 logits，先过 sigmoid；简单判断范围
    if mask.max() > 1.5 or mask.min() < -0.5:
        mask = 1.0 / (1.0 + np.exp(-mask))

    # 轻微高斯平滑，消掉噪点和锯齿
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # 用略高一点的阈值二值化
    _, mask_bin = cv2.threshold(
        mask, prob_threshold, 1.0, cv2.THRESH_BINARY
    )
    mask_bin = mask_bin.astype(np.uint8)

    h, w = mask_bin.shape
    if h * w == 0:
        return mask_bin

    # 形态学：先闭再开，平滑边缘、填小缝
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    # 只保留面积较大的连通域，去掉孤立小块
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bin, connectivity=8
    )
    if num_labels > 1:
        min_area = h * w * 0.002  # 0.2% 作为下限
        keep = np.zeros_like(mask_bin)
        for lab in range(1, num_labels):
            if stats[lab, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == lab] = 1
        mask_bin = keep

    # flood fill 填内部小洞
    flood = (mask_bin * 255).astype(np.uint8)
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask_ff, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    mask_bin = cv2.bitwise_or(flood_inv, mask_bin * 255) // 255

    # 轻微膨胀一圈，把边缘补齐
    mask_bin = cv2.dilate(mask_bin.astype(np.uint8), kernel, iterations=1)

    return mask_bin


app = FastAPI(title="SAM3 Hover Auto Selection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelHolder:
    def __init__(self) -> None:
        self.model_pcs = None
        self.processor_pcs = None
        self.model_pvs = None
        self.processor_pvs = None
        self.device = None

    def ensure_loaded(self, model_path: str) -> None:
        """
        Ensure SAM3 models are loaded from the given model_path.
        If模型已经从同一路径加载过，就直接复用；如果路径变化，则重新加载。
        """
        # 如果已经加载过并且路径相同，不必重复加载
        if getattr(self, "model_path", None) == model_path and self.model_pvs is not None:
            return

        # 记录当前路径并重新加载（允许用户在前端修改模型目录）
        self.model_path = model_path
        (
            self.model_pcs,
            self.processor_pcs,
            self.model_pvs,
            self.processor_pvs,
            self.device,
            _,
        ) = load_models(model_path=model_path, device_choice=get_default_device())
        if self.model_pvs is None:
            raise RuntimeError("Failed to load SAM3 models")


models = ModelHolder()


@dataclass
class StoredImage:
    image: Image.Image
    original_sizes: List[List[int]]
    image_embeddings: Optional[List[torch.Tensor]] = None


stored_images: Dict[str, StoredImage] = {}


def pil_from_base64(data: str) -> Image.Image:
    try:
        header, _, encoded = data.partition(",")
        if not encoded:
            encoded = header
        image_bytes = base64.b64decode(encoded)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid image data") from exc


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class SetImageRequest(BaseModel):
    image_data: str
    model_path: Optional[str] = None


class SegmentPointRequest(BaseModel):
    image_id: str
    x: int
    y: int


class SegmentTextRequest(BaseModel):
    image_id: str
    text: str


class PrecomputeRequest(BaseModel):
    image_id: str
    model_path: Optional[str] = None


class EncodePointRequest(BaseModel):
    image_id: str
    x: int
    y: int


class GetEmbeddingsRequest(BaseModel):
    image_id: str


@app.get("/health")
def health() -> Dict[str, str]:
    """
    Simple health check endpoint so that external tools
    polling /health do not get 404.
    """
    return {"status": "ok"}


@app.post("/set_image")
def set_image(req: SetImageRequest) -> Dict[str, str]:
    """
    Register an image on the server and return an image_id.
    image_data should be a base64 data URL or raw base64 string.
    """
    model_path = req.model_path or "D:/HF_DATA/sam3"
    models.ensure_loaded(model_path)
    image = pil_from_base64(req.image_data)

    # 初始只存图像本身，不做耗时预处理
    image_id = str(uuid.uuid4())
    stored_images[image_id] = StoredImage(image=image, original_sizes=[], image_embeddings=None)
    return {"image_id": image_id}


@app.post("/precompute_image")
def precompute_image(req: PrecomputeRequest) -> Dict[str, str]:
    """
    Run preprocessing on a stored image to compute image embeddings once.
    This speeds up hover / click selection for subsequent requests.
    """
    model_path = req.model_path or "D:/HF_DATA/sam3"
    models.ensure_loaded(model_path)

    if req.image_id not in stored_images:
        raise HTTPException(status_code=404, detail="Unknown image_id")

    stored = stored_images[req.image_id]

    encoding = models.processor_pvs(images=stored.image, return_tensors="pt")
    pixel_values = encoding["pixel_values"].to(models.device)
    original_sizes_tensor = encoding["original_sizes"]
    if isinstance(original_sizes_tensor, torch.Tensor):
        original_sizes = original_sizes_tensor.cpu().tolist()
    else:
        original_sizes = original_sizes_tensor

    with torch.no_grad():
        image_embeddings = models.model_pvs.get_image_embeddings(pixel_values)

    stored.original_sizes = original_sizes
    stored.image_embeddings = image_embeddings

    return {"status": "precomputed"}


@app.get("/sam3_decoder.onnx")
def download_decoder() -> FileResponse:
    """
    Serve the decoder-only ONNX model for browser-side onnxruntime-web.
    """
    from pathlib import Path

    # 模型文件放在 sam3_decoder_onnx 子目录下，
    # 使用当前文件所在目录作为基准，避免受启动目录影响。
    base_dir = Path(__file__).resolve().parent
    onnx_path = base_dir / "sam3_decoder_onnx" / "sam3_decoder.onnx"
    if not onnx_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="sam3_decoder.onnx not found in sam3_decoder_onnx directory",
        )
    return FileResponse(
        path=str(onnx_path),
        media_type="application/octet-stream",
        filename="sam3_decoder.onnx",
    )


@app.post("/get_embeddings")
def get_embeddings(req: GetEmbeddingsRequest) -> Dict[str, object]:
    """
    Return precomputed image embeddings and original_sizes for a given image_id.

    This is intended for browser-side ONNX decoding. The response can be large
    (~20MB) but只在点击“预处理”时调用一次。
    """
    if req.image_id not in stored_images:
        raise HTTPException(status_code=404, detail="Unknown image_id")

    stored = stored_images[req.image_id]

    if stored.image_embeddings is None or not stored.original_sizes:
        encoding = models.processor_pvs(images=stored.image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].to(models.device)
        original_sizes_tensor = encoding["original_sizes"]
        if isinstance(original_sizes_tensor, torch.Tensor):
            stored.original_sizes = original_sizes_tensor.cpu().tolist()
        else:
            stored.original_sizes = original_sizes_tensor

        with torch.no_grad():
            stored.image_embeddings = models.model_pvs.get_image_embeddings(pixel_values)

    emb_list = stored.image_embeddings
    if emb_list is None or len(emb_list) != 3:
        raise HTTPException(status_code=500, detail="Unexpected number of image embeddings")

    # 返回 original_sizes + tracker 的 target_size，方便前端自行归一化坐标
    return {
        "original_sizes": stored.original_sizes,
        "target_size": models.processor_pvs.target_size,
        "embeddings": [e.detach().cpu().numpy().tolist() for e in emb_list],
    }


@app.post("/encode_point")
def encode_point(req: EncodePointRequest) -> Dict[str, object]:
    """
    Normalize a click point into model input_points/input_labels using the
    official processor logic. This is了便浏览器端直接喂给 ONNX decoder。
    """
    if req.image_id not in stored_images:
        raise HTTPException(status_code=404, detail="Unknown image_id")

    stored = stored_images[req.image_id]

    if not stored.original_sizes:
        encoding = models.processor_pvs(images=stored.image, return_tensors="pt")
        original_sizes_tensor = encoding["original_sizes"]
        if isinstance(original_sizes_tensor, torch.Tensor):
            stored.original_sizes = original_sizes_tensor.cpu().tolist()
        else:
            stored.original_sizes = original_sizes_tensor

    input_points = [[[[req.x, req.y]]]]
    input_labels = [[[1]]]

    inputs = models.processor_pvs(
        images=None,
        input_points=input_points,
        input_labels=input_labels,
        original_sizes=stored.original_sizes,
        return_tensors="pt",
    )

    ip = inputs["input_points"].detach().cpu().numpy().tolist()
    il = inputs["input_labels"].detach().cpu().numpy().tolist()

    return {"input_points": ip, "input_labels": il}


@app.post("/segment_point")
def segment_point(req: SegmentPointRequest) -> Dict[str, str]:
    """
    Segment object around (x, y) for a previously uploaded image.
    Used for both hover and click selection.
    """
    if req.image_id not in stored_images:
        raise HTTPException(status_code=404, detail="Unknown image_id")

    stored = stored_images[req.image_id]
    image = stored.image

    input_points = [[[[req.x, req.y]]]]
    input_labels = [[[1]]]

    # 如果已经预处理过，走快速路径；否则走原始完整路径
    if stored.image_embeddings is not None and stored.original_sizes:
        inputs = models.processor_pvs(
            images=None,
            input_points=input_points,
            input_labels=input_labels,
            original_sizes=stored.original_sizes,
            return_tensors="pt",
        )
        inputs = {k: (v.to(models.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = models.model_pvs(
                input_points=inputs.get("input_points"),
                input_labels=inputs.get("input_labels"),
                image_embeddings=stored.image_embeddings,
                multimask_output=False,
            )

        masks = models.processor_pvs.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs.get("original_sizes").tolist(),
            binarize=False,   # 关键：拿概率 mask
        )[0]
    else:
        # 未预处理：每次完整计算一次（较慢，但不依赖预处理按钮）
        inputs = models.processor_pvs(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(models.device)

        with torch.no_grad():
            outputs = models.model_pvs(**inputs, multimask_output=False)

        masks = models.processor_pvs.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs.get("original_sizes").tolist(),
            binarize=False,   # 同样拿概率 mask
        )[0]

    if masks is None or masks.numel() == 0:
        raise HTTPException(status_code=404, detail="No object detected at this point")

    # masks: [1, 1, H, W] 概率 / logits
    mask_logits = masks[0, 0].cpu().numpy().astype(np.float32)
    mask_np = refine_mask_from_logits(mask_logits)

    result_image = apply_mask_overlay(image, mask_np, color=(30, 144, 255), alpha=0.65)
    return {"image": image_to_base64(result_image)}


@app.post("/segment_text")
def segment_text(req: SegmentTextRequest) -> Dict[str, str]:
    """
    Text prompt based segmentation on the stored image.
    """
    if models.model_pcs is None or models.processor_pcs is None:
        raise HTTPException(status_code=503, detail="Text model not loaded")

    if req.image_id not in stored_images:
        raise HTTPException(status_code=404, detail="Unknown image_id")

    stored = stored_images[req.image_id]
    image = stored.image
    prompts = [p.strip() for p in req.text.split(",") if p.strip()]
    if not prompts:
        raise HTTPException(status_code=400, detail="Empty text prompt")

    overlay_img = np.array(image).copy()

    for idx, prompt in enumerate(prompts):
        inputs = models.processor_pcs(
            images=image, text=prompt, return_tensors="pt"
        ).to(models.device)
        with torch.no_grad():
            outputs = models.model_pcs(**inputs)

        results = models.processor_pcs.post_process_instance_segmentation(
            outputs, threshold=0.4, target_sizes=[image.size[::-1]]
        )[0]

        masks = results.get("masks")
        if masks is None or len(masks) == 0:
            continue

        np.random.seed(idx)
        color = np.random.randint(50, 255, 3).tolist()
        combined_mask = np.any(masks.cpu().numpy(), axis=0)
        overlay_img = np.array(
            apply_mask_overlay(overlay_img, combined_mask, color=color)
        )

    result_image = Image.fromarray(overlay_img)
    return {"image": image_to_base64(result_image)}


if __name__ == "__main__":
    
    html_file_path = "web.html"
    webbrowser.open(f'file:///{os.path.abspath(html_file_path)}')
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
