import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import Sam3TrackerModel, Sam3TrackerProcessor

'''
    decoder ONNX 验证
    python sam3_decoder_onnx/verify_sam3_decoder_onnx.py \
    --model-path "D:\HF_DATA\sam3" \
    --onnx-path "sam3_decoder_onnx/sam3_decoder.onnx"
'''

def load_image(image_path: str | None) -> Image.Image:
    if image_path is not None:
        return Image.open(image_path).convert("RGB")

    images_dir = Path("images")
    if images_dir.is_dir():
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            files = list(images_dir.glob(ext))
            if files:
                return Image.open(files[0]).convert("RGB")

    print("[verify] No image provided/found, using a dummy black image 1024x768.")
    return Image.new("RGB", (1024, 768), color="black")


def main(
    model_path: str,
    onnx_path: str,
    image_path: str | None,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[verify] Loading Sam3TrackerModel from {model_path} on {device}...")

    model = Sam3TrackerModel.from_pretrained(model_path).to(device)
    processor = Sam3TrackerProcessor.from_pretrained(model_path)
    model.eval()

    img = load_image(image_path)
    w, h = img.size
    click_x, click_y = w // 2, h // 2
    print(f"[verify] Using click at image center: ({click_x}, {click_y})")

    inputs = processor(
        images=img,
        input_points=[[[[float(click_x), float(click_y)]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(device)
    input_points = inputs["input_points"].to(device)
    input_labels = inputs["input_labels"].to(device)

    with torch.no_grad():
        pt_outputs = model(
            pixel_values=pixel_values,
            input_points=input_points,
            input_labels=input_labels,
            multimask_output=False,
        )
        pt_masks = pt_outputs.pred_masks.detach().cpu()

        image_embeddings = model.get_image_embeddings(pixel_values)

    if len(image_embeddings) != 3:
        raise RuntimeError(
            f"Expected 3 image embeddings for this verify script, got {len(image_embeddings)}."
        )

    emb0 = image_embeddings[0].detach().cpu().numpy()
    emb1 = image_embeddings[1].detach().cpu().numpy()
    emb2 = image_embeddings[2].detach().cpu().numpy()

    np_points = input_points.detach().cpu().numpy()
    np_labels = input_labels.detach().cpu().numpy()

    print(f"[verify] Loading decoder ONNX from {onnx_path}...")
    # 如果本地没有 CUDA 版 onnxruntime，会自动回退到 CPUExecutionProvider
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # 根据实际的输入名字构造 feeds，避免因为多余的输入（比如 multimask_output）
    # 被优化掉而报 INVALID_ARGUMENT。
    feeds: dict[str, np.ndarray] = {}
    for inp in sess.get_inputs():
        name = inp.name
        if name == "image_embedding_s0":
            feeds[name] = emb0
        elif name == "image_embedding_s1":
            feeds[name] = emb1
        elif name == "image_embedding_s2":
            feeds[name] = emb2
        elif name == "input_points":
            feeds[name] = np_points
        elif name == "input_labels":
            feeds[name] = np_labels
        # 对于 input_masks / multimask_output，如果在图里是常量/被优化掉，就不会出现在 inputs 里，
        # 我们也就不用在 feeds 里传，保持默认值即可。

    print("[verify] Running ONNX decoder inference...")
    onnx_masks = sess.run(["pred_masks"], feeds)[0]
    onnx_masks_t = torch.from_numpy(onnx_masks)

    print(f"[verify] PyTorch masks shape: {tuple(pt_masks.shape)}")
    print(f"[verify] ONNX masks shape:    {tuple(onnx_masks_t.shape)}")

    if pt_masks.shape != onnx_masks_t.shape:
        print("[verify] Shape mismatch between PyTorch and ONNX outputs.")
        return

    diff = (pt_masks - onnx_masks_t).abs()
    print(f"[verify] Diff mean: {diff.mean().item():.6f}")
    print(f"[verify] Diff max:  {diff.max().item():.6f}")

    print("[verify] Done. If diff 很小，说明 decoder ONNX 行为基本一致，可以用来做前端悬停加速。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify SAM3 decoder-only ONNX against PyTorch model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=r"D:\HF_DATA\sam3",
        help="Local path or HF ID of the SAM3 tracker model.",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="sam3_decoder.onnx",
        help="Path to decoder-only ONNX (quantized or not).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to a test image. If omitted, will try images/ or use a dummy image.",
    )
    args = parser.parse_args()

    main(args.model_path, args.onnx_path, args.image)
