import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Sam3TrackerModel, Sam3TrackerProcessor

try:
    import onnx
except Exception:  # noqa: BLE001
    onnx = None

'''
    进入虚拟环境后
    python sam3_decoder_onnx/export_sam3_decoder_onnx.py \
    --model-path "D:\HF_DATA\sam3" \
    --out sam3_decoder_onnx/sam3_decoder.onnx \
    --quant-out sam3_decoder_onnx/sam3_decoder_quant.onnx
'''

class Sam3DecoderOnly(torch.nn.Module):

    def __init__(self, model: Sam3TrackerModel) -> None:
        super().__init__()
        self.prompt_encoder = model.prompt_encoder
        self.mask_decoder = model.mask_decoder

        # Precompute positional embedding as buffer
        with torch.no_grad():
            pos = model.get_image_wide_positional_embeddings()
        self.register_buffer("image_positional_embeddings", pos, persistent=False)

    def forward(  # type: ignore[override]
        self,
        image_embedding_s0: torch.FloatTensor,
        image_embedding_s1: torch.FloatTensor,
        image_embedding_s2: torch.FloatTensor,
        input_points: torch.FloatTensor,
        input_labels: torch.LongTensor,
        input_masks: torch.FloatTensor | None = None,
        multimask_output: bool = False,
    ) -> torch.FloatTensor:
        batch_size = image_embedding_s2.shape[0]

        image_embeddings = [
            image_embedding_s0,
            image_embedding_s1,
            image_embedding_s2,
        ]

        image_positional_embeddings = self.image_positional_embeddings.repeat(
            batch_size, 1, 1, 1
        )

        if input_masks is not None:
            if input_masks.shape[-2:] != self.prompt_encoder.mask_input_size:
                input_masks = F.interpolate(
                    input_masks.float(),
                    size=self.prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,
                ).to(input_masks.dtype)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=None,
            input_masks=input_masks,
        )

        low_res_multimasks, iou_scores, _, object_score_logits = self.mask_decoder(
            image_embeddings=image_embeddings[-1],
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            high_resolution_features=image_embeddings[:-1],
            attention_similarity=None,
            target_embedding=None,
        )

        return low_res_multimasks


def export_decoder_onnx(
    model_path: str,
    onnx_out: str,
    opset: int = 17,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Decoder-ONNX] Loading Sam3TrackerModel from {model_path} on {device}...")

    model = Sam3TrackerModel.from_pretrained(model_path).to(device)
    processor = Sam3TrackerProcessor.from_pretrained(model_path)

    decoder = Sam3DecoderOnly(model).to(device)
    decoder.eval()

    # Dummy image & click to trace shapes
    dummy_image = Image.new("RGB", (1024, 768), color="black")
    inputs = processor(
        images=dummy_image,
        input_points=[[[[512.0, 384.0]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"].to(device)
    input_points = inputs["input_points"].to(device)
    input_labels = inputs["input_labels"].to(device)

    # Use model helper to get image embeddings list
    with torch.no_grad():
        image_embeddings = model.get_image_embeddings(pixel_values)

    if len(image_embeddings) != 3:
        raise RuntimeError(
            f"Expected 3 image embeddings for this script, but got {len(image_embeddings)}."
        )

    image_embedding_s0 = image_embeddings[0]
    image_embedding_s1 = image_embeddings[1]
    image_embedding_s2 = image_embeddings[2]

    print("[Decoder-ONNX] Exporting ONNX decoder-only model...")
    torch.onnx.export(
        decoder,
        (
            image_embedding_s0,
            image_embedding_s1,
            image_embedding_s2,
            input_points,
            input_labels,
            None,
            False,
        ),
        onnx_out,
        input_names=[
            "image_embedding_s0",
            "image_embedding_s1",
            "image_embedding_s2",
            "input_points",
            "input_labels",
            "input_masks",
            "multimask_output",
        ],
        output_names=["pred_masks"],
        opset_version=opset,
        dynamic_axes={
            "image_embedding_s0": {0: "batch"},
            "image_embedding_s1": {0: "batch"},
            "image_embedding_s2": {0: "batch"},
            "input_points": {0: "batch", 1: "point_batch", 2: "num_points"},
            "input_labels": {0: "batch", 1: "point_batch", 2: "num_points"},
            "pred_masks": {0: "batch", 1: "point_batch"},
        },
        do_constant_folding=True,
    )

    print(f"[Decoder-ONNX] Exported decoder ONNX to: {onnx_out}")

    # 尝试把可能产生的 external data 合并回单文件，方便前端使用
    if onnx is not None:
        try:
            print("[Decoder-ONNX] Merging external data into single ONNX file...")
            model = onnx.load(onnx_out, load_external_data=True)
            onnx.save_model(model, onnx_out, save_as_external_data=False)
            print("[Decoder-ONNX] Merge success, model is now single-file.")
        except TypeError:
            # 旧版本 onnx 没有 save_as_external_data，这种情况就保持原样
            print(
                "[Decoder-ONNX] onnx.save_model 不支持 save_as_external_data，"
                "保留 .onnx + .onnx.data 形式。"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Decoder-ONNX] Failed to merge external data: {exc}")


def quantize_decoder_onnx(onnx_in: str, onnx_out: str) -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except Exception:  # noqa: BLE001
        print(
            "[Decoder-ONNX] onnxruntime and onnxruntime-tools are required for quantization.\n"
            "Install with: pip install onnxruntime onnxruntime-tools"
        )
        return

    print(f"[Decoder-ONNX] Quantizing {onnx_in} -> {onnx_out} (dynamic int8 weights)...")
    quantize_dynamic(
        onnx_in,
        onnx_out,
        weight_type=QuantType.QInt8,
    )
    print("[Decoder-ONNX] Quantization finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SAM3 decoder-only ONNX (and optional quantized).")
    parser.add_argument(
        "--model-path",
        type=str,
        default=r"D:\HF_DATA\sam3",
        help="Local path or HF ID of the SAM3 tracker model.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="sam3_decoder.onnx",
        help="Output decoder ONNX file path.",
    )
    parser.add_argument(
        "--quant-out",
        type=str,
        default="sam3_decoder_quant.onnx",
        help="Output path for quantized decoder ONNX.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version to use.",
    )
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Skip quantization step (only export plain ONNX).",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_decoder_onnx(args.model_path, str(out_path), opset=args.opset)

    if not args.no_quant:
        quant_out_path = Path(args.quant_out)
        quant_out_path.parent.mkdir(parents=True, exist_ok=True)
        quantize_decoder_onnx(str(out_path), str(quant_out_path))
