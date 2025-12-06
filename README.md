# SAM 3 Auto Selection (Web Demo)

English | [中文说明](README_cn.md)

---

This project is a web demo for **SAM3** interactive segmentation with three modes:

- **Hover Selection** – real‑time hover segmentation accelerated by a **decoder‑only ONNX model** running in the browser.
- **Click Segmentation** – precise backend segmentation around a clicked point (PyTorch).
- **Text Segmentation** – segmentation driven by natural‑language prompts (e.g. `car, wheel`).

The system uses a **SAM3 encoder (PyTorch, backend)** together with a **decoder‑only ONNX model (browser)** to provide fast interaction and high‑quality masks.

For a deeper technical breakdown (architecture, APIs, ONNX export, coordinate math, etc.), see:

- `SAM3_auto_selection_summary.md`

---

## 1. Download the SAM3 Model

1. **Official checkpoint (preferred)**

   - Hugging Face model card:  
     <https://huggingface.co/facebook/sam3>

   - Request access and download the model to a local folder, e.g.:

     ```text
     D:\HF_DATA\sam3
     ```

2. **If access is denied**

   - You can also download the same model from ModelScope:

     <https://www.modelscope.cn/models/facebook/sam3/>

   - Place the downloaded files in a local directory (for example the same `D:\HF_DATA\sam3`).

You will later point the UI to this **local SAM3 model path**.

---

## 2. Project Structure

- `server.py` – FastAPI backend: image upload, point / text segmentation, embedding export, ONNX model serving.
- `web.html` – Frontend UI (HTML + CSS + JS) using `onnxruntime-web` for hover decoding.
- `sam3_decoder_onnx/export_sam3_decoder_onnx.py` – Export **decoder‑only** SAM3 ONNX (optionally quantized).
- `sam3_decoder_onnx/verify_sam3_decoder_onnx.py` – Verify ONNX decoder vs. PyTorch decoder.
- `SAM3_auto_selection_summary.md` – Detailed technical notes.

---

## 3. Export Decoder ONNX

From the repo root (`sam3-demo`):

```bash
python sam3_decoder_onnx/export_sam3_decoder_onnx.py ^
  --model-path "D:\HF_DATA\sam3" ^
  --out sam3_decoder_onnx/sam3_decoder.onnx ^
  --quant-out sam3_decoder_onnx/sam3_decoder_quant.onnx
```

If the defaults in the script already match your paths, simply run:

```bash
python sam3_decoder_onnx/export_sam3_decoder_onnx.py
```

---

## 4. Verify Decoder ONNX (Optional)

```bash
python sam3_decoder_onnx/verify_sam3_decoder_onnx.py ^
  --model-path "D:\HF_DATA\sam3" ^
  --onnx-path "sam3_decoder_onnx/sam3_decoder.onnx"

python sam3_decoder_onnx/verify_sam3_decoder_onnx.py ^
  --model-path "D:\HF_DATA\sam3" ^
  --onnx-path "sam3_decoder_onnx/sam3_decoder_quant.onnx"
```

The script prints mean / max differences between PyTorch and ONNX masks.

---

## 5. Create and Use the Virtual Environment

You can use the provided one‑click script to set up a virtual environment and install dependencies.

```bash
cd Sam3_AutoSelection

setup_venv.bat
```

This will:

- Create a virtual environment in `.\venv`.
- Activate it.
- Install all dependencies from `requirements.txt`.

After it finishes, activate the environment manually whenever you work on the project:

```bash
cd Sam3_AutoSelection
venv\Scripts\activate
```

---

## 6. Run Backend

```bash
python server.py
# serves API on http://0.0.0.0:8000
```

By default the frontend assumes:

- Backend URL: `http://localhost:8000`

You can change this in the header of `web.html`.

---

## 7. Use the Web UI

1. Open `web.html` in your browser.
2. At the top of the page, set:
   - **Backend URL** – usually `http://localhost:8000`.
   - **Sam3 Model Path** – the local path where you downloaded the SAM3 model, e.g.:

     ```text
     D:/HF_DATA/sam3
     ```
3. Click **Upload Image**:
   - The app uploads the image, runs the encoder once, downloads the decoder ONNX, and precomputes embeddings for hover.
   - If something fails, use **Retry Precompute** to re‑run the pipeline without re‑selecting the file.
4. Use the three modes:
   - **Hover Selection** – hover to see ONNX masks; click to fix the current mask with a white border.
   - **Click Segmentation** – click a point; backend returns a high‑quality mask with a white outline.
   - **Text Segmentation** – enter prompts like `window,wheel` and click **Run Text Segmentation**.

---

## 6. License

See `LICENSE` in this repo for licensing terms.
