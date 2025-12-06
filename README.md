# SAM 3 Auto Selection (Web Demo)

English | [中文说明](README_cn.md)

---

This project is a web demo for **SAM3** interactive segmentation with three modes:

- **Hover Selection** – real‑time hover segmentation accelerated by a **decoder‑only ONNX model** running in the browser.
- **Click Segmentation** – precise backend segmentation around a clicked point (PyTorch).
- **Text Segmentation** – segmentation driven by natural‑language prompts (e.g. `car, wheel`).

The backend runs the **SAM3 encoder (PyTorch)** once per image, and the frontend runs a **decoder‑only ONNX** model in the browser for fast hover segmentation.

For more detailed technical notes (architecture, APIs, ONNX export, coordinate math, etc.), see:

- `SAM3_auto_selection_summary.md`

---

## 1. Setup & Usage

### Step 1: Download the SAM3 model

1. **Official checkpoint (preferred)**

   - Hugging Face model card:  
     <https://huggingface.co/facebook/sam3>

   - Request access and download the model to a local folder, for example:

     ```text
     D:\HF_DATA\sam3
     ```

2. **If Hugging Face access is denied**

   - Download the same model from ModelScope instead:  
     <https://www.modelscope.cn/models/facebook/sam3/>
   - Place the files into a local directory as well (e.g. still `D:\HF_DATA\sam3`).

You will point the Web UI to this **local SAM3 model path** later.

### Step 2: Clone this repository

```bash
git clone https://github.com/Hasasasa/Sam3_AutoSelection.git
cd Sam3_AutoSelection
```

### Step 3: Create and populate the virtual environment

Use the one‑click script to create a virtual environment and install all dependencies:

```bash
setup_venv.bat
```

This will:

- Create a virtual environment in `.\venv`.
- Activate it.
- Install all packages listed in `requirements.txt`.

By default, the installed `torch` / `torchvision` work on **CPU**.  
If you want to use **GPU (CUDA)**, install the corresponding CUDA builds of `torch` and `torchvision` yourself inside the virtual environment, following the official PyTorch instructions: <https://pytorch.org/get-started/>.

Later, whenever you work on the project, activate the environment manually:

```bash
cd Sam3_AutoSelection
venv\Scripts\activate
```

### Step 4: Start the backend server

With the virtual environment activated:

```bash
python server.py
```

By default the API is served on:

- `http://0.0.0.0:8000` (so the frontend usually uses `http://localhost:8000`)

### Step 5: Use the Web UI

1. Open `web.html` in your browser (double‑click from Explorer or open via your editor).
2. At the top of the page, set:
   - **Backend URL** – usually `http://localhost:8000`.
   - **Sam3 Model Path** – the local path to your SAM3 model, e.g.:

     ```text
     D:/HF_DATA/sam3
     ```
3. Click **Upload Image**:
   - The app uploads the image to the backend.
   - Runs the SAM3 encoder once to compute features.
   - Downloads the decoder‑only ONNX model (if not already loaded).
   - Precomputes embeddings for hover segmentation.
   - If something fails, the **Retry Precompute** button next to “Upload Image” becomes enabled. After you fix the backend or model path, click it to re‑run the pipeline without re‑selecting the file.
4. Use the three modes:
   - **Hover Selection**
     - Move the mouse over the image to see ONNX masks in real time.
     - Click to “fix” the current mask; the fixed region gets a 2‑px white border.
   - **Click Segmentation**
     - Click a point; the backend returns a high‑quality mask with a white outline.
   - **Text Segmentation**
     - Enter prompts like `window,wheel` and click **Run Text Segmentation** to get text‑driven masks.

---

## 2. License

See `LICENSE` in this repository for licensing terms.
