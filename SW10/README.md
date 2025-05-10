
# 🏗️ Multimodal AI Project — Week 10 (ML2 FS2025)

This repository contains notebooks and exercises for **Machine Learning II (Week 10)** on Multimodal AI.

## 📂 Contents

- `SW10_E1_VLM_Basics.ipynb`: Vision-Language Models basics
- `SW10_E2_Identify_Obj_Positions_startingpoint.ipynb`: Object detection comparison (YOLO, GPT-4, Gemini)
- `SW10_HandsOn_Object_Detection_Basics_YOLO_OWL_ViT.ipynb`: YOLO and OWL-ViT object detection

## 🚀 How to Run

1️⃣ Clone the repo  
```bash
git clone https://github.com/zhaw-iwi/MultimodalInteraction_ObjDet.git
```

2️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

3️⃣ Load `.env` file for API keys  
```bash
from dotenv import load_dotenv
load_dotenv()
```

4️⃣ Run notebooks (locally, Colab, LightningAI, or Codespaces)

---

## 💡 Main Tasks

✅ Run VLM models (CLIP, OWL-ViT)  
✅ Run YOLO for object detection  
✅ Compare results across methods  
✅ Integrate multimodal inputs (text, image, audio)

---

## 📦 Dependencies

- `torch`
- `transformers`
- `ultralytics`
- `opencv-python`
- `python-dotenv`

---

## 🛡️ Best Practices

- Use `.env` for sensitive keys  
- Check CUDA/GPU support  
- Validate input shapes  
- Set random seeds for reproducibility

---

## 📞 Contact

Maintainer: Dr. Elena Gavagnin  
Email: elena.gavagnin@zhaw.ch
