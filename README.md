content = """# 🚀 GPU Dual-Polarization SAR Enhancement Project

This project provides a **GPU-accelerated Python pipeline** for enhancing and visualizing dual-polarization SAR imagery using **VV and VH channels**.

It is designed for **Ubuntu with an NVIDIA GPU** and uses **CuPy** for high-performance processing.

---

## 📌 What it does

The pipeline:

- reads VV and VH TIFF SAR images  
- applies log scaling + robust normalization  
- performs GPU-based smoothing  
- computes feature maps:
  - VV + VH (intensity sum)  
  - |VV − VH| (polarimetric contrast)  
  - normalized difference  
  - local contrast  
  - anomaly map (NEW)  
- builds a pseudo-RGB visualization:
  - R = VV  
  - G = VH  
  - B = |VV − VH|  
- computes a final saliency map  
- exports readable PNGs + a printable PDF report  
- optionally benchmarks CPU vs GPU performance  

---

## ⭐ New Addition: Anomaly Map

A = max(0, S - blur(S))  
where S = VV + VH

This suppresses ocean clutter and enhances localized targets such as ships.

---

## 🧠 Method Summary

The final saliency map combines:

- VV + VH  
- |VV − VH|  
- normalized difference  
- local contrast  
- anomaly response  
- VH contribution  

---

## ⚠️ Important Scope Note

This is an enhancement pipeline, not a final ship detector.

---

## 📂 Project Files

- gpu_dualpol_sar_enhancement_updated.py  
- requirements.txt  
- run_example.sh  
- README.md  

---

## 📊 Outputs

Original:
- vv_original.png  
- vh_original.png  

Feature maps:
- sum_map.png  
- diff_map.png  
- norm_diff_map.png  
- local_contrast.png  
- anomaly_map.png  

Outputs:
- rgb_composite.png  
- saliency_map.png  
- saliency_heatmap.png  
- overlay.png  

Reports:
- report.pdf  
- timing_report.txt  

---

## ⚙️ Installation

python3 -m venv .venv  
source .venv/bin/activate  

pip install -r requirements.txt  
pip install cupy-cuda12x  
pip install rasterio  

---

## ▶️ Run

python3 gpu_dualpol_sar_enhancement_updated.py \\
  --vv /path/to/VV.tif \\
  --vh /path/to/VH.tif \\
  --outdir results  

---

## 🖨️ Print

lp results/report.pdf  

---

## 📡 Notes

- VV and VH must match in size  
- One band per file  
- Crop large scenes if needed  

---


