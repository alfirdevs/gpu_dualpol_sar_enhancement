# GPU Dual-Polarization SAR Enhancement Project

This project provides a GPU-accelerated Python pipeline for enhancing and visualizing dual-polarization SAR imagery using VV and VH channels.

It is designed for Ubuntu with an NVIDIA GPU and uses CuPy for acceleration.

## What it does

The pipeline:
- reads a VV TIFF and a VH TIFF
- applies log scaling and robust normalization
- smooths both channels on the GPU
- computes feature maps:
  - VV + VH
  - |VV - VH|
  - normalized difference
  - local contrast
- builds a pseudo-RGB image using:
  - R = VV
  - G = VH
  - B = |VV - VH|
- computes a final saliency map
- exports readable PNG files and a printable PDF report
- optionally benchmarks CPU vs GPU

## Important scope note

This is an enhancement and saliency visualization pipeline.
It is **not** a final validated ship detector.

Because land can produce strong responses too, this version does **not** produce binary detections or boxes.

## Project files

- `gpu_dualpol_sar_enhancement.py` — main GPU script
- `requirements.txt` — Python dependencies
- `run_example.sh` — helper shell script
- `README.md` — this file

## Outputs

The script saves:
- `vv.png`
- `vh.png`
- `rgb_composite.png`
- `sum_map.png`
- `diff_map.png`
- `norm_diff_map.png`
- `saliency_map.png`
- `saliency_heatmap.png`
- `overlay.png`
- `report.pdf`
- `timing_report.txt`

## Installation on Ubuntu

Create and activate a virtual environment if you want:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install base packages:

```bash
pip install -r requirements.txt
```

Install CuPy matching your CUDA version. Example for CUDA 12:

```bash
pip install cupy-cuda12x
```

Optional for some GeoTIFF files:

```bash
pip install rasterio
```

## Run

```bash
python3 gpu_dualpol_sar_enhancement.py \
  --vv /path/to/VV.tif \
  --vh /path/to/VH.tif \
  --outdir results
```

Optional CPU benchmark:

```bash
python3 gpu_dualpol_sar_enhancement.py \
  --vv /path/to/VV.tif \
  --vh /path/to/VH.tif \
  --outdir results \
  --benchmark-cpu
```

## Example helper script

Edit `run_example.sh` and set your file paths, then run:

```bash
bash run_example.sh
```

## Printing on Ubuntu

Once the run completes, print the PDF report with:

```bash
lp results/report.pdf
```

## Notes on input data

- VV and VH must have the same shape.
- The code assumes one band per file.
- For large Sentinel-1 scenes, pre-cropping to the area of interest is often helpful.
- If you later want ship-oriented candidate extraction, the right next step is adding a water/land mask before thresholding.
