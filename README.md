# A Comparative Study of Deep Learning Based Image Composition Methods for Image Stitching

This project provides a framework for a focused comparative analysis of deep learning-based image composition methods, specifically evaluating **NIS**, **UDIS**, and **UDIS++**. It supports two main evaluation workflows: one for the public **UDIS-D dataset** and one for a custom-captured **Beehive dataset**.

## Acknowledgements & Original Sources

This project builds directly on the work of the original authors of the models and datasets.

-   **NIS (Neural Image Stitching)**
    -   **GitHub:** [https://github.com/minshu-kim/Neural-Image-Stitching](https://github.com/minshu-kim/Neural-Image-Stitching)
    -   **Models:** The pre-trained `ihn.pth` and `NIS_blending.pth` should be placed in the `NIS/pretrained/` directory.

-   **UDIS & UDIS++ (Unsupervised Deep Image Stitching)**
    -   **GitHub:** [https://github.com/nie-lang/UnsupervisedDeepImageStitching](https://github.com/nie-lang/UnsupervisedDeepImageStitching) (UDIS)
    -   **GitHub:** [https://github.com/nie-lang/UDIS2](https://github.com/nie-lang/UDIS2) (UDIS++)
    -   **Models:** The pre-trained models (`model.ckpt-200000` for UDIS, `epoch050_model.pth` for UDIS++) should be placed in their corresponding project directories (`UDIS/` and `UDIS++/`).

### Datasets

-   **UDIS-D Dataset:** Can be downloaded from the official UDIS repository.
-   **Beehive Dataset:** The custom Beehive dataset is available for download from [Google Drive](https://drive.google.com/drive/folders/1v_GwWWWO9Ju3nm2is6biXaEqTynbWGau?usp=sharing).

## Project Structure Overview

```
.
├── README.md
│
├── WORKFLOW SCRIPTS
│   │
│   ├── Beehive Dataset Workflow
│   │   ├── run_beehive_batch_testing.sh  # (Recommended) Main entry point for Beehive
│   │   ├── run_beehive_methods.sh       # Processes a single Beehive pair
│   │   └── prepare_beehive_inputs.py    # Generates data from known translations
│   │
│   └── UDIS-D Dataset Workflow
│       ├── prepare_warp_homographies_udisd.sh  # Step 1: Generates ORB+FLANN warps
│       ├── run_udis_d_methods.sh            # Step 2: Runs all composition methods
│       └── prepare_udisd_inputs.py          # Helper to format data for UDIS/UDIS++
│
├── EVALUATION SCRIPTS
│   ├── evaluate_quantitative_metrics.py  # For NIQE, PIQE, BRISQUE
│   └── run_siqs_evaluation.py            # For LLM-based quantitative scores
│
├── DATA & MODELS
│   ├── beehive_dataset/          # Raw Beehive images
│   ├── UDIS-D/                   # Raw UDIS-D images
│   ├── NIS/                      # NIS model and code
│   ├── UDIS/                     # UDIS model and code
│   └── UDIS++/                   # UDIS++ model and code
│
├── requirements/               # Environment setup files
│   ├── nis_requirements.txt
│   ├── udis_env_requirements.txt
│   └── stitch_requirements.txt
│
├── RESULTS (Output)
│   ├── beehive_results/
│   └── udis_d_results/
│
└── processing_data/ (Intermediate files, can be ignored)
```

## Requirements & Setup

This project requires three separate Conda environments. Use the provided files to set them up:

```bash
# Example setup for the 'nis' environment
conda create --name nis python=3.6 -y
conda activate nis
pip install -r requirements/nis_requirements.txt
conda deactivate
```
Repeat this process for `udis_env` and `stitch`, using their corresponding requirements files.

## How to Run Experiments

This project supports two distinct workflows.

### Workflow 1: UDIS-D Dataset (Feature-Based Alignment)

This workflow uses a classic feature-matching approach (ORB+FLANN) to generate alignments, providing a baseline for comparing the composition methods.

**Step 1: Prepare Warps & Homographies**
This script detects features, calculates homographies, and generates warped images for the UDIS-D dataset.

```bash
# Prepare warps for image pairs 1 to 100 in UDIS-D dataset
./prepare_warp_homographies_udisd.sh --start 1 --end 100 --batch-size 10
```

**Step 2: Run Composition Methods**
This script takes the prepared warps and runs them through the NIS, UDIS, and UDIS++ composition networks.

```bash
# Run all methods on the first 100 pairs 
./run_udis_d_methods.sh --start 1 --end 100 --batch-size 10
```

### Workflow 2: Beehive Dataset (Known Translation Alignment)

This workflow uses precisely known translations from the custom Beehive dataset, allowing for a pure evaluation of the composition stage. The recommended way to run this is with the batch script.

```bash
# Run the full batch test with default settings
./run_beehive_batch_testing.sh
```
You can customize the run with flags like `--randomize` or `--scale 0.4`.

## How to Evaluate Results

Once you have generated results (in `results/udis_d_results` or `results/beehive_results`), you can evaluate them.

### 1. Quantitative Metrics (NIQE, PIQE, BRISQUE)

```bash
# Example for UDIS-D results
python evaluate_quantitative_metrics.py results/udis_d_results --dataset udis-d

# Example for Beehive results
python evaluate_quantitative_metrics.py results/beehive_results --dataset beehive
```

### 2. Qualitative Metrics (SIQS with GPT-4)

This requires a valid OpenAI API key.

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Example for UDIS-D results
python run_siqs_evaluation.py results/udis_d_results --dataset udis-d

# Example for Beehive results
python run_siqs_evaluation.py results/beehive_results --dataset beehive
```

## 📖 Citation
