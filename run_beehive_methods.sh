#!/bin/bash

# Beehive Dataset Evaluation Script
# Processes beehive image pairs using NIS, UDIS, and UDIS++ methods
# Uses translational homographies for standardized evaluation

# Check arguments
if [ $# -ne 8 ]; then
    echo "Usage: $0 <ref_scan> <ref_img> <tgt_scan> <tgt_img> <methods> <dx_original> <dy_original> <scale_factor>"
    echo "  ref_scan: Reference scan number (1 or 2)"
    echo "  ref_img: Reference image number (1 or 2)"
    echo "  tgt_scan: Target scan number (1 or 2)"
    echo "  tgt_img: Target image number (1 or 2)" 
    echo "  methods: Comma-separated list of methods or 'all' (nis,udis,udis_plus_plus)"
    echo "  dx_original: Original translation in x direction"
    echo "  dy_original: Original translation in y direction"
    echo "  scale_factor: Scale factor for processing (e.g., 0.4)"
    echo ""
    echo "Example: $0 1 1 1 2 all 490 330 0.4"
    echo ""
    echo "SPECIAL FEATURES:"
    echo "  - NIS image order fixed to match UDIS/UDIS++ visual results"
    echo "  - NIS seam cutting DISABLED for comparison purposes"
    exit 1
fi

# Parse arguments
REF_SCAN=$1
REF_IMG=$2
TGT_SCAN=$3
TGT_IMG=$4
METHODS=$5
DX_ORIGINAL=$6
DY_ORIGINAL=$7
SCALE_FACTOR=$8

# Convert comma-separated methods to array
if [ "$METHODS" = "all" ]; then
    METHODS_ARRAY=("nis" "udis" "udis_plus_plus")
else
    IFS=',' read -ra METHODS_ARRAY <<< "$METHODS"
fi

# Project configuration
PROJECT_ROOT=$(pwd)
BEEHIVE_DATASET_DIR="${PROJECT_ROOT}/beehive_dataset/2024-12-19 11-19-05_716"

# Results directories
BEEHIVE_RESULTS_DIR="${PROJECT_ROOT}/results/beehive_results"
NIS_BEEHIVE_RESULTS_DIR="${BEEHIVE_RESULTS_DIR}/nis"
UDIS_BEEHIVE_RESULTS_DIR="${BEEHIVE_RESULTS_DIR}/udis"
UDIS_PLUS_PLUS_BEEHIVE_RESULTS_DIR="${BEEHIVE_RESULTS_DIR}/udis_plus_plus"

# Shared data directories
SHARED_BEEHIVE_DATA_DIR="${PROJECT_ROOT}/processing_data/shared_beehive_data"
BEEHIVE_PREPROCESSED_DATA_DIR="${PROJECT_ROOT}/processing_data/beehive_preprocessed_data"

# For UDIS/UDIS++ compatibility
SIFT_PREPROCESSED_DATA_DIR="${PROJECT_ROOT}/processing_data/sift_preprocessed_data"
WARP1_DIR="${SIFT_PREPROCESSED_DATA_DIR}/warp1"
WARP2_DIR="${SIFT_PREPROCESSED_DATA_DIR}/warp2"
MASK1_DIR="${SIFT_PREPROCESSED_DATA_DIR}/mask1"
MASK2_DIR="${SIFT_PREPROCESSED_DATA_DIR}/mask2"

# Create result directories
mkdir -p "${NIS_BEEHIVE_RESULTS_DIR}"
mkdir -p "${UDIS_BEEHIVE_RESULTS_DIR}"
mkdir -p "${UDIS_PLUS_PLUS_BEEHIVE_RESULTS_DIR}"

# Conda environment
CONDA_ENV_NAME="nis"

# Timeouts and limits
HOMOGRAPHY_TIMEOUT=30
PROCESS_TIMEOUT=30
MAX_CANVAS_RATIO=1.8

# Calculate scaled translations (using awk instead of bc for compatibility)
DX_SCALED=$(awk "BEGIN {printf \"%.2f\", $DX_ORIGINAL * $SCALE_FACTOR}")
DY_SCALED=$(awk "BEGIN {printf \"%.2f\", $DY_ORIGINAL * $SCALE_FACTOR}")

# Generate pair identifier
PAIR_ID="r${REF_SCAN}c${REF_IMG}_r${TGT_SCAN}c${TGT_IMG}"

echo "=================================================================="
echo "Beehive Dataset Evaluation - Processing Image Pairs"
echo "=================================================================="
echo "Processing beehive pair: scan${REF_SCAN}/img${REF_IMG} -> scan${TGT_SCAN}/img${TGT_IMG}"
echo "Using method(s): ${METHODS}"
echo "Using translations: dx=${DX_SCALED}, dy=${DY_SCALED} (scale: ${SCALE_FACTOR})"
echo "Pair ID: ${PAIR_ID}"
echo "🔧 SPECIAL FEATURES:"
echo "   - NIS image order fixed to match UDIS/UDIS++ visual results"
echo "   - NIS seam cutting DISABLED for comparison purposes"

# Proper Conda activation
__conda_setup_script=""
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then 
    __conda_setup_script="$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then 
    __conda_setup_script="$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -n "${CONDA_EXE}" ]; then 
    __conda_setup_script="$(dirname $(dirname "${CONDA_EXE}"))/etc/profile.d/conda.sh"
fi

if [ ! -f "${__conda_setup_script}" ]; then
    echo "Error: Could not find conda.sh. Please ensure conda is properly installed."
    exit 1
fi

# Source conda.sh and activate environment
source "${__conda_setup_script}"
if ! conda activate "${CONDA_ENV_NAME}"; then
    echo "Error: Failed to activate conda environment '${CONDA_ENV_NAME}'."
    exit 1
fi
echo "Activated conda environment '${CONDA_ENV_NAME}'."

# Statistics for tracking
nis_processed=0
udis_processed=0
udis_plus_plus_processed=0
nis_failed=0
udis_failed=0
udis_plus_plus_failed=0

# Function to generate beehive homography data
generate_beehive_homography_data() {
    echo ""
    echo "=================================================================="
    echo "Generating homography data for beehive pair ${PAIR_ID}"
    echo "=================================================================="
    
    echo "Running prepare_beehive_inputs.py..."
    if python prepare_beehive_inputs.py \
        --dataset_path "${BEEHIVE_DATASET_DIR}" \
        --ref_scan "${REF_SCAN}" \
        --ref_img "${REF_IMG}" \
        --tgt_scan "${TGT_SCAN}" \
        --tgt_img "${TGT_IMG}" \
        --dx "${DX_SCALED}" \
        --dy "${DY_SCALED}" \
        --scale "${SCALE_FACTOR}" \
        --output_dir_root "${PROJECT_ROOT}"; then
        
        echo "Successfully generated homography data for pair ${PAIR_ID}"
        
        # Create compatibility structure for UDIS/UDIS++
        echo "Creating compatibility structure for UDIS/UDIS++..."
        mkdir -p "${SIFT_PREPROCESSED_DATA_DIR}"
        mkdir -p "${WARP1_DIR}"
        mkdir -p "${WARP2_DIR}"
        mkdir -p "${MASK1_DIR}"
        mkdir -p "${MASK2_DIR}"
        
        # Create UDIS-D directory structure for original images
        mkdir -p "${PROJECT_ROOT}/UDIS-D/testing/input1"
        mkdir -p "${PROJECT_ROOT}/UDIS-D/testing/input2"
        
        # Create numeric ID that avoids octal interpretation
        # Use format: 1SSIITT where 1 is prefix, SS=ref_scan, II=ref_img, TT=tgt_scan*10+tgt_img
        NUMERIC_PART=$((100000 + REF_SCAN * 1000 + REF_IMG * 100 + TGT_SCAN * 10 + TGT_IMG))
        NUMERIC_ID="$NUMERIC_PART"
        echo "Using numeric ID: ${NUMERIC_ID}"
        
        # Construct original image paths
        REF_IMAGE_PATH="${BEEHIVE_DATASET_DIR}/horizontal_scan${REF_SCAN}/img_${REF_IMG}.jpg"
        TGT_IMAGE_PATH="${BEEHIVE_DATASET_DIR}/horizontal_scan${TGT_SCAN}/img_${TGT_IMG}.jpg"
        
        # Create symlinks for preprocessed data (warps and masks)
        ln -sf "${BEEHIVE_PREPROCESSED_DATA_DIR}/warp1/${PAIR_ID}.png" "${WARP1_DIR}/${NUMERIC_ID}.png"
        ln -sf "${BEEHIVE_PREPROCESSED_DATA_DIR}/warp2/${PAIR_ID}.png" "${WARP2_DIR}/${NUMERIC_ID}.png"
        ln -sf "${BEEHIVE_PREPROCESSED_DATA_DIR}/mask1/${PAIR_ID}.png" "${MASK1_DIR}/${NUMERIC_ID}.png"
        ln -sf "${BEEHIVE_PREPROCESSED_DATA_DIR}/mask2/${PAIR_ID}.png" "${MASK2_DIR}/${NUMERIC_ID}.png"
        
        # Create symlinks for original images in UDIS-D structure
        ln -sf "${REF_IMAGE_PATH}" "${PROJECT_ROOT}/UDIS-D/testing/input1/${NUMERIC_ID}.jpg"
        ln -sf "${TGT_IMAGE_PATH}" "${PROJECT_ROOT}/UDIS-D/testing/input2/${NUMERIC_ID}.jpg"
        
        # Create symlink for homography params
        mkdir -p "${SIFT_PREPROCESSED_DATA_DIR}/homography_params/${NUMERIC_ID}"
        ln -sf "${BEEHIVE_PREPROCESSED_DATA_DIR}/homography_params/${PAIR_ID}/H.txt" "${SIFT_PREPROCESSED_DATA_DIR}/homography_params/${NUMERIC_ID}/H.txt"
        ln -sf "${BEEHIVE_PREPROCESSED_DATA_DIR}/homography_params/${PAIR_ID}/h_params.npz" "${SIFT_PREPROCESSED_DATA_DIR}/homography_params/${NUMERIC_ID}/h_params.npz"
        
        echo "Created compatibility structure with numeric ID: ${NUMERIC_ID}"
        
        # Store the numeric ID for later use
        export BEEHIVE_NUMERIC_ID="${NUMERIC_ID}"
        return 0
    else
        echo "Failed to generate homography data for pair ${PAIR_ID}"
        return 1
    fi
}

# Function to run NIS 
run_nis_for_beehive_pair() {
    echo ""
    echo "=================================================================="
    echo "Running NIS for beehive pair ${PAIR_ID}"
    echo "=================================================================="
    
    # Construct image paths
    REF_IMAGE_PATH="${BEEHIVE_DATASET_DIR}/horizontal_scan${REF_SCAN}/img_${REF_IMG}.jpg"
    TGT_IMAGE_PATH="${BEEHIVE_DATASET_DIR}/horizontal_scan${TGT_SCAN}/img_${TGT_IMG}.jpg"
    
    # Output path
    NIS_OUTPUT="${NIS_BEEHIVE_RESULTS_DIR}/${PAIR_ID}.png"
    HOMOGRAPHY_PATH="${SHARED_BEEHIVE_DATA_DIR}/homography/${PAIR_ID}.txt"
    
    echo "Processing NIS Pair:"
    echo "  Original Ref: ${REF_IMAGE_PATH}"
    echo "  Original Tgt: ${TGT_IMAGE_PATH}"
    echo "  Homography: ${HOMOGRAPHY_PATH}"
    echo "  🔧 ORDER FIX: Swapping ref/tgt arguments to match UDIS order"
    echo "  🚫 SEAM CUTTING: DISABLED for comparison purposes"
    
    # Fix to NIS
    echo "  Swapped Ref: ${TGT_IMAGE_PATH} (was tgt)"
    echo "  Swapped Tgt: ${REF_IMAGE_PATH} (was ref)"
    
    # Use the super simple NIS script with swapped arguments but WITHOUT seam cutting
    echo "Running NIS with SWAPPED ARGUMENTS and NO SEAM CUTTING..."
    if python NIS/nis_stitch_beehive.py \
        --ref "${TGT_IMAGE_PATH}" \
        --tgt "${REF_IMAGE_PATH}" \
        --out "${NIS_OUTPUT}" \
        --scale "${SCALE_FACTOR}" \
        --homography_path "${HOMOGRAPHY_PATH}" \
        --max_canvas_ratio "${MAX_CANVAS_RATIO}" \
        --create_skip_image; then
        
        if [ -f "${NIS_OUTPUT}" ]; then
            echo "NIS successfully processed pair ${PAIR_ID}"
            echo "   Result matches UDIS visual order with swapped arguments"
            nis_processed=1
        else
            echo "Error: NIS output file not created for pair ${PAIR_ID}"
            nis_failed=1
        fi
    else
        echo "Error: NIS processing failed for pair ${PAIR_ID}"
        nis_failed=1
    fi
}

# Function to run UDIS for beehive pair
run_udis_for_beehive_pair() {
    echo ""
    echo "=================================================================="
    echo "Running UDIS for beehive pair ${PAIR_ID}"
    echo "=================================================================="
    
    # Use the stored numeric ID
    NUMERIC_ID="${BEEHIVE_NUMERIC_ID}"
    echo "Using numeric ID: ${NUMERIC_ID}"
    
    # Verify that the required files exist
    if [ ! -f "${WARP1_DIR}/${NUMERIC_ID}.png" ] || \
       [ ! -f "${WARP2_DIR}/${NUMERIC_ID}.png" ] || \
       [ ! -f "${MASK1_DIR}/${NUMERIC_ID}.png" ] || \
       [ ! -f "${MASK2_DIR}/${NUMERIC_ID}.png" ]; then
        echo "Error: Required warps or masks not found for pair ${PAIR_ID} (numeric ID: ${NUMERIC_ID}). Skipping."
        udis_failed=1
        return 1
    fi
    
    # Verify that the original images exist in UDIS-D structure
    if [ ! -f "${PROJECT_ROOT}/UDIS-D/testing/input1/${NUMERIC_ID}.jpg" ] || \
       [ ! -f "${PROJECT_ROOT}/UDIS-D/testing/input2/${NUMERIC_ID}.jpg" ]; then
        echo "Error: Original images not found in UDIS-D structure for numeric ID: ${NUMERIC_ID}. Skipping."
        udis_failed=1
        return 1
    fi
    
    # Run the UDIS script
    if ! bash "${PROJECT_ROOT}/run_udis.sh" "${NUMERIC_ID}" "${NUMERIC_ID}" 1; then
        echo "UDIS processing script failed for numeric ID ${NUMERIC_ID}"
        udis_failed=1
        return
    fi
    
    # Check for the renamed output file from run_udis.sh
    UDIS_OUTPUT_FILE="${PROJECT_ROOT}/processing_data/global_stitching_results/UDIS/${NUMERIC_ID}.jpg"
    if [ -f "${UDIS_OUTPUT_FILE}" ]; then
        # Move the final result to the beehive results directory
        mv "${UDIS_OUTPUT_FILE}" "${UDIS_BEEHIVE_RESULTS_DIR}/${PAIR_ID}.jpg"
        echo "UDIS successfully processed pair ${PAIR_ID}"
        udis_processed=1
    else
        echo "Warning: UDIS output not found at ${UDIS_OUTPUT_FILE}"
        udis_failed=1
    fi
}

# Function to run UDIS++ for beehive pair
run_udis_plus_plus_for_beehive_pair() {
    # Check if scale factor is sufficient for UDIS++
    if [ $(awk -v sf="$SCALE_FACTOR" 'BEGIN { print (sf < 0.4) }') -eq 1 ]; then
        echo "Skipping UDIS++: scale factor ${SCALE_FACTOR} is too small and may cause runtime errors."
        udis_plus_plus_failed=1
        return
    fi
    
    echo ""
    echo "=================================================================="
    echo "Running UDIS++ for beehive pair ${PAIR_ID}"
    echo "=================================================================="
    
    # Use the stored numeric ID
    NUMERIC_ID="${BEEHIVE_NUMERIC_ID}"
    echo "Using numeric ID: ${NUMERIC_ID}"
    
    # Verify that the required files exist
    if [ ! -f "${WARP1_DIR}/${NUMERIC_ID}.png" ] || \
       [ ! -f "${WARP2_DIR}/${NUMERIC_ID}.png" ] || \
       [ ! -f "${MASK1_DIR}/${NUMERIC_ID}.png" ] || \
       [ ! -f "${MASK2_DIR}/${NUMERIC_ID}.png" ]; then
        echo "Error: Required warps or masks not found for pair ${PAIR_ID} (numeric ID: ${NUMERIC_ID}). Skipping."
        udis_plus_plus_failed=1
        return 1
    fi
    
    # Clear any existing outputs
    rm -f "${PROJECT_ROOT}/global_stitching_results/UDIS_plus_plus/${NUMERIC_ID}_stitched.jpg"
    
    # Run UDIS++
    echo "Running UDIS++ for numeric ID ${NUMERIC_ID}"
    if ! bash "${PROJECT_ROOT}/run_udis_plus_plus.sh" "${NUMERIC_ID}" "${NUMERIC_ID}" 1; then
        echo "UDIS++ processing script failed for numeric ID ${NUMERIC_ID}"
        udis_plus_plus_failed=1
        return
    fi

    # Check for the renamed output file from run_udis_plus_plus.sh
    UDISPP_OUTPUT_FILE="${PROJECT_ROOT}/processing_data/global_stitching_results/UDIS_plus_plus/${NUMERIC_ID}_stitched.jpg"
    if [ -f "${UDISPP_OUTPUT_FILE}" ]; then
        # Move the final result to the beehive results directory
        mv "${UDISPP_OUTPUT_FILE}" "${UDIS_PLUS_PLUS_BEEHIVE_RESULTS_DIR}/${PAIR_ID}.jpg"
        echo "UDIS++ successfully processed pair ${PAIR_ID}"
        udis_plus_plus_processed=1
    else
        echo "Warning: UDIS++ output not found at ${UDISPP_OUTPUT_FILE}"
        udis_plus_plus_failed=1
    fi
}

# Main processing pipeline
echo ""

# Step 1: Generate beehive homography data
if ! generate_beehive_homography_data; then
    echo "Error: Failed to generate beehive homography data. Exiting."
    exit 1
fi

# Step 2: Run selected methods
for method in "${METHODS_ARRAY[@]}"; do
    case $method in
        "nis")
            run_nis_for_beehive_pair
            ;;
        "udis")
            run_udis_for_beehive_pair
            ;;
        "udis_plus_plus")
            run_udis_plus_plus_for_beehive_pair
            ;;
        *)
            echo "Warning: Unknown method '$method'. Skipping."
            ;;
    esac
done

# Summary
echo ""
echo "=================================================================="
echo "Summary of processing for pair ${PAIR_ID}"
echo "=================================================================="

for method in "${METHODS_ARRAY[@]}"; do
    case $method in
        "nis")
            if [ $nis_processed -eq 1 ]; then
                echo "NIS: SUCCESS"
            else
                echo "NIS: FAILED"
            fi
            ;;
        "udis")
            if [ $udis_processed -eq 1 ]; then
                echo "UDIS: SUCCESS"
            else
                echo "UDIS: FAILED"
            fi
            ;;
        "udis_plus_plus")
            if [ $udis_plus_plus_processed -eq 1 ]; then
                echo "UDIS++: SUCCESS"
            else
                echo "UDIS++: FAILED"
            fi
            ;;
    esac
done

echo ""
echo "Results saved in: ${BEEHIVE_RESULTS_DIR}"
echo "Homography data saved in: ${SHARED_BEEHIVE_DATA_DIR}"
echo ""
echo "Technical details:"
echo "   - NIS uses swapped arguments for consistent image order"
echo "   - Seam cutting disabled for fair comparison"
echo "   - Expected: img_${REF_IMG} (left) → img_${TGT_IMG} (right)"
echo ""
echo "Script completed." 