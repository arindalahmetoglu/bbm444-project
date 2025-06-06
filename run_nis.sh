#!/bin/bash
# Script to run no-seam image stitching with enhanced feature detection
# on multiple image pairs with a specified range and stride

echo "IMPORTANT: Running enhanced feature detection + NIS without seam cutting"

# Ensure script exits on first error
set -e

# Define project root
PROJECT_ROOT=$(pwd)

# Define dataset paths using UDIS-D structure
UDIS_D_BASE_DIR="${PROJECT_ROOT}/UDIS-D/testing"
UDIS_D_TESTING_INPUT1_DIR="${UDIS_D_BASE_DIR}/input1"
UDIS_D_TESTING_INPUT2_DIR="${UDIS_D_BASE_DIR}/input2"

# Output directory for results
RESULTS_DIR="${PROJECT_ROOT}/processing_data/enhanced_features_results"

# Conda environment name
CONDA_ENV_NAME="nis"

# Configuration file (relative to NIS/)
NIS_CONFIG_FILE="NIS/configs/test/NIS_blending.yaml"

# Parameters
SCALE_FACTOR="1.0"
FEATURE_TYPE="orb"  # Default feature type: sift, orb, akaze, or brisk
MATCHER_TYPE="flann" # Default matcher type: flann or bf
DEBUG_MODE="false"   # Whether to save visualization of the feature matching
HOMOGRAPHY_TIMEOUT="30" # Maximum time in seconds for homography estimation
PROCESS_TIMEOUT="300" # Maximum time in seconds for entire stitching process
MAX_CANVAS_RATIO="3.0" # Maximum allowed ratio for canvas size
CREATE_SKIP_IMAGE="true" # Create a black placeholder for skipped images
CHECK_GPU_MEMORY="true" # Check for sufficient GPU memory
MIN_GPU_MEMORY="4000" # Minimum required GPU memory in MB

# --- Argument Parsing for range and stride ---
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <START_INDEX> <END_INDEX> <STRIDE> [FEATURE_TYPE] [MATCHER_TYPE] [DEBUG_MODE] [HOMO_TIMEOUT] [PROC_TIMEOUT] [MAX_RATIO] [MIN_GPU_MB] [HOMOGRAPHY_FILE_PATH]"
    echo "Processes image pairs from UDIS-D/testing based on the provided range and stride."
    echo ""
    echo "Arguments:"
    echo "  START_INDEX   Starting index of image pairs to process"
    echo "  END_INDEX     Ending index of image pairs to process"
    echo "  STRIDE        Step size between indices (e.g., 5 for 100, 105, 110, ...)"
    echo "  FEATURE_TYPE  Optional: Type of feature detector to use (sift, orb, akaze, brisk) [default: sift]"
    echo "  MATCHER_TYPE  Optional: Type of matcher to use (flann, bf) [default: flann]"
    echo "  DEBUG_MODE    Optional: Enable debug visualizations (true, false) [default: false]"
    echo "  HOMO_TIMEOUT  Optional: Maximum time in seconds for homography estimation [default: 30]"
    echo "  PROC_TIMEOUT  Optional: Maximum time in seconds for entire process [default: 300]"
    echo "  MAX_RATIO     Optional: Maximum allowed ratio for canvas size [default: 3.0]"
    echo "  MIN_GPU_MB    Optional: Minimum required GPU memory in MB [default: 4000]"
    echo "  HOMOGRAPHY_FILE_PATH Optional: Path to pre-computed homography file"
    echo ""
    echo "Example: $0 100 199 5 orb flann true 30 300 3.0 4000 /path/to/homography_file.txt"
    exit 1
fi

START_INDEX=$1
END_INDEX=$2
STRIDE=$3

# Optional parameters
if [ "$#" -ge 4 ]; then
    FEATURE_TYPE=$4
fi
if [ "$#" -ge 5 ]; then
    MATCHER_TYPE=$5
fi
if [ "$#" -ge 6 ]; then
    DEBUG_MODE=$6
fi
if [ "$#" -ge 7 ]; then
    HOMOGRAPHY_TIMEOUT=$7
fi
if [ "$#" -ge 8 ]; then
    PROCESS_TIMEOUT=$8
fi
if [ "$#" -ge 9 ]; then
    MAX_CANVAS_RATIO=$9
fi
if [ "$#" -ge 10 ]; then
    MIN_GPU_MEMORY=${10}
fi
if [ "$#" -ge 11 ]; then
    HOMOGRAPHY_FILE_PATH=${11}
fi

# Format debug mode as command-line flag
DEBUG_FLAG=""
if [ "$DEBUG_MODE" = "true" ]; then
    DEBUG_FLAG="--debug_mode"
fi

# Format create skip image flag
SKIP_IMAGE_FLAG="--create_skip_image"

# Format GPU memory check flag
GPU_CHECK_FLAG="--check_gpu_memory"

echo "Starting enhanced feature detection + no-seam stitching for pairs from index ${START_INDEX} to ${END_INDEX} with stride ${STRIDE}..."
echo "Feature type: ${FEATURE_TYPE}"
echo "Matcher type: ${MATCHER_TYPE}"
echo "Debug mode: ${DEBUG_MODE}"
echo "Homography timeout: ${HOMOGRAPHY_TIMEOUT} seconds"
echo "Process timeout: ${PROCESS_TIMEOUT} seconds"
echo "Maximum canvas ratio: ${MAX_CANVAS_RATIO}"
echo "Check GPU memory: ${CHECK_GPU_MEMORY}"
echo "Minimum GPU memory: ${MIN_GPU_MEMORY} MB"
echo "Create skip images: ${CREATE_SKIP_IMAGE}"
echo "Input1 Directory: ${UDIS_D_TESTING_INPUT1_DIR}"
echo "Input2 Directory: ${UDIS_D_TESTING_INPUT2_DIR}"
echo "Results will be saved to: ${RESULTS_DIR}/${FEATURE_TYPE}_${MATCHER_TYPE}"

# Check if directories exist
if [ ! -d "${UDIS_D_TESTING_INPUT1_DIR}" ]; then
    echo "Error: Input directory '${UDIS_D_TESTING_INPUT1_DIR}' not found."
    exit 1
fi

if [ ! -d "${UDIS_D_TESTING_INPUT2_DIR}" ]; then
    echo "Error: Input directory '${UDIS_D_TESTING_INPUT2_DIR}' not found."
    exit 1
fi

# Create output directory with feature type and matcher type
RESULTS_SUBDIR="${RESULTS_DIR}/${FEATURE_TYPE}_${MATCHER_TYPE}"
mkdir -p "${RESULTS_SUBDIR}"

# Proper Conda activation
# Try to find conda.sh
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

processed_pairs_count=0
failed_pairs=()
skipped_pairs=()

# Loop through specified image indices
for (( K=START_INDEX; K<=END_INDEX; K+=STRIDE )); do
    pair_idx_formatted=$(printf "%06d" $K)
    ref_image_path="${UDIS_D_TESTING_INPUT1_DIR}/${pair_idx_formatted}.jpg"
    tgt_image_path="${UDIS_D_TESTING_INPUT2_DIR}/${pair_idx_formatted}.jpg"

    if [ ! -f "${ref_image_path}" ]; then
        echo "Warning: Reference image ${ref_image_path} not found for pair index ${K}. Skipping."
        continue
    fi
    if [ ! -f "${tgt_image_path}" ]; then
        echo "Warning: Target image ${tgt_image_path} not found for pair index ${K}. Skipping."
        continue
    fi
    
    output_filename="${FEATURE_TYPE}_${MATCHER_TYPE}_${pair_idx_formatted}.png"
    output_path="${RESULTS_SUBDIR}/${output_filename}"

    echo ""
    echo "Processing Pair K=${K}:"
    echo "  Ref: ${ref_image_path}"
    echo "  Tgt: ${tgt_image_path}"
    echo "  Output: ${output_path}"

    # Run the stitching with enhanced feature detection
    HOMOGRAPHY_ARG=""
    COMPUTE_ON_FLY_ARG=""
    if [ -n "${HOMOGRAPHY_FILE_PATH}" ] && [ -f "${HOMOGRAPHY_FILE_PATH}" ]; then
        HOMOGRAPHY_ARG="--homography_path ${HOMOGRAPHY_FILE_PATH}"
        # When homography file is provided, don't pass --compute_on_the_fly flag (defaults to False)
        echo "  Using pre-computed homography: ${HOMOGRAPHY_FILE_PATH}"
    else
        # When no homography file, pass --compute_on_the_fly flag to compute on-the-fly
        COMPUTE_ON_FLY_ARG="--compute_on_the_fly"
    fi
    
    if python NIS/stitch_enhanced_features.py \
        --config "${NIS_CONFIG_FILE}" \
        --ref "${ref_image_path}" \
        --tgt "${tgt_image_path}" \
        --out "${output_path}" \
        --scale "${SCALE_FACTOR}" \
        --feature_type "${FEATURE_TYPE}" \
        --matcher_type "${MATCHER_TYPE}" \
        --timeout_seconds "${HOMOGRAPHY_TIMEOUT}" \
        --process_timeout "${PROCESS_TIMEOUT}" \
        --max_canvas_ratio "${MAX_CANVAS_RATIO}" \
        --min_gpu_memory "${MIN_GPU_MEMORY}" \
        ${HOMOGRAPHY_ARG} \
        ${COMPUTE_ON_FLY_ARG} \
        ${SKIP_IMAGE_FLAG} \
        ${GPU_CHECK_FLAG} \
        ${DEBUG_FLAG}; then
        
        # Check if the output is a skip image (small black image)
        if [ -f "${output_path}" ]; then
            # Get image dimensions
            img_info=$(identify -format "%w %h" "${output_path}" 2>/dev/null || echo "0 0")
            img_width=$(echo $img_info | cut -d' ' -f1)
            img_height=$(echo $img_info | cut -d' ' -f2)
            
            if [ "$img_width" -eq 100 ] && [ "$img_height" -eq 100 ]; then
                echo "Pair ${K} was skipped (timeout, oversized canvas, or insufficient GPU memory)."
                skipped_pairs+=($K)
            else
                echo "Successfully processed pair ${K}."
                processed_pairs_count=$((processed_pairs_count + 1))
            fi
        else
            echo "Error: Output file not created for pair ${K}."
            failed_pairs+=($K)
        fi
    else
        echo "Error processing pair ${K}. Continuing with next pair."
        failed_pairs+=($K)
    fi
done

echo ""
if [ "$processed_pairs_count" -eq 0 ]; then
    echo "No pairs were processed. Check input range, stride, and availability of images."
else
    echo "Enhanced feature detection + no-seam stitching finished for ${processed_pairs_count} pairs."
    echo "Results saved in ${RESULTS_SUBDIR}"
    
    if [ ${#skipped_pairs[@]} -gt 0 ]; then
        echo "Skipped pairs (timeout, oversized canvas, or insufficient GPU memory): ${skipped_pairs[@]}"
    fi
    
    if [ ${#failed_pairs[@]} -gt 0 ]; then
        echo "Failed pairs: ${failed_pairs[@]}"
    else
        echo "All remaining pairs processed successfully."
    fi
fi

conda deactivate
echo "Script completed." 