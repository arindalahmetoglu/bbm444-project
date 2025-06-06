#!/bin/bash
# Script to run UDIS++ Composition stage using globally preprocessed SIFT data

echo "IMPORTANT: It is strongly recommended to run this script WITHOUT sudo."

# Ensure script exits on first error
set -e

# Define project root
PROJECT_ROOT=$(pwd)
SIFT_PREPROCESSED_DIR="${PROJECT_ROOT}/processing_data/sift_preprocessed_data"
UDIS_D_BASE_DIR="${PROJECT_ROOT}/UDIS-D/testing"
UDIS_PLUS_PLUS_DIR="${PROJECT_ROOT}/UDIS++"
RESULTS_DIR="${PROJECT_ROOT}/processing_data/global_stitching_results/UDIS_plus_plus"
CONDA_ENV_NAME="stitch"
TEMP_SUBSET_DIR_BASE="${PROJECT_ROOT}/processing_data/temp_sift_subset_udis_pp"

# Path to the UDIS++ Composition script directory
UDISPP_COMPOSITION_DIR="${PROJECT_ROOT}/UDIS++/Composition/Codes"
# Output directory for results (already absolute)
UDISPP_RESULTS_SAVE_DIR="${PROJECT_ROOT}/processing_data/global_stitching_results/UDIS_plus_plus"
mkdir -p "${UDISPP_RESULTS_SAVE_DIR}"

# --- Argument Parsing for range and stride ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <START_INDEX> <END_INDEX> <STRIDE>"
    echo "Processes image pairs from UDIS-D/testing (using preprocessed SIFT data) based on the provided range and stride."
    echo "Example: $0 100 200 5 (processes pairs 100, 105,..., up to 200)"
    exit 1
fi

START_INDEX=$1
END_INDEX=$2
STRIDE=$3

EXPECTED_UDISPP_OUTPUT_FILENAME="stitched.jpg" # Adjust if test.py produces a different generic name

# Create a unique temporary directory for this run
run_id=$RANDOM
TEMP_SUBSET_DIR="${TEMP_SUBSET_DIR_BASE}_${run_id}"

echo "Starting UDIS++ pipeline for pairs from index ${START_INDEX} to ${END_INDEX} with stride ${STRIDE}..."
echo "Using pre-processed data from: ${SIFT_PREPROCESSED_DIR}"
echo "Temporary subset directory: ${TEMP_SUBSET_DIR}"
echo "Results will be saved to: ${RESULTS_DIR}"

# Check if global SIFT data directory exists (basic check)
if [ ! -d "${SIFT_PREPROCESSED_DIR}/warp1" ]; then # Check for a typical subfolder
    echo "Error: Global SIFT data directory '${SIFT_PREPROCESSED_DIR}/warp1' not found."
    echo "Please ensure preprocess_sift_ransac_globally.py has been run successfully."
    exit 1
fi

# Check if ImageMagick convert tool is available (for PNG to JPG conversion)
if ! command -v convert &> /dev/null; then
    echo "Warning: ImageMagick 'convert' tool not found. Will attempt to use opencv_converter.py for PNG to JPG conversion."
    # Create a simple Python script for conversion if ImageMagick is not available
    cat > "${PROJECT_ROOT}/opencv_converter.py" << 'EOF'
import sys
import cv2
import os

if len(sys.argv) != 3:
    print("Usage: python opencv_converter.py <input_png> <output_jpg>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

if not os.path.exists(input_file):
    print(f"Error: Input file {input_file} does not exist")
    sys.exit(1)

# Read and write the image
img = cv2.imread(input_file)
if img is None:
    print(f"Error: Could not read {input_file}")
    sys.exit(1)
    
cv2.imwrite(output_file, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
print(f"Converted {input_file} to {output_file}")
EOF
    CONVERT_CMD="python ${PROJECT_ROOT}/opencv_converter.py"
else
    CONVERT_CMD="convert"
fi

# Attempt to activate conda environment (common anaconda/miniconda paths)
__conda_setup_script=""
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then __conda_setup_script="$HOME/anaconda3/etc/profile.d/conda.sh";
elif type conda &>/dev/null && conda info --base &>/dev/null; then __conda_setup_script="$(conda info --base)/etc/profile.d/conda.sh";
elif [ -n "${CONDA_EXE}" ]; then __conda_setup_script="$(dirname $(dirname "${CONDA_EXE}"))/etc/profile.d/conda.sh"; fi
if [ ! -f "$__conda_setup_script" ]; then
    COMMON_PATHS=("$HOME/miniconda3/etc/profile.d/conda.sh" "/opt/anaconda3/etc/profile.d/conda.sh" "/opt/miniconda3/etc/profile.d/conda.sh")
    for path_to_check in "${COMMON_PATHS[@]}"; do if [ -f "${path_to_check}" ]; then __conda_setup_script="${path_to_check}"; break; fi; done
fi
if [ -f "$__conda_setup_script" ]; then source "$__conda_setup_script"; conda activate "${CONDA_ENV_NAME}";
    if [ $? -ne 0 ]; then echo "Error: Failed to activate conda environment '${CONDA_ENV_NAME}'."; exit 1; fi
    echo "Conda environment '${CONDA_ENV_NAME}' successfully activated.";
else echo "Error: conda.sh could not be located."; exit 1;
fi

cd "${UDISPP_COMPOSITION_DIR}"
echo "Changed directory to ${UDISPP_COMPOSITION_DIR}"

processed_pairs_count=0
for (( K=START_INDEX; K<=END_INDEX; K+=STRIDE )); do
    echo ""
    echo "Processing UDIS++ for pair index K=${K}..."
    
    IMG_BASENAME_K=$(printf "%06d" $K)
    TEMP_SIFT_SUBSET_DIR_NAME="temp_sift_subset_udis_pp_${K}"
    # Path relative to PROJECT_ROOT for creation/deletion
    TEMP_SIFT_SUBSET_DIR_ABS="${PROJECT_ROOT}/${TEMP_SIFT_SUBSET_DIR_NAME}"
    # Path relative to UDISPP_COMPOSITION_DIR for test.py --test_path
    TEMP_SIFT_SUBSET_DIR_RELATIVE="../../../${TEMP_SIFT_SUBSET_DIR_NAME}"

    # Create temporary SIFT subset directory and populate it
    echo "Creating temporary SIFT subset directory: ${TEMP_SIFT_SUBSET_DIR_ABS}"
    mkdir -p "${TEMP_SIFT_SUBSET_DIR_ABS}/warp1"
    mkdir -p "${TEMP_SIFT_SUBSET_DIR_ABS}/warp2"
    mkdir -p "${TEMP_SIFT_SUBSET_DIR_ABS}/mask1"
    mkdir -p "${TEMP_SIFT_SUBSET_DIR_ABS}/mask2"
    mkdir -p "${TEMP_SIFT_SUBSET_DIR_ABS}/homography_params"

    sift_files_missing=false
    for comp in warp1 warp2 mask1 mask2; do
        src_file="${SIFT_PREPROCESSED_DIR}/${comp}/${IMG_BASENAME_K}.png"
        # Change the extension to .jpg since TestDataset looks for .jpg files
        dst_file="${TEMP_SIFT_SUBSET_DIR_ABS}/${comp}/${IMG_BASENAME_K}.jpg"
        if [ -f "${src_file}" ]; then
            # Convert PNG to JPG
            if [ "$CONVERT_CMD" = "convert" ]; then
                convert "${src_file}" "${dst_file}"
            else
                python "${PROJECT_ROOT}/opencv_converter.py" "${src_file}" "${dst_file}"
            fi
            echo "Converted ${src_file} to ${dst_file}"
        else
            echo "Warning: SIFT file ${src_file} not found for pair K=${K}."
            sift_files_missing=true; break
        fi
    done
    if ${sift_files_missing}; then rm -rf "${TEMP_SIFT_SUBSET_DIR_ABS}"; echo "Skipping pair K=${K} due to missing SIFT image components."; continue; fi

    src_param_file="${SIFT_PREPROCESSED_DIR}/homography_params/${IMG_BASENAME_K}/h_params.npz"
    dst_param_file="${TEMP_SIFT_SUBSET_DIR_ABS}/homography_params/params_${K}.npz"
    if [ -f "${src_param_file}" ]; then
        cp "${src_param_file}" "${dst_param_file}"
    else
        echo "Warning: SIFT params file ${src_param_file} not found for pair K=${K}."
        rm -rf "${TEMP_SIFT_SUBSET_DIR_ABS}"; echo "Skipping pair K=${K} due to missing SIFT params."; continue
    fi

    echo "Running UDIS++ Composition test.py for pair K=${K}..."
    # UDISPP_RESULTS_SAVE_DIR is absolute path to global_stitching_results/UDIS_plus_plus
    python test.py \
        --test_path "${TEMP_SIFT_SUBSET_DIR_RELATIVE}" \
        --save_path "${UDISPP_RESULTS_SAVE_DIR}"
    
    if [ $? -ne 0 ]; then
        echo "Error running UDIS++ test.py for pair K=${K}."
        # Decide if to abort all or continue
    else
        echo "UDIS++ test.py finished for pair K=${K}."
        # UDIS++ test.py saves into subdirectories of UDISPP_RESULTS_SAVE_DIR
        # e.g., UDISPP_RESULTS_SAVE_DIR/final_fusion/some_image_name.png
        
        FINAL_FUSION_DIR="${UDISPP_RESULTS_SAVE_DIR}/final_fusion"
        # Try to find the first .png or .jpg in the final_fusion directory
        # This assumes test.py produces one primary output image per run for a single input pair.
        found_stitched_image=$(find "${FINAL_FUSION_DIR}" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' \) -print -quit)

        TARGET_OUTPUT_PATH="${UDISPP_RESULTS_SAVE_DIR}/${IMG_BASENAME_K}_stitched.jpg"

        if [ -n "${found_stitched_image}" ] && [ -f "${found_stitched_image}" ]; then
            echo "Found stitched image: ${found_stitched_image}"
            echo "Renaming to ${TARGET_OUTPUT_PATH}"
            mkdir -p "$(dirname "${TARGET_OUTPUT_PATH}")"
            mv "${found_stitched_image}" "${TARGET_OUTPUT_PATH}"
            rmdir "${FINAL_FUSION_DIR}" 2>/dev/null || true 
            rmdir "${UDISPP_RESULTS_SAVE_DIR}/raw_composition_mask" 2>/dev/null || true
        else
            echo "Warning: No .png or .jpg output file found in ${FINAL_FUSION_DIR} for pair K=${K}. Manual check may be needed."
            rm -rf "${FINAL_FUSION_DIR}" # Remove if it exists, as it didn't contain the expected output
            rm -rf "${UDISPP_RESULTS_SAVE_DIR}/raw_composition_mask"
        fi
        processed_pairs_count=$((processed_pairs_count + 1))
    fi

    echo "Cleaning up temporary SIFT directory: ${TEMP_SIFT_SUBSET_DIR_ABS}"
    rm -rf "${TEMP_SIFT_SUBSET_DIR_ABS}"

done

cd "${PROJECT_ROOT}" # Go back to project root
echo "Changed directory back to ${PROJECT_ROOT}"

# Clean up the temporary converter script if it was created
if [ -f "${PROJECT_ROOT}/opencv_converter.py" ]; then
    rm "${PROJECT_ROOT}/opencv_converter.py"
fi

echo ""
if [ "$processed_pairs_count" -eq 0 ]; then
    echo "No pairs were processed by UDIS++. Check input range, stride, and data availability."
else
    echo "UDIS++ Composition finished for ${processed_pairs_count} specified pairs. Results should be in ${UDISPP_RESULTS_SAVE_DIR}"
fi
echo "Deactivating conda environment."
conda deactivate

echo "Script completed." 