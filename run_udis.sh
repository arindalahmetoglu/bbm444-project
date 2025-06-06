#!/bin/bash
# Script to run UnsupervisedDeepImageStitching (UDIS) inference using globally preprocessed SIFT data
# This script is a wrapper around the UDIS python scripts, handling path setup,
# conda environment activation, and calling the inference script with the correct arguments.
# It's designed to be called from other master scripts like `run_udis_d_methods.sh`
# or `run_beehive_methods.sh`.

PROJECT_ROOT=$(pwd)
UDIS_DIR="${PROJECT_ROOT}/UDIS"
RESULTS_DIR="${PROJECT_ROOT}/processing_data/global_stitching_results/UDIS"

# Ensure the results directory exists
mkdir -p "${RESULTS_DIR}"

# Path to UDIS codes and the constant.py file that needs modification
UDIS_CODES_DIR="${PROJECT_ROOT}/UDIS/ImageReconstruction/Codes"
UDIS_CONSTANT_PY="${UDIS_CODES_DIR}/constant.py"

# --- Main Execution ---

# Activate the correct conda environment for UDIS
# Using source to ensure the environment is activated in the current shell
if ! source activate udis_env; then
    echo "Error: Failed to activate conda environment 'udis_env'." >&2
    echo "Please ensure conda is initialized and the environment exists." >&2
    exit 1
fi

echo "Conda environment 'udis_env' activated."

# Dynamically update the UDIS result directory in constant.py
# This is a critical step to ensure UDIS saves its output to the correct, non-hardcoded location.
sed -i.bak "s|RESULT_DIR = .*|RESULT_DIR = r'${RESULTS_DIR}'|" "${UDIS_CONSTANT_PY}"
echo "Updated UDIS result directory in ${UDIS_CONSTANT_PY} to: ${RESULTS_DIR}"


# Navigate to the UDIS source code directory to run the inference
# The UDIS scripts often rely on relative paths, so running from here is safest.
cd "${UDIS_CODES_DIR}"
echo "Changed directory to ${UDIS_CODES_DIR}"

# --- Argument Parsing for range and stride ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <START_INDEX> <END_INDEX> <STRIDE>"
    echo "Processes image pairs from UDIS-D/testing (using preprocessed SIFT data) based on the range and stride."
    echo "Example: $0 100 200 5"
    exit 1
fi

START_INDEX=$1
END_INDEX=$2
STRIDE=$3

# Loop through the specified range of pairs to process
processed_pairs_count=0
for (( K=START_INDEX; K<=END_INDEX; K+=STRIDE )); do
    K_padded=$(printf "%06d" "${K}")
    echo "----------------------------------------------------------------------"
    echo "Processing UDIS for pair index K=${K}..."
    echo "----------------------------------------------------------------------"

    # Define paths for the input data for this specific pair
    # Path for constant.py TEST_FOLDER (relative from UDIS_CODES_DIR)
    # The UDIS code expects this path to be relative to its own location.
    # The actual data is in `udisd-m/` at the project root.
    test_folder_relative="../../udisd-m/${K_padded}_${K_padded}"

    # Check that the required input files actually exist before running the model
    # This prevents cryptic errors from the Python script.
    if [ ! -f "${test_folder_relative}/${K_padded}_0.jpg" ] || [ ! -f "${test_folder_relative}/${K_padded}_1.jpg" ]; then
        echo "Error: Input images for pair ${K} not found at ${test_folder_relative}. Skipping."
        continue
    fi
     if [ ! -f "${test_folder_relative}/H_0_1" ]; then
        echo "Error: Homography file for pair ${K} not found at ${test_folder_relative}/H_0_1. Skipping."
        continue
    fi


    # Dynamically update the TEST_FOLDER in constant.py for the current pair
    # This tells the UDIS script which specific image pair to process.
    sed -i.bak "s|TEST_FOLDER = .*|TEST_FOLDER = r'${test_folder_relative}'|" "${UDIS_CONSTANT_PY}"
    echo "Updated TEST_FOLDER in ${UDIS_CONSTANT_PY} for pair ${K}"

    # Now, run the actual UDIS inference script
    echo "Running UDIS inference.py for pair K=${K}..."
    # The timeout is a safeguard against the process hanging indefinitely.
    if ! timeout "${PROCESS_TIMEOUT}" python inference.py; then
        echo "Error running UDIS inference.py for pair K=${K}. It might have timed out or crashed." >&2
    else
        echo "UDIS inference.py finished for pair K=${K}."
        # inference.py saves to UDIS_RESULTS_SAVE_DIR (e.g., global_stitching_results/UDIS/)
        # The output filename is the pair index, e.g., "2.jpg" for pair 2.
        # We need to construct the full path to check for success.
        expected_output="${RESULTS_DIR}/${K}.jpg"
        if [ -f "${expected_output}" ]; then
            echo "Successfully generated output for pair ${K} at ${expected_output}"
            processed_pairs_count=$((processed_pairs_count + 1))
        else
            echo "Warning: UDIS script finished but expected output file was not found: ${expected_output}" >&2
        fi
    fi
done

# Return to the original project root directory
cd "${PROJECT_ROOT}"
echo "Returned to directory ${PROJECT_ROOT}"


# Final summary message
if [ "${processed_pairs_count}" -eq 0 ]; then
    echo "Warning: No pairs were successfully processed by UDIS. Check range, stride, and input data." >&2
else
    echo "UDIS inference finished for ${processed_pairs_count} specified pairs. Results are in ${RESULTS_DIR}"
fi

echo "Deactivating conda environment."
conda deactivate

echo "Script completed." 