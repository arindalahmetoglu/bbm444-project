#!/bin/bash

# =============================================================================
# UDIS-D Warp Homography Preparation Script
# =============================================================================
# 
# This script generates standardized geometric warps for the UDIS-D dataset
# using ORB feature detection + FLANN matching + RANSAC homography estimation
# as described in the paper's Appendix A.
#
# The pipeline:
# 1. Detects ORB features in image pairs
# 2. Matches features using FLANN-based matcher
# 3. Estimates homography using RANSAC
# 4. Generates warped images and masks for composition network evaluation

# =============================================================================

set -e  # Exit on any error

# =============================================================================
# Configuration and Default Parameters
# =============================================================================

# Default processing parameters
START_INDEX=1
END_INDEX=100
STRIDE=1
BATCH_SIZE=10
GPU_ID=0
PROCESS_TIMEOUT=120

# ORB + FLANN parameters
MAX_ORB_FEATURES=5000
MATCH_RATIO=0.75
RANSAC_THRESHOLD=4.0
MAX_CANVAS_RATIO=3.0

# Directory structure
PROJECT_ROOT=$(pwd)
DATASET_DIR="${PROJECT_ROOT}/UDIS-D/testing"
UDIS_D_TESTING_INPUT1_DIR="${DATASET_DIR}/input1"
UDIS_D_TESTING_INPUT2_DIR="${DATASET_DIR}/input2"

# Output directory for warps and homographies
OUTPUT_DIR="${PROJECT_ROOT}/processing_data/udis_d_warps"
HOMOGRAPHY_DIR="${OUTPUT_DIR}/homographies"
WARPED_DIR="${OUTPUT_DIR}/warped"
MASKS_DIR="${OUTPUT_DIR}/masks"

# Conda environment
CONDA_ENV_NAME="nis"

# Path to the Python script for homography extraction
HOMOGRAPHY_SCRIPT_PATH="${PROJECT_ROOT}/orb_flann_homography.py"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo "=============================================================================="
    echo "$1"
    echo "=============================================================================="
}

print_section() {
    echo ""
    echo "--- $1 ---"
}

check_prerequisites() {
    print_section "Checking Prerequisites"
    
    # Check dataset directory
    if [[ ! -d "${UDIS_D_TESTING_INPUT1_DIR}" ]] || [[ ! -d "${UDIS_D_TESTING_INPUT2_DIR}" ]]; then
        echo "Error: UDIS-D dataset not found!"
        echo "Expected directories:"
        echo "  ${UDIS_D_TESTING_INPUT1_DIR}"
        echo "  ${UDIS_D_TESTING_INPUT2_DIR}"
        echo ""
        echo "Please ensure UDIS-D dataset is properly extracted in ${DATASET_DIR}"
        exit 1
    fi
    
    # Check homography script
    if [[ ! -f "${HOMOGRAPHY_SCRIPT_PATH}" ]]; then
        echo "Error: ORB+FLANN homography script not found!"
        echo "Expected: ${HOMOGRAPHY_SCRIPT_PATH}"
        exit 1
    fi
    
    echo "✓ Dataset directories found"
    echo "✓ Homography script found"
}

setup_conda_environment() {
    print_section "Setting up Conda Environment"
    
    # Find conda setup script
    CONDA_SETUP_SCRIPT=""
    for conda_path in "$HOME/anaconda3" "$HOME/miniconda3" "$(dirname $(dirname ${CONDA_EXE:-/usr/bin/conda}))"; do
        if [[ -f "${conda_path}/etc/profile.d/conda.sh" ]]; then
            CONDA_SETUP_SCRIPT="${conda_path}/etc/profile.d/conda.sh"
            break
        fi
    done
    
    if [[ -z "${CONDA_SETUP_SCRIPT}" ]]; then
        echo "Error: Could not locate conda.sh"
        echo "Please ensure conda is properly installed"
        exit 1
    fi
    
    # Activate conda
    source "${CONDA_SETUP_SCRIPT}"
    if ! conda activate "${CONDA_ENV_NAME}"; then
        echo "Error: Failed to activate conda environment '${CONDA_ENV_NAME}'"
        echo "Please ensure the environment exists and contains required packages"
        exit 1
    fi
    
    echo "✓ Activated conda environment '${CONDA_ENV_NAME}'"
}

create_output_directories() {
    print_section "Creating Output Directories"
    
    mkdir -p "${HOMOGRAPHY_DIR}"
    mkdir -p "${WARPED_DIR}/input1"
    mkdir -p "${WARPED_DIR}/input2" 
    mkdir -p "${MASKS_DIR}/input1"
    mkdir -p "${MASKS_DIR}/input2"
    
    echo "Output directories created:"
    echo "  Homographies: ${HOMOGRAPHY_DIR}"
    echo "  Warped images: ${WARPED_DIR}"
    echo "  Masks: ${MASKS_DIR}"
}

# =============================================================================
# Command Line Argument Parsing
# =============================================================================

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Generate standardized warp homographies for UDIS-D dataset using ORB+FLANN.

OPTIONS:
    --start N              Starting image index (default: ${START_INDEX})
    --end N                Ending image index (default: ${END_INDEX})
    --stride N             Step between indices (default: ${STRIDE})
    --batch-size N         Process N images at a time (default: ${BATCH_SIZE})
    --gpu N                GPU device ID (default: ${GPU_ID})
    --timeout N            Process timeout in seconds (default: ${PROCESS_TIMEOUT})
    
    --max-features N       Maximum ORB features (default: ${MAX_ORB_FEATURES})
    --match-ratio F        FLANN match ratio threshold (default: ${MATCH_RATIO})
    --ransac-threshold F   RANSAC threshold (default: ${RANSAC_THRESHOLD})
    --max-canvas-ratio F   Maximum canvas size ratio (default: ${MAX_CANVAS_RATIO})
    
    --output-dir PATH      Output directory (default: ${OUTPUT_DIR})
    --help, -h             Show this help message

EXAMPLES:
    $0 --start 1 --end 50 --stride 5
    $0 --start 100 --end 200 --batch-size 20
    $0 --max-features 10000 --match-ratio 0.8

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --start)
            START_INDEX="$2"
            shift 2
            ;;
        --end)
            END_INDEX="$2"
            shift 2
            ;;
        --stride)
            STRIDE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --timeout)
            PROCESS_TIMEOUT="$2"
            shift 2
            ;;
        --max-features)
            MAX_ORB_FEATURES="$2"
            shift 2
            ;;
        --match-ratio)
            MATCH_RATIO="$2"
            shift 2
            ;;
        --ransac-threshold)
            RANSAC_THRESHOLD="$2"
            shift 2
            ;;
        --max-canvas-ratio)
            MAX_CANVAS_RATIO="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            # Update dependent paths
            HOMOGRAPHY_DIR="${OUTPUT_DIR}/homographies"
            WARPED_DIR="${OUTPUT_DIR}/warped"
            MASKS_DIR="${OUTPUT_DIR}/masks"
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Main Processing Pipeline
# =============================================================================

main() {
    print_header "UDIS-D Warp Homography Preparation"
    
    echo "Configuration:"
    echo "  Image range: ${START_INDEX} to ${END_INDEX} (stride: ${STRIDE})"
    echo "  Batch size: ${BATCH_SIZE}"
    echo "  GPU device: ${GPU_ID}"
    echo "  Process timeout: ${PROCESS_TIMEOUT}s"
    echo ""
    echo "ORB+FLANN Parameters:"
    echo "  Max ORB features: ${MAX_ORB_FEATURES}"
    echo "  Match ratio: ${MATCH_RATIO}"
    echo "  RANSAC threshold: ${RANSAC_THRESHOLD}"
    echo "  Max canvas ratio: ${MAX_CANVAS_RATIO}"
    echo ""
    echo "Output directory: ${OUTPUT_DIR}"
    
    # Setup
    check_prerequisites
    setup_conda_environment
    create_output_directories
    
    # Calculate total pairs
    total_pairs=0
    for pair_idx in $(seq ${START_INDEX} ${STRIDE} ${END_INDEX}); do
        total_pairs=$((total_pairs + 1))
    done
    
    print_header "Processing ${total_pairs} Image Pairs"
    
    # Statistics tracking
    successful_pairs=0
    failed_pairs=()
    skipped_pairs=()
    
    current_pair=0
    batch_count=0
    
    # Process images in batches
    for pair_idx in $(seq ${START_INDEX} ${STRIDE} ${END_INDEX}); do
        current_pair=$((current_pair + 1))
        batch_index=$((current_pair % BATCH_SIZE))
        
        # Print batch header
        if [[ $batch_index -eq 1 ]]; then
            batch_count=$((batch_count + 1))
            echo ""
            echo "Processing batch ${batch_count}..."
            echo "=============================================================================="
        fi
        
        # Format pair index
        pair_idx_formatted=$(printf "%06d" $pair_idx)
        ref_image_path="${UDIS_D_TESTING_INPUT1_DIR}/${pair_idx_formatted}.jpg"
        tgt_image_path="${UDIS_D_TESTING_INPUT2_DIR}/${pair_idx_formatted}.jpg"
        
        # Check if images exist
        if [[ ! -f "${ref_image_path}" ]] || [[ ! -f "${tgt_image_path}" ]]; then
            echo "Pair ${current_pair}/${total_pairs} (${pair_idx}): Images not found, skipping"
            skipped_pairs+=($pair_idx)
            continue
        fi
        
        # Check if already processed
        homography_file="${HOMOGRAPHY_DIR}/${pair_idx_formatted}.txt"
        if [[ -f "${homography_file}" ]]; then
            echo "Pair ${current_pair}/${total_pairs} (${pair_idx}): Already processed, skipping"
            successful_pairs=$((successful_pairs + 1))
            continue
        fi
        
        # Process the pair
        echo "Pair ${current_pair}/${total_pairs} (${pair_idx}): Processing..."
        echo "  Ref: $(basename ${ref_image_path})"
        echo "  Tgt: $(basename ${tgt_image_path})"
        
        start_time=$(date +%s)
        
        # Run ORB+FLANN homography estimation with timeout
        if timeout ${PROCESS_TIMEOUT} python "${HOMOGRAPHY_SCRIPT_PATH}" \
            --ref_image "${ref_image_path}" \
            --tgt_image "${tgt_image_path}" \
            --output_dir "${OUTPUT_DIR}" \
            --pair_idx ${pair_idx} \
            --max_features ${MAX_ORB_FEATURES} \
            --match_ratio ${MATCH_RATIO} \
            --ransac_threshold ${RANSAC_THRESHOLD} \
            --max_canvas_ratio ${MAX_CANVAS_RATIO} \
            --quiet 2>/dev/null; then
            
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            
            # Verify output files were created
            ref_warp="${WARPED_DIR}/input1/${pair_idx_formatted}.png"
            tgt_warp="${WARPED_DIR}/input2/${pair_idx_formatted}.png"
            ref_mask="${MASKS_DIR}/input1/${pair_idx_formatted}.png"
            tgt_mask="${MASKS_DIR}/input2/${pair_idx_formatted}.png"
            
            if [[ -f "${homography_file}" ]] && [[ -f "${ref_warp}" ]] && \
               [[ -f "${tgt_warp}" ]] && [[ -f "${ref_mask}" ]] && [[ -f "${tgt_mask}" ]]; then
                
                successful_pairs=$((successful_pairs + 1))
                echo "  ✓ SUCCESS (${duration}s)"
            else
                failed_pairs+=($pair_idx)
                echo "  ✗ FAILED - output files missing"
            fi
        else
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            failed_pairs+=($pair_idx)
            
            if [[ $duration -ge $PROCESS_TIMEOUT ]]; then
                echo "  ✗ FAILED - timeout (${PROCESS_TIMEOUT}s)"
            else
                echo "  ✗ FAILED - processing error"
            fi
        fi
        
        # Print batch summary
        if [[ $batch_index -eq 0 ]] || [[ $current_pair -eq $total_pairs ]]; then
            success_rate=$((successful_pairs * 100 / current_pair))
            echo ""
            echo "Batch summary: ${successful_pairs}/${current_pair} successful (${success_rate}%)"
        fi
    done
    
    # Final summary
    print_header "Processing Complete"
    
    echo "Summary:"
    echo "  Total pairs: ${total_pairs}"
    echo "  Successful: ${successful_pairs}"
    echo "  Failed: ${#failed_pairs[@]}"
    echo "  Skipped (missing images): ${#skipped_pairs[@]}"
    
    if [[ ${#failed_pairs[@]} -gt 0 ]]; then
        echo ""
        echo "Failed pairs: ${failed_pairs[*]}"
    fi
    
    if [[ ${#skipped_pairs[@]} -gt 0 ]]; then
        echo ""
        echo "Skipped pairs: ${skipped_pairs[*]}"
    fi
    
    if [[ ${successful_pairs} -eq 0 ]]; then
        echo ""
        echo "Error: No pairs were processed successfully!"
        exit 1
    fi
    
    echo ""
    echo "Output files saved to:"
    echo "  Homographies: ${HOMOGRAPHY_DIR}/"
    echo "  Warped images: ${WARPED_DIR}/"
    echo "  Masks: ${MASKS_DIR}/"
    echo ""
    echo "These standardized warps can now be used to evaluate composition networks."
    
    # Deactivate conda environment
    conda deactivate
}

# Run main function
main "$@" 