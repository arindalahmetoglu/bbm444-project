#!/bin/bash

# =============================================================================
# 
# This script evaluates all three composition methods on UDIS-D dataset:
# 1. NIS - Using standardized ORB+FLANN warps directly
# 2. UDIS - Using ORB+FLANN warps converted to UDIS-compatible format
# 3. UDIS++ - Using ORB+FLANN warps converted to UDIS-compatible format
#
# APPROACH:
# - For NIS: Use ORB+FLANN warps from prepare_warp_homographies_udisd.sh directly
# - For UDIS/UDIS++: Convert ORB+FLANN warps to sift_preprocessed_data format
#
# This ensures all methods use the same standardized geometric warps
# for a fair comparative evaluation.
#
# =============================================================================

set -e  # Exit on any error

# =============================================================================
# Configuration and Default Parameters
# =============================================================================

# Default processing parameters
START_INDEX=1
END_INDEX=100
STRIDE=1
BATCH_SIZE=5
GPU_ID=0
PROCESS_TIMEOUT=300

# Methods to run (can be overridden)
RUN_NIS=true
RUN_UDIS=true
RUN_UDIS_PLUS_PLUS=true

# Directory structure
PROJECT_ROOT=$(pwd)
WARPS_DIR="${PROJECT_ROOT}/processing_data/udis_d_warps"  # ORB+FLANN warps
SIFT_COMPAT_DIR="${PROJECT_ROOT}/processing_data/sift_preprocessed_data"  # UDIS-compatible format
OUTPUT_DIR_BASE="${PROJECT_ROOT}/results/udis_d_results"

# UDIS-D dataset
DATASET_DIR="${PROJECT_ROOT}/UDIS-D/testing"
INPUT1_DIR="${DATASET_DIR}/input1"
INPUT2_DIR="${DATASET_DIR}/input2"

# Conda environments
CONDA_SETUP_SCRIPT=""  # Will be set during setup

# Method execution scripts - Using correct working versions
NIS_SCRIPT="${PROJECT_ROOT}/run_nis.sh"
UDIS_SCRIPT="${PROJECT_ROOT}/run_udis.sh"
UDIS_PLUS_PLUS_SCRIPT="${PROJECT_ROOT}/run_udis_plus_plus.sh"

# Conversion script
PREPARE_INPUTS_SCRIPT="${PROJECT_ROOT}/prepare_udisd_inputs.py"

# Method-specific output directories
NIS_RESULTS_DIR="${OUTPUT_DIR_BASE}/nis"

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

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Evaluate all composition methods on UDIS-D dataset using standardized ORB+FLANN warps.

OPTIONS:
    --start N              Starting image index (default: ${START_INDEX})
    --end N                Ending image index (default: ${END_INDEX})
    --stride N             Step between indices (default: ${STRIDE})
    --batch-size N         Process N images at a time (default: ${BATCH_SIZE})
    --gpu N                GPU device ID (default: ${GPU_ID})
    --timeout N            Process timeout in seconds (default: ${PROCESS_TIMEOUT})
    
    --output-dir PATH      Output directory (default: ${OUTPUT_DIR_BASE})
    
    --methods METHODS      Comma-separated methods to run: nis,udis,udis_plus_plus
                          Or 'all' for all methods (default: all)
    
    --nis-only            Run only NIS
    --udis-only           Run only UDIS  
    --udis-plus-plus-only Run only UDIS++
    
    --help, -h            Show this help message

EXAMPLES:
    $0 --start 1 --end 50 --stride 5
    $0 --start 100 --end 200 --nis-only
    $0 --methods "nis,udis" --batch-size 10

PREREQUISITES:
    1. Run prepare_warp_homographies_udisd.sh for ORB+FLANN warps
    2. Ensure UDIS-D dataset is available at ${DATASET_DIR}

EOF
}

check_prerequisites() {
    print_section "Checking Prerequisites"
    
    # Check UDIS-D dataset
    if [[ ! -d "${INPUT1_DIR}" ]] || [[ ! -d "${INPUT2_DIR}" ]]; then
        echo "Error: UDIS-D dataset not found!"
        echo "Expected directories:"
        echo "  ${INPUT1_DIR}"
        echo "  ${INPUT2_DIR}"
        exit 1
    fi
    
    # Check ORB+FLANN warps
    if [[ ! -d "${WARPS_DIR}" ]]; then
        echo "Error: ORB+FLANN warps not found!"
        echo "Expected directory: ${WARPS_DIR}"
        echo ""
        echo "Please run prepare_warp_homographies_udisd.sh first."
        exit 1
    fi
    
    # Check method execution scripts
    if [[ $RUN_NIS == "true" ]] && [[ ! -f "${NIS_SCRIPT}" ]]; then
        echo "Error: NIS execution script not found: ${NIS_SCRIPT}"
        exit 1
    fi
    
    if [[ $RUN_UDIS == "true" ]] && [[ ! -f "${UDIS_SCRIPT}" ]]; then
        echo "Error: UDIS execution script not found: ${UDIS_SCRIPT}"
        exit 1
    fi
    
    if [[ $RUN_UDIS_PLUS_PLUS == "true" ]] && [[ ! -f "${UDIS_PLUS_PLUS_SCRIPT}" ]]; then
        echo "Error: UDIS++ execution script not found: ${UDIS_PLUS_PLUS_SCRIPT}"
        exit 1
    fi
    
    # Check conversion script for UDIS/UDIS++
    if [[ ($RUN_UDIS == "true" || $RUN_UDIS_PLUS_PLUS == "true") ]] && [[ ! -f "${PREPARE_INPUTS_SCRIPT}" ]]; then
        echo "Error: Conversion script not found: ${PREPARE_INPUTS_SCRIPT}"
        exit 1
    fi
    
    echo "✓ UDIS-D dataset found"
    echo "✓ ORB+FLANN warps found: ${WARPS_DIR}"
    
    # Count available warps
    homography_count=$(find "${WARPS_DIR}/homographies" -name "*.txt" 2>/dev/null | wc -l)
    echo "✓ Available warp pairs: ${homography_count}"
    
    # Check method scripts
    if [[ $RUN_NIS == "true" ]]; then
        echo "✓ NIS script found: ${NIS_SCRIPT}"
    fi
    if [[ $RUN_UDIS == "true" ]]; then
        echo "✓ UDIS script found: ${UDIS_SCRIPT}"
    fi
    if [[ $RUN_UDIS_PLUS_PLUS == "true" ]]; then
        echo "✓ UDIS++ script found: ${UDIS_PLUS_PLUS_SCRIPT}"
    fi
    
    if [[ ($RUN_UDIS == "true" || $RUN_UDIS_PLUS_PLUS == "true") ]]; then
        echo "✓ Warp conversion script found: ${PREPARE_INPUTS_SCRIPT}"
    fi
}

setup_conda_environment() {
    print_section "Setting up Conda Environment"
    
    # Find conda setup script and set global variable
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
    
    echo "✓ Conda environment setup ready"
}

create_output_directories() {
    print_section "Creating Output Directories"
    
    mkdir -p "${OUTPUT_DIR_BASE}"
    
    if [[ $RUN_NIS == "true" ]]; then
        mkdir -p "${NIS_RESULTS_DIR}"
    fi
    
    if [[ $RUN_UDIS == "true" ]]; then
        mkdir -p "${OUTPUT_DIR_BASE}/udis"
    fi
    
    if [[ $RUN_UDIS_PLUS_PLUS == "true" ]]; then
        mkdir -p "${OUTPUT_DIR_BASE}/udis_plus_plus"
    fi
    
    echo "Output directories created:"
    if [[ $RUN_NIS == "true" ]]; then
        echo "  NIS: ${NIS_RESULTS_DIR}"
    fi
    if [[ $RUN_UDIS == "true" ]]; then
        echo "  UDIS: ${OUTPUT_DIR_BASE}/udis/"
    fi
    if [[ $RUN_UDIS_PLUS_PLUS == "true" ]]; then
        echo "  UDIS++: ${OUTPUT_DIR_BASE}/udis_plus_plus/"
    fi
}

setup_udis_compatibility_layer() {
    print_section "Setting up UDIS/UDIS++ Compatibility Layer"
    
    if [[ $RUN_UDIS == "true" ]] || [[ $RUN_UDIS_PLUS_PLUS == "true" ]]; then
        echo "Converting ORB+FLANN warps to UDIS-compatible format..."
        echo "This allows UDIS/UDIS++ to use the same standardized warps as NIS."
        
        # Convert all available ORB+FLANN warps to UDIS format
        if python "${PREPARE_INPUTS_SCRIPT}" \
            --warps-dir "${WARPS_DIR}" \
            --output-dir "${SIFT_COMPAT_DIR}"; then
            
            echo "✓ Successfully converted ORB+FLANN warps to UDIS format"
            echo "  UDIS/UDIS++ compatibility data: ${SIFT_COMPAT_DIR}"
        else
            echo "✗ Failed to convert ORB+FLANN warps to UDIS format"
            if [[ $RUN_UDIS == "true" ]]; then
                echo "  Disabling UDIS due to conversion failure"
                RUN_UDIS=false
            fi
            if [[ $RUN_UDIS_PLUS_PLUS == "true" ]]; then
                echo "  Disabling UDIS++ due to conversion failure"
                RUN_UDIS_PLUS_PLUS=false
            fi
        fi
    fi
}

run_nis_for_pair() {
    local pair_idx=$1
    local pair_idx_formatted=$(printf "%06d" $pair_idx)
    
    echo "Running NIS..."
    if ! timeout ${PROCESS_TIMEOUT} bash "${NIS_SCRIPT}" "${pair_idx}" "${pair_idx}" 1 sift; then
        echo "  ✗ NIS failed - processing error or timeout"
        return 1
    fi
    
    # The run_nis.sh script saves output to a specific folder. Find it.
    local nis_output_pattern="processing_data/enhanced_features_results/sift_flann/*${pair_idx_formatted}*"
    local nis_output_file=$(ls -t ${nis_output_pattern} 2>/dev/null | head -1)
    
    if [[ -f "${nis_output_file}" ]]; then
        cp "${nis_output_file}" "${NIS_RESULTS_DIR}/${pair_idx_formatted}.png"
        echo "  ✓ NIS completed"
        return 0
    else
        echo "  ✗ NIS failed - output file not found at pattern: ${nis_output_pattern}"
        return 1
    fi
}

run_udis_for_pair() {
    local pair_idx=$1
    local pair_idx_formatted=$(printf "%06d" $pair_idx)

    echo "Running UDIS..."
    if ! timeout ${PROCESS_TIMEOUT} bash "${UDIS_SCRIPT}" "${pair_idx}" "${pair_idx}" 1; then
        echo "  ✗ UDIS failed - processing error or timeout"
        return 1
    fi

    local udis_output_file="${PROJECT_ROOT}/processing_data/global_stitching_results/UDIS/${pair_idx_formatted}.jpg"
    if [[ -f "${udis_output_file}" ]]; then
        echo "  ✓ UDIS completed"
        rm -f "${udis_output_file}" # Clean up intermediate file
        return 0
    else
        echo "  ✗ UDIS failed - output file not found at: ${udis_output_file}"
        return 1
    fi
}

run_udis_plus_plus_for_pair() {
    local pair_idx=$1
    local pair_idx_formatted=$(printf "%06d" $pair_idx)
    
    echo "Running UDIS++..."
    if ! timeout ${PROCESS_TIMEOUT} bash "${UDIS_PLUS_PLUS_SCRIPT}" "${pair_idx}" "${pair_idx}" 1; then
        echo "  ✗ UDIS++ failed - processing error or timeout"
        return 1
    fi
    
    local udis_pp_output_file="${PROJECT_ROOT}/processing_data/global_stitching_results/UDIS_plus_plus/${pair_idx_formatted}_stitched.jpg"
    if [[ -f "${udis_pp_output_file}" ]]; then
        cp "${udis_pp_output_file}" "${OUTPUT_DIR_BASE}/udis_plus_plus/"
        echo "  ✓ UDIS++ completed"
        rm -f "${udis_pp_output_file}" # Clean up intermediate file
        return 0
    else
        echo "  ✗ UDIS++ failed - output file not found at: ${udis_pp_output_file}"
        return 1
    fi
}

# =============================================================================
# Command Line Argument Parsing
# =============================================================================

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
        --output-dir)
            OUTPUT_DIR_BASE="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            if [[ "$METHODS" == "all" ]]; then
                RUN_NIS=true
                RUN_UDIS=true
                RUN_UDIS_PLUS_PLUS=true
            else
                RUN_NIS=false
                RUN_UDIS=false
                RUN_UDIS_PLUS_PLUS=false
                IFS=',' read -ra METHOD_ARRAY <<< "$METHODS"
                for method in "${METHOD_ARRAY[@]}"; do
                    case $method in
                        nis) RUN_NIS=true ;;
                        udis) RUN_UDIS=true ;;
                        udis_plus_plus) RUN_UDIS_PLUS_PLUS=true ;;
                        *) echo "Unknown method: $method"; exit 1 ;;
                    esac
                done
            fi
            shift 2
            ;;
        --nis-only)
            RUN_NIS=true
            RUN_UDIS=false
            RUN_UDIS_PLUS_PLUS=false
            shift
            ;;
        --udis-only)
            RUN_NIS=false
            RUN_UDIS=true
            RUN_UDIS_PLUS_PLUS=false
            shift
            ;;
        --udis-plus-plus-only)
            RUN_NIS=false
            RUN_UDIS=false
            RUN_UDIS_PLUS_PLUS=true
            shift
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
    print_header "UDIS-D All Methods Evaluation (Standardized ORB+FLANN Warps)"
    
    echo "Configuration:"
    echo "  Image range: ${START_INDEX} to ${END_INDEX} (stride: ${STRIDE})"
    echo "  Batch size: ${BATCH_SIZE}"
    echo "  GPU device: ${GPU_ID}"
    echo "  Process timeout: ${PROCESS_TIMEOUT}s"
    echo ""
    echo "Methods to run:"
    if [[ $RUN_NIS == "true" ]]; then echo "  ✓ NIS - Using ORB+FLANN warps directly"; fi
    if [[ $RUN_UDIS == "true" ]]; then echo "  ✓ UDIS - Using converted ORB+FLANN warps"; fi
    if [[ $RUN_UDIS_PLUS_PLUS == "true" ]]; then echo "  ✓ UDIS++ - Using converted ORB+FLANN warps"; fi
    echo ""
    echo "Output directory: ${OUTPUT_DIR_BASE}"
    
    # Setup
    check_prerequisites
    setup_conda_environment
    create_output_directories
    setup_udis_compatibility_layer
    
    # Generate list of pairs to process
    pairs_to_process=()
    for pair_idx in $(seq ${START_INDEX} ${STRIDE} ${END_INDEX}); do
        pair_idx_formatted=$(printf "%06d" $pair_idx)
        
        # Check if images exist
        if [[ -f "${INPUT1_DIR}/${pair_idx_formatted}.jpg" ]] && \
           [[ -f "${INPUT2_DIR}/${pair_idx_formatted}.jpg" ]]; then
            
            # Check if ORB+FLANN warp exists
            if [[ -f "${WARPS_DIR}/homographies/${pair_idx_formatted}.txt" ]]; then
                pairs_to_process+=($pair_idx)
            fi
        fi
    done
    
    total_pairs=${#pairs_to_process[@]}
    
    if [[ $total_pairs -eq 0 ]]; then
        echo "Error: No valid pairs found in the specified range!"
        echo "Please check:"
        echo "  - ORB+FLANN warps have been generated using prepare_warp_homographies_udisd.sh"
        echo "  - UDIS-D dataset images exist"
        exit 1
    fi
    
    print_header "Processing ${total_pairs} Image Pairs"
    
    # Statistics tracking
    nis_success=0
    udis_success=0
    udis_plus_plus_success=0
    nis_failed=()
    udis_failed=()
    udis_plus_plus_failed=()
    
    current_pair=0
    batch_count=0
    
    # Process pairs in batches
    for pair_idx in "${pairs_to_process[@]}"; do
        current_pair=$((current_pair + 1))
        batch_index=$((current_pair % BATCH_SIZE))
        
        # Print batch header
        if [[ $batch_index -eq 1 ]]; then
            batch_count=$((batch_count + 1))
            echo ""
            echo "Processing batch ${batch_count}..."
            echo "=============================================================================="
        fi
        
        echo ""
        echo "Processing pair ${current_pair}/${total_pairs}: ${pair_idx}"
        echo "  Using standardized ORB+FLANN warps for all methods"
        
        # Run each method
        if [[ $RUN_NIS == "true" ]]; then
            if run_nis_for_pair $pair_idx; then
                nis_success=$((nis_success + 1))
            else
                nis_failed+=($pair_idx)
            fi
        fi
        
        if [[ $RUN_UDIS == "true" ]]; then
            if run_udis_for_pair $pair_idx; then
                udis_success=$((udis_success + 1))
            else
                udis_failed+=($pair_idx)
            fi
        fi
        
        if [[ $RUN_UDIS_PLUS_PLUS == "true" ]]; then
            if run_udis_plus_plus_for_pair $pair_idx; then
                udis_plus_plus_success=$((udis_plus_plus_success + 1))
            else
                udis_plus_plus_failed+=($pair_idx)
            fi
        fi
        
        # Print batch summary
        if [[ $batch_index -eq 0 ]] || [[ $current_pair -eq $total_pairs ]]; then
            echo ""
            echo "Batch summary (${current_pair}/${total_pairs} pairs processed):"
            if [[ $RUN_NIS == "true" ]]; then
                success_rate=$((nis_success * 100 / current_pair))
                echo "  NIS: ${nis_success}/${current_pair} (${success_rate}%)"
            fi
            if [[ $RUN_UDIS == "true" ]]; then
                success_rate=$((udis_success * 100 / current_pair))
                echo "  UDIS: ${udis_success}/${current_pair} (${success_rate}%)"
            fi
            if [[ $RUN_UDIS_PLUS_PLUS == "true" ]]; then
                success_rate=$((udis_plus_plus_success * 100 / current_pair))
                echo "  UDIS++: ${udis_plus_plus_success}/${current_pair} (${success_rate}%)"
            fi
        fi
    done
    
    # Final summary
    print_header "All Methods Evaluation Complete"
    
    echo "Processing Summary:"
    echo "  Total pairs processed: ${total_pairs}"
    echo "  All methods used the same standardized ORB+FLANN warps"
    echo ""
    
    if [[ $RUN_NIS == "true" ]]; then
        success_rate=$((nis_success * 100 / total_pairs))
        echo "NIS Results (direct ORB+FLANN warps):"
        echo "  Successful: ${nis_success}/${total_pairs} (${success_rate}%)"
        if [[ ${#nis_failed[@]} -gt 0 ]]; then
            echo "  Failed pairs: ${nis_failed[*]}"
        fi
        echo ""
    fi
    
    if [[ $RUN_UDIS == "true" ]]; then
        success_rate=$((udis_success * 100 / total_pairs))
        echo "UDIS Results (converted ORB+FLANN warps):"
        echo "  Successful: ${udis_success}/${total_pairs} (${success_rate}%)"
        if [[ ${#udis_failed[@]} -gt 0 ]]; then
            echo "  Failed pairs: ${udis_failed[*]}"
        fi
        echo ""
    fi
    
    if [[ $RUN_UDIS_PLUS_PLUS == "true" ]]; then
        success_rate=$((udis_plus_plus_success * 100 / total_pairs))
        echo "UDIS++ Results (converted ORB+FLANN warps):"
        echo "  Successful: ${udis_plus_plus_success}/${total_pairs} (${success_rate}%)"
        if [[ ${#udis_plus_plus_failed[@]} -gt 0 ]]; then
            echo "  Failed pairs: ${udis_plus_plus_failed[*]}"
        fi
        echo ""
    fi
    
    echo "Output directories:"
    if [[ $RUN_NIS == "true" ]]; then
        echo "  NIS: ${NIS_RESULTS_DIR} (${nis_success} images)"
    fi
    if [[ $RUN_UDIS == "true" ]]; then
        echo "  UDIS: ${OUTPUT_DIR_BASE}/udis/ (${udis_success} images)"
    fi
    if [[ $RUN_UDIS_PLUS_PLUS == "true" ]]; then
        echo "  UDIS++: ${OUTPUT_DIR_BASE}/udis_plus_plus/ (${udis_plus_plus_success} images)"
    fi
    
    echo ""
    echo " Standardized evaluation completed successfully!"
    echo " All methods used the same ORB+FLANN geometric warps for fair comparison."
    echo " Results are ready for quantitative analysis of composition methods only."
    
    # Final cleanup
    conda deactivate 2>/dev/null || true
}

# Run main function
main "$@" 