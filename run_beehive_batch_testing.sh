#!/bin/bash

# ==============================================================================
# Script to run a series of stitching tests on the
# Beehive dataset. It processes predefined adjacent image pairs, allowing for
# easy expansion and configuration.
# ==============================================================================

# --- Configuration ---
# Set  default parameters here. Users can override these with command-line flags.
METHODS_TO_RUN="all"
SCALE_FACTOR="0.5"
RANDOMIZE_ORDER=false

# Default translations for different pair types.
DX_HORIZONTAL=490
DY_HORIZONTAL=330
DX_VERTICAL=0
DY_VERTICAL=330

# --- Helper Functions for Logging ---
C_INFO='\033[0;36m'    # Cyan
C_SUCCESS='\033[0;32m' # Green
C_ERROR='\033[0;31m'   # Red
C_RESET='\033[0m'      # Reset color

log_info() {
    echo -e "${C_INFO}[INFO]${C_RESET} $1"
}

log_success() {
    echo -e "${C_SUCCESS}[SUCCESS]${C_RESET} $1"
}

log_error() {
    echo -e "${C_ERROR}[ERROR]${C_RESET} $1" >&2
}

print_header() {
    echo ""
    echo -e "${C_SUCCESS}=======================================================================${C_RESET}"
    echo -e "${C_SUCCESS} Beehive Batch Test Runner ${C_RESET}"
    echo -e "${C_SUCCESS}=======================================================================${C_RESET}"
    echo ""
}

# --- Define the work to be done ---
# An array of strings, where each string defines a test case.
# Format: "ref_scan ref_img tgt_scan tgt_img dx dy"
# This makes it easy to add or remove pairs.
declare -a TEST_PAIRS=(
    # Horizontal pairs (left-to-right)
    "1 1 1 2 ${DX_HORIZONTAL} ${DY_HORIZONTAL}"
    "2 1 2 2 ${DX_HORIZONTAL} ${DY_HORIZONTAL}"

    # Vertical pairs (top-to-bottom)
    "1 1 2 1 ${DX_VERTICAL} ${DY_VERTICAL}"
    "1 2 2 2 ${DX_VERTICAL} ${DY_VERTICAL}"
)

# --- Script Logic ---

# Function to display help text
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "A friendly batch tester for the Beehive stitching project."
    echo ""
    echo "Options:"
    echo "  --methods <METHODS>   Methods to run (e.g., 'all', 'nis,udis'). Default: '${METHODS_TO_RUN}'"
    echo "  --scale <FACTOR>      Image scaling factor for processing. Default: '${SCALE_FACTOR}'"
    echo "  --randomize           Randomize the processing order of image pairs."
    echo "  -h, --help            Display this help message."
    echo ""
    echo "Note: Translations can be configured directly inside the script."
}

# Parse command-line arguments in a simple loop
while [[ $# -gt 0 ]]; do
    case "$1" in
        --methods)
            METHODS_TO_RUN="$2"
            shift 2
            ;;
        --scale)
            SCALE_FACTOR="$2"
            shift 2
            ;;
        --randomize)
            RANDOMIZE_ORDER=true
            shift
            ;;
        -h | --help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# --- Main Execution ---
print_header

log_info "Starting batch process with the following settings:"
log_info "  - Methods to run: ${METHODS_TO_RUN}"
log_info "  - Scale Factor:   ${SCALE_FACTOR}"
log_info "  - Randomize:      ${RANDOMIZE_ORDER}"
echo ""

# Randomize the array if the user asked for it.
if [ "$RANDOMIZE_ORDER" = true ]; then
    log_info "Randomizing the order of test pairs..."
    readarray -t TEST_PAIRS < <(printf "%s\n" "${TEST_PAIRS[@]}" | shuf)
    echo ""
fi

# Keep track of progress and failures
TOTAL_PAIRS=${#TEST_PAIRS[@]}
processed_count=0
failed_count=0
start_time_total=$SECONDS

# Loop through each defined pair and run the test
for pair_definition in "${TEST_PAIRS[@]}"; do
    ((processed_count++))
    
    # Easily read the variables from the definition string
    read -r ref_scan ref_img tgt_scan tgt_img dx dy <<< "$pair_definition"

    pair_id="Scan${ref_scan}_Img${ref_img}-to-Scan${tgt_scan}_Img${tgt_img}"
    
    echo "------------------------------------------------------------------------"
    log_info "Running Test ${processed_count} of ${TOTAL_PAIRS}: ${pair_id}"
    echo "------------------------------------------------------------------------"
    
    start_time_pair=$SECONDS

    # Call the main processing script
    ./run_beehive_methods.sh \
        "$ref_scan" "$ref_img" \
        "$tgt_scan" "$tgt_img" \
        "$METHODS_TO_RUN" \
        "$dx" "$dy" \
        "$SCALE_FACTOR"

    # Check the result and log it
    if [ $? -eq 0 ]; then
        duration=$(( SECONDS - start_time_pair ))
        log_success "Pair ${pair_id} processed successfully in ${duration}s."
    else
        duration=$(( SECONDS - start_time_pair ))
        log_error "Pair ${pair_id} failed after ${duration}s."
        ((failed_count++))
    fi
    echo ""
done

# --- Final Report ---
duration_total=$(( SECONDS - start_time_total ))
echo "========================================================================"
log_info "Batch Processing Complete"
echo "========================================================================"
log_info "Total time: ${duration_total} seconds."
log_info "Pairs processed: ${TOTAL_PAIRS}"

if [ $failed_count -eq 0 ]; then
    log_success "All tests passed! "
else
    log_error "${failed_count} test(s) failed. Please check the logs above for details."
fi
echo "========================================================================"

# Exit with a status code indicating the number of failures
exit ${failed_count} 