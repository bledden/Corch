#!/bin/bash
# Simple CLI dashboard to monitor running tests
# Usage: ./watch_tests.sh [bash_id1] [bash_id2] [bash_id3]

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Default test IDs if none provided
if [ $# -eq 0 ]; then
    TESTS=("fe6e5e" "8c42af" "d9ac2f")
else
    TESTS=("$@")
fi

# Function to get test progress from process
get_test_info() {
    local test_id=$1
    local tmp_file="/tmp/test_${test_id}.txt"

    # This would need to be integrated with Claude Code's BashOutput tool
    # For now, show placeholder
    echo "Loading..."
}

# Clear screen and show header
clear_and_header() {
    clear
    echo -e "${CYAN}${BOLD}+===============================================================+${NC}"
    echo -e "${CYAN}${BOLD}|           Facilitair Test Monitor Dashboard                  |${NC}"
    echo -e "${CYAN}${BOLD}+===============================================================+${NC}"
    echo ""
    echo -e "${BOLD}Monitoring ${#TESTS[@]} tests${NC} | Press Ctrl+C to exit"
    echo ""
}

# Main monitoring loop
while true; do
    clear_and_header

    echo -e "${BOLD}Test ID${NC}     ${BOLD}Status${NC}      ${BOLD}Progress${NC}              ${BOLD}Results${NC}"
    echo "----------------------------------------------------------------"

    for test_id in "${TESTS[@]}"; do
        # Check if process exists (simplified - would need actual implementation)
        if ps aux | grep -q "[p]ython.*${test_id}"; then
            status="${GREEN}* RUNNING${NC}"
            progress="[..........]"  # Would parse from output
            results="Pending..."
        else
            status="${YELLOW}o UNKNOWN${NC}"
            progress="----------"
            results="N/A"
        fi

        echo -e "${CYAN}${test_id}${NC}   ${status}   ${progress}   ${results}"
    done

    echo ""
    echo "----------------------------------------------------------------"
    echo -e "${BOLD}Active Tests:${NC}"

    # Show running Python test processes
    echo -e "${BLUE}"
    ps aux | grep -E "[p]ython.*(smoke_test|eval)" | awk '{print "  PID " $2 ": " $11 " " $12 " " $13}' | head -10
    echo -e "${NC}"

    echo ""
    echo -e "${MAGENTA}Last updated: $(date '+%H:%M:%S')${NC}"
    echo -e "${BOLD}Note:${NC} Use 'python3 monitor_tests.py' for detailed live monitoring"

    sleep 3
done
