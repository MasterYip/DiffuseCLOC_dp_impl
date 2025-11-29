#!/bin/bash
"""
Script to start and manage wandb sweep for DiffuseCLoC parameter tuning.
This script helps you initialize the sweep and run sweep agents.
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}DiffuseCLoC Wandb Sweep Manager${NC}"
echo "================================="

# Check if wandb is installed and logged in
if ! command -v wandb &> /dev/null; then
    echo -e "${RED}Error: wandb is not installed. Please install with 'pip install wandb'${NC}"
    exit 1
fi

# # Check if logged in to wandb
# if ! wandb status | grep -q "logged in"; then
#     echo -e "${YELLOW}Warning: Not logged in to wandb. Please run 'wandb login' first.${NC}"
#     exit 1
# fi

# Function to initialize sweep
init_sweep() {
    echo -e "${GREEN}Initializing wandb sweep...${NC}"
    echo "Configuration file: sweep_config.yaml"
    echo "Training config: diffusion_policy/config_files/legged_gym_diffuse_sweep.yaml"
    echo ""
    
    # Initialize the sweep
    SWEEP_ID=$(wandb sweep sweep_config.yaml 2>&1 | grep -o 'wandb agent.*' | cut -d' ' -f3)
    
    if [ -z "$SWEEP_ID" ]; then
        echo -e "${RED}Failed to initialize sweep. Please check your configuration.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Sweep initialized successfully!${NC}"
    echo -e "${YELLOW}Sweep ID: ${SWEEP_ID}${NC}"
    echo ""
    echo "To start sweep agents, run:"
    echo "  ./start_sweep.sh agent $SWEEP_ID"
    echo ""
    echo "Or run agents manually:"
    echo "  wandb agent $SWEEP_ID"
    
    # Save sweep ID for later use
    echo "$SWEEP_ID" > .sweep_id
}

# Function to run sweep agent
run_agent() {
    SWEEP_ID=$1
    
    if [ -z "$SWEEP_ID" ]; then
        # Try to read from saved file
        if [ -f ".sweep_id" ]; then
            SWEEP_ID=$(cat .sweep_id)
            echo -e "${YELLOW}Using saved sweep ID: ${SWEEP_ID}${NC}"
        else
            echo -e "${RED}Error: No sweep ID provided and no saved sweep ID found.${NC}"
            echo "Usage: $0 agent <sweep_id>"
            exit 1
        fi
    fi
    
    echo -e "${GREEN}Starting wandb agent for sweep: ${SWEEP_ID}${NC}"
    echo "Press Ctrl+C to stop the agent"
    echo ""
    
    # Activate conda environment and run agent
    mamba activate pdplanner 2>/dev/null || echo -e "${YELLOW}Note: Could not activate pdplanner environment${NC}"
    
    # Run the sweep agent (config is handled by the training script)
    wandb agent $SWEEP_ID
}

# Function to show sweep status
show_status() {
    if [ -f ".sweep_id" ]; then
        SWEEP_ID=$(cat .sweep_id)
        echo -e "${GREEN}Current sweep ID: ${SWEEP_ID}${NC}"
        echo ""
        echo "Visit your sweep dashboard at:"
        echo "https://wandb.ai/$(wandb status | grep "Logged in" | cut -d' ' -f4)/diffuse_cloc/sweeps/$SWEEP_ID"
        echo ""
        echo "To run more agents:"
        echo "  ./start_sweep.sh agent $SWEEP_ID"
    else
        echo -e "${YELLOW}No active sweep found. Initialize one with:${NC}"
        echo "  ./start_sweep.sh init"
    fi
}

# Function to show help
show_help() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  init              Initialize a new wandb sweep"
    echo "  agent [sweep_id]  Start a sweep agent (uses saved sweep ID if not provided)"
    echo "  status            Show current sweep status"
    echo "  help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 init                                    # Initialize new sweep"
    echo "  $0 agent                                   # Start agent with saved sweep ID"
    echo "  $0 agent your-entity/project/sweep_id     # Start agent with specific sweep ID"
    echo "  $0 status                                  # Show sweep status"
}

# Main script logic
case "${1:-help}" in
    "init")
        init_sweep
        ;;
    "agent")
        run_agent "$2"
        ;;
    "status")
        show_status
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac