#!/bin/bash
set -e

# Base Config
CONFIG_TEMPLATE="lora_config.yaml"
BASE_CMD="python3 -m mlx_lm.lora --config"

ranks=(8 16 32)
# We want to test different alpha values. 
# scale = alpha / rank.
# If we want alpha=16, 32, 64:
# For r=8:  a=16->s=2.0, a=32->s=4.0
# For r=16: a=32->s=2.0, a=64->s=4.0
# For r=32: a=64->s=2.0, a=128->s=4.0
# So we can just sweep scale=2.0 and scale=4.0 for all ranks to maintain consistent alpha/r ratio.

scales=(2.0 4.0)

for r in "${ranks[@]}"; do
    for s in "${scales[@]}"; do
        
        ADAPTER_NAME="tinyllama-lora-r${r}-s${s}"
        CONFIG_FILE="lora_config_r${r}_s${s}.yaml"
        
        echo "Creating config for Rank=$r, Scale=$s -> $ADAPTER_NAME"
        
        # Create temp config by replacing values in template
        # We use sed to replace the specific lines. 
        # Only creating a temporary file for this run.
        cp $CONFIG_TEMPLATE $CONFIG_FILE
        
        # Replace rank
        sed -i '' "s/rank: [0-9]*/rank: $r/" $CONFIG_FILE
        # Replace scale
        sed -i '' "s/scale: [0-9.]*/scale: $s/" $CONFIG_FILE
        # Replace adapter_path
        sed -i '' "s|adapter_path: .*|adapter_path: \"./adapters/$ADAPTER_NAME\"|" $CONFIG_FILE
        
        echo "Starting training for $ADAPTER_NAME..."
        python3 -m mlx_lm.lora --config $CONFIG_FILE
        
        echo "Finished $ADAPTER_NAME"
        rm $CONFIG_FILE
        echo "------------------------------------------------"
        
    done
done

echo "Sweep complete!"
