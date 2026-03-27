#!/bin/bash
# setup_env.sh (Micromamba)

# 1. Dynamically find the repository root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# 2. Define Micromamba paths (keeping everything isolated inside your repo)
MAMBA_BIN_DIR="$PROJECT_ROOT/micromamba_bin"
MAMBA_BIN="$MAMBA_BIN_DIR/micromamba"
export MAMBA_ROOT_PREFIX="$PROJECT_ROOT/micromamba_root"
ENV_NAME="itransformer_env"

echo "Checking environment setup..."

# 3. Download the Micromamba binary if it doesn't exist
if [ ! -f "$MAMBA_BIN" ]; then
    echo "⚡ Micromamba not found. Downloading..."
    mkdir -p "$MAMBA_BIN_DIR"
    # Pull the latest Linux-64 binary and extract just the executable
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$MAMBA_BIN_DIR" --strip-components=1 bin/micromamba
else
    echo "✅ Micromamba already installed."
fi

# 4. Dynamically initialize Micromamba for this subshell
eval "$("$MAMBA_BIN" shell hook -s bash)"

# 5. Check if our specific environment exists
if [ ! -d "$MAMBA_ROOT_PREFIX/envs/$ENV_NAME" ]; then
    echo "⚡ Environment '$ENV_NAME' not found. Creating it blazing fast..."
    
    # Create the environment with Python 3.10 using the conda-forge channel
    micromamba create -y -n "$ENV_NAME" python=3.10 -c conda-forge
    
    # Activate it
    micromamba activate "$ENV_NAME"
    
    echo "⚡ Installing requirements.txt..."
    pip install -q -r "$PROJECT_ROOT/requirements.txt"
    echo "✅ Dependencies installed successfully!"
else
    # If it exists, just activate it
    echo "✅ Environment '$ENV_NAME' found. Activating..."
    micromamba activate "$ENV_NAME"
fi

# 6. Force CUDA visibility for PyTorch
export CUDA_VISIBLE_DEVICES=0

echo "🚀 Environment is ready. Starting script..."
echo "=========================================="