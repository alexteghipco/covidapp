#!/bin/bash
# JOB HEADERS HERE

#SBATCH --partition=v100-32gb-hiprio
#SBATCH --cpus-per-task=42
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --output=jupyter_log_%j.out
#SBATCH --error=jupyter_error_%j.err
#SBATCH --time=10:00:00

# Load necessary modules (modify based on your system)
module load python3/anaconda/2023.9
module load cuda/12.3

# Activate your Conda environment
source activate llm_v100

# Choose a fixed port or let Jupyter select one
PORT=8888

export JUPYTER_CONFIG_DIR=/tmp/jupyter_config
export JUPYTER_RUNTIME_DIR=/tmp/jupyter_runtime

mkdir -p /tmp/jupyter_config
mkdir -p /tmp/jupyter_runtime

# Start Jupyter Notebook
# --no-browser: Don't open a browser
# --ip=0.0.0.0: Listen on all interfaces (required for SSH tunneling)
# --port: Specify the port
# --NotebookApp.token='': Disable token authentication (optional, less secure)
jupyter notebook --no-browser --ip=0.0.0.0 --port=$PORT --NotebookApp.token='' > jupyter_notebook.log 2>&1 &

# Get the Jupyter process ID
JUPYTER_PID=$!

# Wait for Jupyter to start
sleep 5

# Check if Jupyter started successfully
if ps -p $JUPYTER_PID > /dev/null; then
    echo "Jupyter Notebook is running on port $PORT"
    echo "Access it via SSH tunneling."
else
    echo "Failed to start Jupyter Notebook. Check jupyter_notebook.log for details."
    exit 1
fi

# Keep the job running to maintain Jupyter Notebook
wait $JUPYTER_PID
