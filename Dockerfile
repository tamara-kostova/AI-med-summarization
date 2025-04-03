# Ignore everything by default
*

# Explicitly allow necessary files and directories
!app/
!summarization/
!requirements.txt
!Dockerfile

# Python specifics
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
venv/
.env
*.venv

# Large data directories to explicitly ignore
data/
datasets/
*.dataset
*.parquet
*.arrow

# Hugging Face cache
.cache/
huggingface/

# Large model files
model_checkpoints/
*.pt
*.pth
*.bin

# Logs and temporary files
*.log
*.tmp

# IDE and editor folders
.vscode/
.idea/
*.swp
*~

# Operating system files
.DS_Store
Thumbs.db

# Git
.git
.gitignore