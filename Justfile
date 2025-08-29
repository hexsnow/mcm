# To use this file, install just: https://github.com/casey/just

# Default command, prints help message
default:
    @echo "Available commands:"
    @echo "  install - Install the project in editable mode"
    @echo "  run     - Run the main entrypoint"
    @echo "  clean   - Clean up build artifacts and cache"

# Install the project in editable mode
install:
    pip install -e .

# Run the main entrypoint
run:
    mcm

# Clean up build artifacts and cache
clean:
    rm -rf .venv
    rm -rf dist
    rm -rf *.egg-info
    find . -name '__pycache__' -type d -exec rm -rf {} +
