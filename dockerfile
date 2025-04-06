# Jupyter-compatible base image
FROM jupyter/scipy-notebook:latest

# Set working directory
WORKDIR /home/jovyan/work

# Copy notebooks, source code, requirements
COPY data/ ./data/
COPY notebooks_jupyter/ ./notebooks/
COPY src/ ./src/
COPY requirements_jupyter.txt ./

# Install system dependencies
RUN pip install --no-cache-dir -r requirements_jupyter.txt

# Default: open Jupyter Notebook in the browser
CMD ["start-notebook.sh", "--NotebookApp.notebook_dir=/home/jovyan/work/notebooks"]