FROM quay.io/jupyter/scipy-notebook

LABEL maintainer=stephenanthony@ieee.org

USER jovyan
WORKDIR /data
ADD https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_data_2/flair_2_toy_dataset.zip /data
# Download FLAIR #2 dataset

USER root
RUN chown jovyan flair_2_toy_dataset.zip
RUN chgrp 1000 flair_2_toy_dataset.zip
# Reassign to jovyan

USER jovyan
RUN unzip flair_2_toy_dataset.zip
# Unzip the FLAIR #2 dataset

USER root
RUN rm flair_2_toy_dataset.zip
# Delete the original .zip file

USER jovyan

# Install PyTorch with pip (https://pytorch.org/get-started/locally/)
# hadolint ignore=DL3013
RUN pip install --no-cache-dir --index-url 'https://download.pytorch.org/whl/cpu' \
    'torch' \
    'torchvision' \
    'torchaudio'  && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


WORKDIR /app

EXPOSE 8888