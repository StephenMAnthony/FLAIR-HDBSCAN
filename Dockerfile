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
WORKDIR /app