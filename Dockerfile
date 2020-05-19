FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN apt-get update;
RUN apt-get dist-upgrade --yes;
RUN apt-get install --yes git;
RUN apt-get clean;
RUN rm -rf /var/lib/apt/lists/*;
RUN conda install --yes pandas scikit-learn;
RUN python3 -m pip install --upgrade deap imbalanced-learn;

WORKDIR /workspace
