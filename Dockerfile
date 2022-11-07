FROM tensorflow/tensorflow:1.15.5-gpu-py3
ENV TF_FORCE_GPU_ALLOW_GROWTH true
RUN pip install certifi==2020.4.5 rouge requests