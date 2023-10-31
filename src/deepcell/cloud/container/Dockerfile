# using the image from https://github.com/aws/deep-learning-containers/blob/master/available_images.md
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38-cu113-ubuntu20.04-e3

RUN pip install git+https://github.com/AllenInstitute/DeepCell.git@train_lmf_model

# fixing issue with multiple opencv versions installed
RUN pip uninstall opencv-python -y
RUN pip uninstall opencv-python-headless -y
RUN pip install opencv-python-headless

ENV PATH="/opt/ml/code:${PATH}"

RUN mkdir -p /opt/ml/code

COPY /train.py /opt/ml/code
COPY /train_input.json /opt/ml/
WORKDIR /opt/ml/code

ENTRYPOINT ["python", "train.py"]
