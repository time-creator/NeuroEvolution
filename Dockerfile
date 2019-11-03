FROM pytorch/pytorch

COPY requirements.txt /workspace/
RUN conda --y pip
RUN pip install -r /workspace/requirements.txt

# test package installs
RUN python -c "import Keras, pytorch, torchvision, numpy, PIL"
