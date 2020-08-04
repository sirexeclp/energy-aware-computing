FROM nvcr.io/nvidia/tensorflow:20.07-tf2-py3

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN  rm requirements.txt
CMD ["bash", "/code/run.sh"]