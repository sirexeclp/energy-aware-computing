FROM nvcr.io/nvidia/tensorflow:20.07-tf2-py3

RUN apt-get update && \
      apt-get -y install sudo

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN  rm requirements.txt

ARG uid
RUN echo "using uid: $uid"

RUN adduser user --disabled-password --uid $uid && adduser user sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER user

CMD ["bash", "/code/run.sh"]