FROM tensorflow/tensorflow:2.3.0-gpu

RUN apt-get update && \
      apt-get -y install sudo git

RUN apt-get -y install freeglut3 libglu1 libgl1

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN  rm requirements.txt

ARG uid
ARG group_id
RUN echo "using uid: $uid"

RUN addgroup --gid $group_id usergroup &&\
        adduser user --disabled-password --uid $uid --gid $group_id &&\
        adduser user sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER user

CMD ["bash", "/code/run.sh"]
