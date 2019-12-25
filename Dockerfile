FROM continuumio/miniconda3:4.7.12

RUN echo "source activate base" > ~/.bashrc

RUN apt-get -qq -y update
RUN apt-get -qq -y upgrade
RUN apt-get -qq -y install \
        gcc \
        g++ \
        wget \
        curl \
        git \
        make \
        unzip \
        sudo \
        nano \
        htop

# Use C.UTF-8 locale to avoid issues with ASCII encoding
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Set the working directory to /app
WORKDIR /app

RUN conda install numpy==1.17.4 scipy==1.3.2 cython=0.29.14 -y
RUN pip install transliterate==1.10.2 grpcio==1.26.0 grpcio-tools==1.26.0 langdetect==1.0.7
RUN conda install -c pytorch pytorch==1.3.1 cpuonly==1.0 faiss-cpu==1.6.1 -y

# Download LASER from FB
WORKDIR /app/laservec
RUN git clone https://github.com/facebookresearch/LASER.git

ENV LASER /app/laservec/LASER
WORKDIR $LASER

RUN bash ./install_models.sh
RUN bash ./install_external_tools.sh

WORKDIR /app
COPY laservec/  ./laservec
COPY app.py config.py ./

EXPOSE 8100

CMD ["python", "app.py"]