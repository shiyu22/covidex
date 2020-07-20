# COVID-19 Open Research Dataset Search

This repository contains the API server, neural models, and UI client, a neural search engine for the [COVID-19 Open Research Dataset (CORD-19)](https://pages.semanticscholar.org/coronavirus-research) and is refer to [covidex](https://github.com/castorini/covidex).


## Local Deployment

### Requirements

- Install [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-update2)

+ Install [Anaconda](https://docs.anaconda.com/anaconda/install/linux/)

  ```bash
  $ wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
  $ bash Anaconda3-2020.02-Linux-x86_64.sh
  ```

- Install Java 11

    ```bash
    $ sudo apt-get install openjdk-11-jre openjdk-11-jdk
    ```

### Run Server

#### 1. Start Docker container

The server system uses Milvus 0.10.0. Refer to the [Install Milvus](https://github.com/milvus-io/docs/blob/v0.10.0/site/en/guides/get_started/install_milvus/install_milvus.md) for how to start Milvus server.

```bash
$ docker run -d --name milvus_cpu_0.10.0 \
-p 19530:19530 \
-p 19121:19121 \
-v /home/$USER/milvus/db:/var/lib/milvus/db \
-v /home/$USER/milvus/conf:/var/lib/milvus/conf \
-v /home/$USER/milvus/logs:/var/lib/milvus/logs \
-v /home/$USER/milvus/wal:/var/lib/milvus/wal \
milvusdb/milvus:0.10.0-cpu-d061620-5f3c00
```

#### 2. Prepare Anaconda environment

```bash
# Create an Anaconda environment for Python 3.7
$ conda create -n covidex python=3.7
# Activate the Anaconda environment
$ conda activate covidex
# Install Python dependencies
$ pip install -r api/requirements.txt
```

#### 3. Build the [latest Anserini indices](https://github.com/castorini/anserini/blob/master/docs/experiments-cord19.md) and Milvus index

```bash
$ sh scripts/update-index.sh
```

#### 4. Run the server

```bash
# make sure you are in the api folder
$ uvicorn app.main:app --reload --port=8000 --host=127.0.0.1
```

The server wil be running at [127.0.0.1:8000](http://127.0.0.1:8000) with API documentation at [/docs](http://localhost:8000/docs)


## UI Client

- Install  [Node.js 12+](https://nodejs.org/en/download/) and [Yarn](https://classic.yarnpkg.com/en/docs/install/).

- Install dependencies

    ```bash
    $ yarn install
    ```

- Start the server

    ```bash
    $ yarn start
    ```

The client will be running at [localhost:3000](http://localhost:3000)

