FROM python:3.9.5

RUN apt-get update -y && apt install libgl1-mesa-glx -y

ENV DATA_DIR /images

WORKDIR /albumentations
COPY . .

RUN pip install -U --no-cache-dir pip
RUN pip install -U --no-cache-dir -e .
RUN pip install -U --no-cache-dir -r ./benchmark/requirements.txt

WORKDIR /albumentations/benchmark

ENTRYPOINT ["python", "benchmark.py"]
