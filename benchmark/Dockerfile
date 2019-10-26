FROM python:3.7.5

ENV DATA_DIR /images

WORKDIR /albumentations
COPY . .

RUN pip install -U --no-cache-dir pip
RUN pip install -U --no-cache-dir -e .
RUN pip install -U --no-cache-dir -r ./benchmark/requirements.txt

WORKDIR /albumentations/benchmark

ENTRYPOINT ["python", "benchmark.py"]
