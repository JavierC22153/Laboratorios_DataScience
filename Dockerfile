FROM jupyter/pyspark-notebook:latest
ENV TZ=America/Guatemala
ENV DEBIAN_FRONTEND=noninteractive

COPY notebooks/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

USER root

RUN mkdir -p /home/gus/work/data /home/gus/work/out \
    && chmod -R 777 /home/gus

WORKDIR /home/gus/work

EXPOSE 8888

CMD ["start-notebook.py", "--NotebookApp.token=''", "--NotebookApp.password=''", "--NotebookApp.allow_origin='*'", "--NotebookApp.ip=0.0.0.0", "--notebook-dir=/home/gus/work"]

