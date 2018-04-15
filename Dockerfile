FROM ubuntu:17.10

RUN apt-get update
RUN apt-get install -y python3 python3-dev python3-pip
ADD requirements.txt /
RUN pip3 install -U pip requests
ADD baduk-0.0.0-cp36-cp36m-linux_x86_64.whl /
RUN /usr/local/bin/pip3 install baduk-0.0.0-cp36-cp36m-linux_x86_64.whl
RUN /usr/local/bin/pip3 install -r /requirements.txt
RUN mkdir /app
ADD badukai /app/
ADD *.py /app/
WORKDIR /app
ENTRYPOINT ["/usr/bin/python3"]
