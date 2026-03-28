FROM python:3.12

WORKDIR /app

COPY socket_transfer.py .
COPY memfd_transfer.py .
COPY header.py .
