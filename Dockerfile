FROM codingforentrepreneurs/python:3.9-webapp-cassandra

copy ./pipelines /app/pipelines
copy .env /app/.env
COPY ./app /app/app
COPY ./requirements.txt /app/requirements.txt
COPY ./entrypoint.sh /app/entrypoint.sh

WORKDIR /app


RUN chmod +x entrypoint.sh
RUN python3 -m venv /opt/venv && /opt/venv/bin/python -m pip install -r requirements.txt

#RUN /opt/venv/bin/python -m pypyr /app/pipelines

CMD ["./entrypoint.sh"]