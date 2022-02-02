FROM python:3.9-slim-bullseye

# Mount current directory to /app in the container image
RUN mkdir /app

# Copy local directory to /app in container
# Dont use COPY * /app/ , * will lead to lose of folder structure in /app
COPY flaskr /app/flaskr
COPY model /app/model
COPY requirements.txt /app/
#COPY flaskr/flask_app.py /app/
# Change WORKDIR
WORKDIR /app

# Install dependencies
# use --proxy http://<proxy host>:port if you have proxy
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install torch==1.9.1+cu111 
#torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR flaskr
# In Docker, the containers themselves can have applications running on ports. To access these applications, we need to expose the containers internal port and bind the exposed port to a specified port on the host.
# Expose port and run the application when the container is started
EXPOSE 5000:5000
#CMD  ["python", "flask_dummy.py"]
CMD ["python", "flask_app.py","5000"]
# CMD ["flask_api.py"]