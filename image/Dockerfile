FROM public.ecr.aws/lambda/python:3.10

# # Copy requirements.txt
COPY requirements.txt .

ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

ENV AWS_DEFAULT_REGION=us-east-1
ENV AWS_DEFAULT_OUTPUT=json

# COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install requirements
RUN pip install numpy transformers pandas tqdm joblib boto3 faiss-cpu awscli
RUN pip install torch==1.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html 

# Copy all files in /.src 
COPY src/* .
# COPY src/* ${LAMBDA_TASK_ROOT}

# Set the cmd to your handler # What the lambda function runs first.
CMD ["main.handler"]
