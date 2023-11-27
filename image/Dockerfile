FROM public.ecr.aws/lambda/python:3.10

# # Copy requirements.txt
COPY requirements.txt $(LAMBDA_TASK_ROOT)

# Install requirements
RUN pip install numpy transformers
RUN pip install torch==1.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copy all files in /.src 
COPY src/* $(LAMBDA_TASK_ROOT)

# Set the cmd to your handler # What the lambda function runs first.
CMD [ 'lambda_function.lambda_handler' ]
