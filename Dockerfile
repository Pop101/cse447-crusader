FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# Always mount the following directories:
# - src: the directory containing the training script (usually ./src)
# - work: the directory containing the model weights (usually ./work)

# For training, please link in the following directories:
# - /job/data: the directory containing the training data (usually ./data-train)

# For testing, these directories are linked in:
# - /job/data: <path_to_test_data_containing_input.txt>
# - /job/output: <output_to_write_pred.txt>

# And this command is run is run for testing
#    bash /job/src/predict.sh /job/data/input.txt /job/output/pred.txt`

# You should install any dependencies you need here.
COPY requirements.txt /job/
RUN pip install --no-cache-dir -r requirements.txt