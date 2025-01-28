
#!/bin/bash

# Find docker or use podman
DOCKER=$(command -v docker || command -v podman)
if [ -z "$DOCKER" ]; then
    echo "No docker or podman found"
    exit 1
fi


"$DOCKER" build -t cse447-container .

# use your brain when linking the directories
# The following should work for training & val, not for testing
"$DOCKER" run                        \
    -v $PWD/src:/job/src             \
    -v $PWD/work:/job/work           \
    -v ./data-test:/job/data         \
    -v ./example:/job/data/example   \
    -v $PWD/output:/job/output       \
    cse447-container                 \
    $@
