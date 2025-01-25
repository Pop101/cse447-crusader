
#!/bin/bash

# Find docker or use podman
DOCKER=$(which docker || which podman)
if [ -z "$DOCKER" ]; then
    echo "No docker or podman found"
    exit 1
fi
alias docker=$DOCKER

docker build -t cse447-container .

# use your brain when linking the directories
# The following should work for training & val, not for testing
docker run                      \
    -v $PWD/src:/job/src        \
    -v $PWD/work:/job/work      \
    -v ./data-test:/job/data    \
    -v $PWD/output:/job/output  \
    cse447-container            \
    $@
