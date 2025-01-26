
#!/bin/bash

# Find docker or use podman
DOCKER=$(command -v docker || command -v podman)
if [ -z "$DOCKER" ]; then
    echo "No docker or podman found"
    exit 1
fi
alias docker="$DOCKER"

docker build -t cse447-container .

# use your brain when linking the directories
# The following should work for training & val ON HYAK, not for testing
# On Leon's machine, add -v /mnt/e/data/gutenberg:/mnt/e/data/gutenberg to fix symlink hell
docker run                                          \
    -v $PWD/src:/job/src                            \
    -v $PWD/work:/job/work                          \
    -v ./data:/job/data/data                        \
    -v ./data-train:/job/data/data/data-train       \
    -v ./data-val:/job/data/data/data-val           \
    -v $PWD/output:/job/output                      \
    cse447-container                                \
    $@
