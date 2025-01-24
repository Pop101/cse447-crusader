
#!/bin/bash

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
