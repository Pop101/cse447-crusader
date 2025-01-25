
#!/bin/bash

docker build -t cse447-container .

# use your brain when linking the directories
# The following should work for training & val ON HYAK, not for testing
# On Leon's machine, add -v /mnt/e/data/gutenberg:/mnt/e/data/gutenberg to fix symlink hell
docker run                                          \
    -v $PWD/src:/job/src                            \
    -v $PWD/work:/job/work                          \
    -v /gscratch/stf/lleibm/gutenberg:/job/data/    \
    -v $PWD/output:/job/output                      \
    cse447-container                                \
    $@
