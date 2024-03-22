#!/bin/sh

# How to generate ANACONDA_TOKEN: https://docs.anaconda.com/anaconda-cloud/user-guide/tasks/work-with-accounts#creating-access-tokens
if [ -z "$ANACONDA_TOKEN" ]; then
    echo "ANACONDA_TOKEN is unset. Please set it in your environment before running this script";
    exit 1
fi

conda install -y conda-build conda-verify anaconda-client
conda config --set anaconda_upload no
conda build --quiet --no-test --output-folder conda_build conda.recipe

# Convert to other platforms: OSX, WIN
conda convert --platform win-64 conda_build/linux-64/*.tar.bz2 -o conda_build/
conda convert --platform osx-64 conda_build/linux-64/*.tar.bz2 -o conda_build/

# Upload to Anaconda
# We could use --all but too much platforms to upload
ls conda_build/*/*.tar.bz2 | xargs -I {} anaconda -v -t $ANACONDA_TOKEN upload -u albumentations {}
