# Instructions

## Install (macOS)

1. Install gcc

    ```brew install gcc```

2. Install OpenMPI

    ```brew install open-mpi```

3. Clone epoch code to a different folder

    ```git clone git clone --recursive https://github.com/Warwick-Plasma/epoch.git```

4. navigate to epoch/SDF/utilities and activate environment (if using), e.g. for pyenv

    ```pyenv shell <env-name>```

5. Install sdf and sdf-helper

    ```sudo ./build -r -s```

### (optional) install ffmpeg for animations

1. Install with brew

    ```brew install ffmpeg```

### (optional) Dev dependencies

1. Sphinx Documentation

    ```shell
    pip install sphinx sphinx-book-theme
    ```

2. Run the `make_docs.sh` command

    `./make_docs.sh`
