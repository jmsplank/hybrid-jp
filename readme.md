# Instructions

## Constants
Constants are stored in `src/hybrid_jp/constants.py`
<!-- 
python -c 'with open("src/hybrid_jp/constants.py", "r") as file:
    lines=file.readlines()
print("\n".join([l.split(" = ")[0] for l in lines]));'
-->
```
VAR_GRID
VAR_GRID_MID
VAR_NUMBERDENSITY
VAR_BX
VAR_BY
VAR_BZ
```


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