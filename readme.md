# Instructions

## Install (macOS)

1. Install gcc

   `brew install gcc`

2. Install OpenMPI

   `brew install open-mpi`

3. Clone epoch code to a different folder

   `git clone git clone --recursive https://github.com/Warwick-Plasma/epoch.git`

4. navigate to epoch/SDF/utilities and activate environment (if using), e.g. for pyenv

   `pyenv shell <env-name>`

5. Install sdf and sdf-helper

   `sudo ./build -r -s`

### (optional) install ffmpeg for animations

1. Install with brew

   `brew install ffmpeg`

### (optional) Dev dependencies

1. Sphinx Documentation

   ```shell
   pip install sphinx sphinx-book-theme
   ```

2. Run the `make_docs.sh` command

   `./make_docs.sh`

## Documentation

Documentation is available at [https://jmsplank.github.io/hybrid-jp/](https://jmsplank.github.io/hybrid-jp/) for the most recently pushed code.

### Updating the docs

The script `./make_docs.sh` will search through `src/hybrid_jp` and automatically build documentation for it, as well as automatically start a python `http.server` hosting the local docs on [http://localhost:8000](http://localhost:8000).

The built documentation is stored in `docs/`, this is the directory served to github pages on push. The docs ARE NOT automatically updated on push, so you will need to run `./make_docs.sh` before pushing to github if the changes should be documented.
