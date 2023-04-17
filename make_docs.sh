sphinx-apidoc -f -o docsrc/source . setup.py
cd docsrc
make clean
make html
cd ../docs
cp -r ../docsrc/build/html/ .