sphinx-apidoc -f -e -d 5 -o docsrc/source src/hybrid_jp
cd docsrc
make clean
make html
cd ../docs
cp -r ../docsrc/build/html/ .
cd ..
python -m http.server 8000 --directory docs
