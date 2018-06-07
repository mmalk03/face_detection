#!/bin/sh
find . -name *.png | xargs mogrify -format jpg
find . -name *.png | xargs rm
sed -i 's/png/jpg/g' faces.csv
