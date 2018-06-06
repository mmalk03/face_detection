#!/bin/sh
echo 'Moving images to images/'
mkdir images
for rapper_name in kękę ostr otsochodzi taco_hemingway
do
    cd raw_dataset/$rapper_name
    for file in *
    do
        if [ "${file: -4}" = "jpeg" ]
        then
            convert $file ${file/jpeg/jpg}
            rm $file
            cp ${file/jpeg/jpg} ../../images/$rapper_name-${file/jpeg/jpg}
        else
            cp $file ../../images/$rapper_name-$file
        fi
    done
    cd ../../
done
echo 'Creating labels.csv'
sed -r 's/(\w+)\/([^,]*),(.*)/\1-\2,face,\3/' <raw_dataset/faces.csv >labels.csv
#TODO: create python script for preprocessing