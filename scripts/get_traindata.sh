chmod +x gdown.pl
scripts/gdown.pl https://drive.google.com/file/d/1umUoGFHQe1Xy3kH_6MGNmwh-cmCYS4h7 /tmp/traindata.zip
unzip /tmp/traindata.zip -d /tmp
mv /tmp/traindata traindata