chmod +x gdown.pl
scripts/gdown.pl https://drive.google.com/file/d/1elnSscPCRBxl5PESs5YvBbdyWobi1j8m /tmp/datasets.zip
unzip /tmp/datasets.zip -d /tmp
mv /tmp/datasets/ datasets