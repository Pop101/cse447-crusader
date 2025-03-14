#!/bin/bash
# cd ./data
# wget -H -w 2 -m http://www.gutenberg.org/robot/harvest?filetypes[]=txt \
# --referer="http://www.google.com" \
# --user-agent="Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.6) Gecko/20070725 Firefox/2.0.0.6" \
# --header="Accept: text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5" \
# --header="Accept-Language: en-us,en;q=0.5" \
# --header="Accept-Encoding: gzip,deflate" \
# --header="Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7" \
# --header="Keep-Alive: 300"
# rm -rf "www.gutenberg.org/robot"

# https://download.kiwix.org/zim/gutenberg/ <-- html though

mkdir -p ./data

# Rsync the data into the data folder
rsync -avm --exclude '**/old/' --include '*/' --include '*.txt' --exclude '*' --del ftp.ibiblio.org::gutenberg ./data

# Make index file of all txts
# Note: the -readme files are legal information for audiobooks. Ignore.
find -L ./data -type f -name "*.txt" \
  ! -name "index.txt" \
  ! -name "donate-howto.txt" \
  ! -name "*-readme.txt" \
  -printf "./%P\n" > ./data/index.txt