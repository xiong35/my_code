
# Program:
#   to handle all the requirments in a new server
# History:
# 20/02/08  xiong35

# configure

#!/bin/bash
PATH=/bin:/sbin:/usr/sbin:/usr/bin:/usr/local/sbin:/usr/local/bin:~/bin
export PATH

################################################################

# installing python3.65

yum install -y openssl-devel bzip2-devel expat-devel gdbm-devel readline-devel sqlite-devel

wget https://www.python.org/ftp/python/3.6.5/Python-3.6.5.tgz

tar -xzvf Python-3.6.5.tgz

cd Python-3.6.5

./configure --prefix=/usr/local/python

make

make install

cd /usr/bin

mv python python.bak
mv pip pip.bak

ln -s /usr/local/python/bin/python3.6 /usr/bin/python
ln -s /usr/local/python/bin/pip3.6 /usr/bin/pip

ll yum*
# change the default python into python2
# vim yum*
# vim /usr/libexec/urlgrabber-ext-down

################################################################

# jupyter

pip install jupyter

# PATH="${PATH}:/usr/local/python/bin"

jupyter notebook --generate-config --allow-root

python
from notebook.auth import passwd
# passwd()

vim ~/.jupyter/jupyter_notebook_config.py

# add the following
# c.NotebookApp.ip='*'
# c.NotebookApp.password = u'之前的那串sha1字符串'
# c.NotebookApp.open_browser = False
# c.NotebookApp.port =7777
# c.NotebookApp.allow_remote_access = True

# remember to turn the 7777 port on!!
# jupyter notebook --allow-root

################################################################

# tensorflow

pip install tensorflow
pip install keras
pip install matplotlib


################################################################

# git

yum install git

ssh-keygen -t rsa -C "2242787668@qq.com"

vim ~/.ssh/id_rsa.pub
# add it to the github

git config user.email "2242787668@qq.com"
git config user.name "xiong35"


git clone git@github.com:xiong35/my_code2242787668.git