#!/bin/bash
# install virtualenv
pip2 install --user virtualenv
#create virtual env
python2 ~/.local/lib/python2.7/site-packages/virtualenv.py -p /usr/bin/python2.7 venv --no-site-packages
# activate virtual env
source venv/bin/activate
pip2 install --upgrade pip
# install packages with requirements.txt
pip2 install -r requirements_pdc.txt
# exit virtual env
deactivate
