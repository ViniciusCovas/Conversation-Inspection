# !/usr/bin/env python
# exit on error
set -o errexit
pip install --upgrade pip
pip install gunicorn
pip install -r requirements.txt