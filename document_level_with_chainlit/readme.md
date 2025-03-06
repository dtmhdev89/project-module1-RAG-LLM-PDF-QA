#  Cautions:

-- This demo is just a baseline.

-- The output may not satisfy your need

-- Test it and improve it yourself

# Steps

### Python version

`python --version`

require python version 3.10

### Virtualenv

`python3.10 -m pip install virtualenv`

### Create virtual env

Under the app directory: **document_level_with_chainlit**

`python3.10 -m virtualenv .venv`

### Activate .venv virtual environment

`source .venv/bin/activate`

You can deactivate the activated environment by command:

`deactivate`

### Install requirements libs

`python3.10 -m pip install -r requirements.txt`

### Add your documents

Copy your document to **dataset/docs**

**supported file types: txt, doc, docx, pdf**

### Start chainlit localserver

-- Open terminal window at current directory

-- Run below command for debug mode
`chainlit run app.py --host 0.0.0.0 --port 8000 --debug`

To stop server, press **ctrl + c**

-- Run below command for log mode

`chainlit run app.py --host 0.0.0.0 --port 8000 &>/logs/chainlit_log.txt &`

To stop server:

`ps aux | grep chainlit run app.py`

`kill <PID>`

**<PID>** is the second column's value from above `ps aux` command

### Access server

-- Open a browser
-- type url: `http://0.0.0.0:8000/`
