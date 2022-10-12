# ball_detection_rs

## Install

Install Python 3.6 for Windows: https://www.python.org/ftp/python/3.6.8/python-3.6.8-amd64.exe

### On Windows:
Create a virtualenv to hold all the Python dependency packages.

    python -m venv .venv

Install the Python dependencies using the following commands:

    .venv\Scripts\activate
    
	pip install -r requirements.txt

# Running

    python main.py [-debug True]

The --debug flag is optional to see the detections on OpenCV windows.