# heimdall-raspberry

- This is the project page for a doorbell camera using the raspberry pi. This script uses `python3` and requires the installation of `picamera`, OpenCV as well as the numpy package.
    - `sudo apt-get install python3-picamera`
    - `python3 -m pip install numpy`
    - [opencv3-python3-Raspberry PI](https://www.pyimagesearch.com/2017/09/04/raspbian-stretch-install-opencv-3-python-on-your-raspberry-pi/))

After installing the requirements, clone this repo and change the variable `broker` in the file `heimdall.py` according to your setup.

After starting the heimdall docker containers, start sending images to the server using the following command: `python3 heimdall.py`. 