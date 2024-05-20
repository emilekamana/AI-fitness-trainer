# FitKam: AI fitness trainer
#### Web applcation that counts repetitions, analyses your form and provides feedback while you're exercising
#### By **Kamana Izere Emile**
## Description
The app uses computer vision to analyse every frame of the video. It passes the frame to a classification model to predict the exercise you're doing. And using the mediapipe hollistic mode, gets the joint positions necessary to check them against set thresholds to ensure you're in the valid range of motion
## Installation Process
* Clone the repository and open in terminal
* Install required dependencies
```console
pip install -r requirements.txt
```
* Run the web application on streamlit
```console
streamlit run web/app.py
```
* Test the app on your browser
## Technologies Used
* OpenCV
* MediaPipe
* Streamlit
* Pandas
* InceptionV3
* Tensorflow
## Support and contact details
If you have any questions reach out to me on [e.kamana@alustudent.com]
### License
Licensed by MIT
Copyright (c) 2024 **Kamana Izere Emile**
