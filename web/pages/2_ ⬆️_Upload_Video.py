import os
import streamlit as st
import cv2
import tempfile
import sys
from videoreader import VideoReader
# from utils import process_video

from utils import get_mediapipe_pose, get_joint_csv_data
from process_frame import ProcessFrame

angle_data, form_data = get_joint_csv_data() 

pose=get_mediapipe_pose()

download = None

if 'download' not in st.session_state:
    st.session_state['download'] = False

st.title('AI Fitness Trainer: Exercise Analysis')

output_video_file = f'output_recorded.mp4'

stframe = st.empty()

if os.path.exists(output_video_file):
    os.remove(output_video_file)

# Upload video through Streamlit
with st.form('Upload', clear_on_submit=True):
    uploaded_file = st.file_uploader("Upload a video of type mp4, mov, avi", type=['mp4','mov', 'avi'])
    uploaded = st.form_submit_button("Upload")

stframe = st.empty()

ip_vid_str = '<p style="font-family:Helvetica; font-weight: bold; font-size: 16px;">Input Video</p>'

download_button = st.empty()

if uploaded_file and uploaded:
    download_button.empty()
    tfile = tempfile.NamedTemporaryFile(delete=False)
    try:
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') #codec
        out = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)
        
        vr = VideoReader(tfile.name)
        print(tfile.name)
        ip_video = st.sidebar.video(tfile.name) 

        poseProcess = ProcessFrame(data=angle_data, form_data=form_data)

        print(vr)
        
        for frame in vr[:]:
        
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, _ = poseProcess.process(frame, pose)
            
            stframe.image(processed_frame)
            
            out.write(processed_frame)

        # process_video(vr, out)

        out.release()
        stframe.empty()
        ip_video.empty()
        tfile.close()

        txt = st.sidebar.markdown(ip_vid_str, unsafe_allow_html=True)   
        ip_video = st.sidebar.video(output_video_file) 

    except Exception as e:
        # If the file is not an image, show an error message
        st.error("Something went wrong. Please try uploading another video file.")
        print(e)
else:
    # If no file is uploaded, show a warning message
    st.warning("Please upload a video file")

if os.path.exists(output_video_file):
    with open(output_video_file, 'rb') as op_vid:
        download = download_button.download_button('Download Video', data = op_vid, file_name='output_recorded.mp4')

if download:
    st.session_state['download'] = True



if os.path.exists(output_video_file) and st.session_state['download']:
    os.remove(output_video_file)
    st.session_state['download'] = False
    download_button.empty()


