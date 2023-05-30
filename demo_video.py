import streamlit as st
import base64
from streamlit.components.v1 import html




def demo_video_function():
    if st.button('See User Demo'):
        st.warning("""Because of file size please give the video a few seconds to load!""")


        with open("./miscellaneous/compressed_new.mp4", "rb") as f:
            video_bytes = f.read()
            video_str = "data:video/mp4;base64,%s" % base64.b64encode(video_bytes).decode()
            video_html = f"""
                <video class="myvideo" id="video" width="700" controls autoplay>
                    <source src="{video_str}" type="video/mp4">
                    Your browser does not support HTML5 video.
                </video>
            """
        with open ("./miscellaneous/video_fade.css", "r") as f:
            css = f.read()

        # Render the videos and CSS
        my_html=f"""<style>{css}</style>"""

        st.markdown(css, unsafe_allow_html=True)
        video_container = st.empty()
        video_container.write("<div id='video-container'><div class='myvideo'>" + video_html + "</div><div class='overlay'></div></div>", unsafe_allow_html=True)
        html(my_html, height=0)

