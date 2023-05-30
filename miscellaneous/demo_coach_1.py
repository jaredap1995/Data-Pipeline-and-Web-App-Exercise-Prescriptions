import streamlit as st
import base64
from streamlit.components.v1 import html




def demo_coach_function_1():
    if st.button('View Autoencoder Coach Demo'):
        st.warning("""Because of file size please give the video a few seconds to load!""")


        with open("./miscellaneous/compressed_coach_1.mp4", "rb") as f:
            video_bytes = f.read()
            video_str = "data:video/mp4;base64,%s" % base64.b64encode(video_bytes).decode()
            video_html = f"""
                <video class="myvideo" id="video" width="700" controls autoplay>
                    <source src="{video_str}" type="video/mp4">
                    Your browser does not support HTML5 video.
                </video>
            """

        # Define the CSS for the animation
        css = """
        <style>
            #video-container {
                position: relative;
                width: 640px;
                height: 360px;
            }
            .myvideo {
                opacity: 0;
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                margin: auto;
                animation: fade-in-out 450s linear forwards;
            }

            .myvideo:paused {
                opacity: 0;
            }

            .myvideo.playing {
                animation: none;
            }
            .myvideo.playing + .overlay {
                display: none;
            }
            .myvideo.ended + .overlay {
                animation: fadeout 5s forwards;
            }

            @keyframes fade-in-out {
                0% {
                    opacity: 0;
                }
                1% {opacity: 1;}
                99% {
                    opacity: 1;
                }
                100% {
                    opacity: 0;
                }
                }
        </style>
        """

        # Render the videos and CSS
        my_html=f"""<style>{css}</style>"""

        st.markdown(css, unsafe_allow_html=True)
        video_container = st.empty()
        video_container.write("<div id='video-container'><div class='myvideo'>" + video_html + "</div><div class='overlay'></div></div>", unsafe_allow_html=True)
        html(my_html, height=0)

