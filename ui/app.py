import base64
import hashlib
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from tensorflow.keras.models import load_model

from detection.mask_detector import detect_and_predict_mask, load_face_detector


ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "assets" / "images"


def local_css(file_name):
    with open(ROOT / "ui" / file_name, encoding="utf-8") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)


def image_data_uri(file_name):
    image_path = IMAGE_DIR / file_name
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    suffix = image_path.suffix.lower().lstrip(".")
    mime_type = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return f"data:image/{mime_type};base64,{encoded_image}"


@st.cache_resource
def load_mask_model():
    return load_model(ROOT / "models" / "mask_classifier" / "mask_detector.model")


@st.cache_resource
def load_face_model():
    return load_face_detector()


def annotate_frame(frame, locations, predictions):
    for (start_x, start_y, end_x, end_y), (mask, without_mask) in zip(
        locations, predictions
    ):
        label = "Mask" if mask > without_mask else "No Mask"
        color = (77, 175, 126) if label == "Mask" else (90, 119, 224)
        label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)

        cv2.putText(
            frame,
            label,
            (start_x, max(20, start_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
        )
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)
    return frame


class FaceMaskVideoProcessor(VideoProcessorBase):
    def __init__(self, mask_net):
        self.face_net = load_face_model()
        self.mask_net = mask_net

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        locations, predictions = detect_and_predict_mask(
            image, self.face_net, self.mask_net
        )
        image = annotate_frame(image, locations, predictions)
        return type(frame).from_ndarray(image, format="bgr24")


def render_header(hero_image):
    st.markdown(
        f"""
        <header class="site-header">
          <a class="wordmark" href="#top" aria-label="Veil home">
            <span class="wordmark-mark" aria-hidden="true">◉</span>
            <span>VEIL<span class="wordmark-period">.</span></span>
          </a>
          <nav class="site-nav" aria-label="Main navigation">
            <a href="#approach">The approach</a>
            <a href="#studio">Detection studio</a>
          </nav>
          <a class="header-cta" href="#studio">Open the studio <span aria-hidden="true">↗</span></a>
        </header>
        <main id="top">
          <section class="hero" aria-labelledby="hero-title">
            <img class="hero-photo" src="{hero_image}" alt="People wearing protective face masks in a city crowd">
            <div class="hero-shade"></div>
            <div class="hero-content">
              <p class="eyebrow hero-eyebrow"><span class="live-dot"></span> COMPUTER VISION / 01</p>
              <h1 id="hero-title">A closer look<br>at <em>everyday</em><br>protection.</h1>
              <p class="hero-copy">A study in seeing clearly. Detect face coverings in a still image or live camera feed, with a model running right here on your machine.</p>
              <a class="hero-link" href="#studio"><span class="hero-link-icon">↓</span> Enter the detection studio</a>
            </div>
            <div class="hero-index"><span>01 / 03</span><span class="index-rule"></span><span>VISION IN PRACTICE</span></div>
            <div class="hero-caption">OBSERVATION, MADE VISIBLE <span>— 2026</span></div>
          </section>
          <section class="intro-section" id="approach">
            <div class="intro-kicker"><span class="section-number">01</span><span>THE APPROACH</span></div>
            <div class="intro-copy">
              <h2>Not just seeing a face.<br><em>Reading the detail.</em></h2>
              <p>Veil brings two steps of computer vision into one quiet workspace: first locating faces, then reading whether a mask is present. No upload to a remote service. Just an image, a model, and a clear result.</p>
            </div>
            <div class="intro-note"><span class="note-line"></span><span>LOCAL INFERENCE<br>IMAGE + LIVE VIDEO</span></div>
          </section>
          <section class="visual-section" aria-label="Mask detection in context">
            <img src="{image_data_uri("pic1.jpeg")}" alt="Two people wearing medical masks by the sea">
            <div class="visual-overlay"></div>
            <p class="visual-caption"><span>FIELD STUDY / 01</span><span>PROTECTION, IN EVERYDAY LIFE</span></p>
            <div class="visual-statement">Look closer.<br><em>Understand more.</em></div>
          </section>
          <section class="studio-intro" id="studio">
            <div class="intro-kicker"><span class="section-number">02</span><span>DETECTION STUDIO</span></div>
            <div class="studio-title-row">
              <h2>Bring an image.<br><em>Get a read.</em></h2>
              <p>Choose a still image for a considered scan, or switch to live video for real-time detection. Your media stays in this session.</p>
            </div>
          </section>
        </main>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        """
        <footer class="site-footer">
          <a class="wordmark footer-wordmark" href="#top"><span class="wordmark-mark">◉</span><span>VEIL<span class="wordmark-period">.</span></span></a>
          <span class="footer-caption">A SMALL STUDY IN COMPUTER VISION</span>
          <a class="back-top" href="#top">BACK TO TOP ↑</a>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def render_studio():
    st.markdown('<div class="studio-shell">', unsafe_allow_html=True)
    mode = st.radio(
        "Choose a detection mode",
        ["Image", "Live camera"],
        horizontal=True,
        label_visibility="collapsed",
        key="detection_mode",
    )

    if mode == "Image":
        st.markdown(
            '<p class="studio-helper">01 — STILL IMAGE <span>JPG, JPEG OR PNG</span></p>',
            unsafe_allow_html=True,
        )
        image_file = st.file_uploader(
            "Add an image to inspect",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="image_upload",
        )
        if image_file is None:
            st.markdown(
                """
                <div class="upload-hint">
                  <span class="upload-mark" aria-hidden="true">＋</span>
                  <span class="upload-title">Choose an image to begin</span>
                  <span class="upload-caption">A clear, well-lit image gives the model more to work with.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            image = Image.open(image_file).convert("RGB")
            upload_signature = hashlib.sha256(image_file.getvalue()).hexdigest()
            preview_col, action_col = st.columns([1.25, 0.75], gap="large")
            with preview_col:
                st.image(image, caption="SOURCE IMAGE", use_column_width=True)
            with action_col:
                st.markdown(
                    '<div class="scan-copy"><span class="scan-step">READY TO ANALYSE</span><h3>One image.<br>Two model passes.</h3><p>We locate each face, then estimate whether a mask is present.</p></div>',
                    unsafe_allow_html=True,
                )
                if st.button("Analyse image", type="primary", use_container_width=True):
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    locations, predictions = detect_and_predict_mask(
                        frame, load_face_model(), load_mask_model()
                    )
                    result = annotate_frame(frame, locations, predictions)
                    st.session_state["detection_result"] = cv2.cvtColor(
                        result, cv2.COLOR_BGR2RGB
                    )
                    st.session_state["detection_count"] = len(locations)
                    st.session_state["detection_signature"] = upload_signature
            if st.session_state.get("detection_signature") == upload_signature:
                st.markdown(
                    f'<p class="result-label">ANALYSIS COMPLETE <span>{st.session_state.get("detection_count", 0)} FACES FOUND</span></p>',
                    unsafe_allow_html=True,
                )
                st.image(
                    st.session_state["detection_result"],
                    caption="DETECTION RESULT",
                    use_column_width=True,
                )
    else:
        st.markdown(
            '<p class="studio-helper">02 — LIVE CAMERA <span>ALLOW CAMERA ACCESS TO BEGIN</span></p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="camera-note">The live feed is processed in your browser session. Start the camera below, then use its stop control to end the scan.</p>',
            unsafe_allow_html=True,
        )
        mask_net = load_mask_model()
        webrtc_streamer(
            key="face-mask-detection",
            video_processor_factory=lambda: FaceMaskVideoProcessor(mask_net),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def run_app():
    st.set_page_config(
        page_title="Veil — Face mask detection",
        page_icon="◉",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    local_css("styles.css")
    render_header(image_data_uri("pic2.jpg"))
    render_studio()
    render_footer()
