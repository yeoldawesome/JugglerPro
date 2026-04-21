import cv2
import numpy as np
import streamlit as st

from juggle_analyzer import BallTracker, HandAnalyzer, TrajectoryAnalyzer, create_default_trackers


def analyze_image(frame: np.ndarray):
    trackers = create_default_trackers()
    trajectory = TrajectoryAnalyzer(max_history=1)
    hand_analyzer = HandAnalyzer()

    observations = []
    for tracker in trackers:
        obs = tracker.detect(frame)
        if obs:
            observations.append(obs)
            tracker.draw(frame, obs)

    analysis = trajectory.update(observations)
    hand_data = hand_analyzer.analyze(frame)

    for obs in observations:
        cv2.circle(frame, obs.center, obs.radius, obs.color, 2)
    return frame, analysis, hand_data


def main():
    st.title('JugglePro: Browser Camera Demo')
    st.write('Take a photo or use your webcam to analyze ball positions and hand motion.')

    uploaded = st.camera_input('Use your webcam')
    if uploaded is None:
        st.info('Use the camera button above to capture a frame.')
        return

    bytes_data = uploaded.getvalue()
    image = np.frombuffer(bytes_data, dtype=np.uint8)
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    processed, analysis, hand_data = analyze_image(frame)
    st.image(processed, caption='Analyzed frame', use_column_width=True)

    st.markdown('### Analysis')
    if analysis:
        for label, values in analysis.items():
            if label == 'collision_risk':
                st.write(f'Collision risk: {values:.2f}')
            else:
                st.write(f'{label}: height={values["throw_height_px"]:.0f}px, speed={values["speed_px"]:.1f}')
    st.write(f'Hands detected: {hand_data.get("hands_detected", 0)}')
    st.write(f'Arm pattern summary: {hand_data.get("pattern", "Unknown")}')


if __name__ == '__main__':
    main()
