# JugglePro

A starter Python project for analyzing juggling with webcam computer vision.

## What this does
- Detects moving juggling balls without requiring a specific ball color
- Draws boxes around balls and hands
- Estimates throw height from ball trajectory
- Computes collision risk between tracked balls
- Detects wrist positions and gives a simple arm/hand pattern summary
- Includes a local browser demo using Streamlit

## Requirements
- Python 3.9+
- Webcam attached for live mode
- Recorded video files can also be analyzed with the same script

## Setup
1. Create and activate a virtual environment:

```bash
python -m venv venv
.\\venv\\Scripts\\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the desktop analyzer

```bash
python juggle_analyzer.py
```

Press `q` to quit.

## Analyze a recorded juggling video

```bash
python juggle_analyzer.py --video path/to/video.mp4
```

Press `q` to quit while the video plays.

## Run the web demo

```bash
streamlit run streamlit_app.py
```

Then open the local browser page shown by Streamlit.

## Notes
- The current implementation uses simple color-based ball tracking. For best results, use brightly colored balls and a stable background.
- A full production-level web app is possible, but this repo includes a first pass with Streamlit to demonstrate browser-based capture.
- If you want more accurate throw heights, add a physical calibration marker in the frame and convert pixels to real-world units.
