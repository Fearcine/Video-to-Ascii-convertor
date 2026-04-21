
# Video to ASCII Converter

Converts MP4 video files into ASCII art with live preview, color support, and multiple export formats including ASCII MP4 output.

---

## Installation

### Executable

Download the latest release and run VideoToASCII.exe. No Python required.

### From source

    git clone https://github.com/yourname/video-to-ascii.git
    cd video-to-ascii
    pip install -r requirements.txt
    python main.py

### Build

    build.bat

---

## Usage

Upload a video, adjust character width and height, pick a character set and color mode, then press Play. Settings can be changed while the preview is running. Use the frame scrubber to seek. Export when ready.

### Export formats

- ASCII Video (.txt): every frame as plain text
- Current Frame (.txt): single frame export
- HTML: full-color file with inline RGB, opens in any browser
- ASCII MP4: re-encodes video with ASCII frames at original frame rate, no audio

---

## RAM

Above 400 characters wide, RAM usage increases significantly. 16 GB recommended above 400 characters. 24 GB or more recommended above 700 characters. Reduce resolution if the app slows or export fails.

---

## Requirements

- Windows 10 or 11, 64-bit
- FFmpeg on system PATH
- 8 GB RAM for resolutions under 400 characters wide
- Python 3.11 or higher if running from source

---

