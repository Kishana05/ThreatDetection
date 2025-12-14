# 🛡️ Video Anomaly Detection System

A comprehensive threat detection system that uses YOLOv5 deep learning models to detect various suspicious activities and anomalies in real-time video streams, images, and webcam feeds.

## ✨ Features

### Detection Capabilities
- 🔥 **Fire Detection** - Detects fire and smoke in video streams
- ⚔️ **Fighting Detection** - Identifies physical altercations
- 🏪 **Shop Lifting Detection** - Monitors shoplifting activities
- 🚗 **Accident Detection** - Detects vehicle accidents
- 👥 **Crowd Pushing Detection** - Identifies aggressive crowd behavior
- 👥 **Large Crowd Detection** - Monitors overcrowded areas
- 🎨 **Vandalism Detection** - Detects property damage
- 🔫 **Weapon Detection** - Identifies weapons in frames

### Key Features
- 🎨 **Modern Web Interface** - Beautiful, responsive UI with elegant design
- 📹 **Multiple Input Sources** - Support for webcam, video files, and static images
- 🔔 **Real-time Notifications** - Telegram integration for instant alerts
- 📊 **Live Detection Display** - Real-time visualization with bounding boxes
- 🔐 **User Authentication** - Secure login and registration system
- 📱 **Responsive Design** - Works seamlessly on desktop, tablet, and mobile

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kishana05/ThreatDetection.git
   cd ThreatDetection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Telegram Bot** (optional, for notifications)
   - Create a Telegram bot using [@BotFather](https://t.me/botfather)
   - Get your bot token and chat ID
   - Update the bot credentials in the detection scripts

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the web interface**
   - Open your browser and navigate to `http://localhost:5000`
   - Register a new account or login with existing credentials

## 📁 Project Structure

```
ThreatDetection/
├── app.py                 # Flask application entry point
├── templates/             # HTML templates
│   ├── index.html        # Login/Registration page
│   └── userlog.html      # Dashboard page
├── static/               # Static files (images, CSS, JS)
│   └── test*/           # Test image folders
├── models/               # YOLOv5 model definitions
├── utils/                # Utility functions
│   ├── dataloaders.py   # Data loading utilities
│   ├── plots.py         # Visualization utilities
│   └── ...
├── *.py                  # Detection scripts for each threat type
├── *.pt                  # Trained model weights (not in repo)
└── data/                 # Dataset configurations
```

## 🎯 Usage

### Web Interface

1. **Login/Register**
   - Visit `http://localhost:5000`
   - Create an account or login with existing credentials

2. **Select Detection Type**
   - Choose from the available detection cards on the dashboard
   - Each card represents a different threat detection capability

3. **Upload Media**
   - **Webcam**: Click "Detect" to use your webcam
   - **Video File**: Select a video file from the dropdown
   - **Static Image**: Select an image file from the dropdown

4. **View Results**
   - Detection window opens showing real-time results
   - Bounding boxes highlight detected threats
   - Close the window using the 'X' button when done

### Detection Scripts

Each detection script can be run independently:

```bash
# Fire detection
python fire_detection.py

# Fighting detection
python fighting.py

# Vandalism detection
python vandalism.py

# ... and so on
```

## 🔧 Configuration

### Telegram Notifications

To enable Telegram notifications, update the bot credentials in each detection script:

```python
import telepot
bot = telepot.Bot('YOUR_BOT_TOKEN')
bot.sendMessage('YOUR_CHAT_ID', 'Detection alert!')
```

### Model Weights

Model weights (`.pt` files) are not included in the repository due to size. You'll need to:
- Train your own models, or
- Download pre-trained weights and place them in the project root

## 🛠️ Technologies Used

- **Backend**: Flask (Python web framework)
- **Deep Learning**: YOLOv5 (You Only Look Once v5)
- **Computer Vision**: OpenCV
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, JavaScript
- **Notifications**: Telegram Bot API

## 📝 Notes

- Model weights (`.pt` files) are excluded from the repository
- Detection results and temporary files are excluded via `.gitignore`
- The system requires a webcam or video files for detection
- Real-time detection performance depends on your hardware

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Kishana05**
- GitHub: [@Kishana05](https://github.com/Kishana05)

## 🙏 Acknowledgments

- YOLOv5 by Ultralytics
- Flask community
- OpenCV contributors

---

⭐ If you find this project helpful, please consider giving it a star!

