# MNIST Digit Prediction - Full Stack Project

A complete end-to-end machine learning application that recognizes handwritten digits (0-9) using a deep learning CNN model. The project includes a trained PyTorch model, FastAPI backend, and React frontend.

## 🎯 Features

- **Upload handwritten digit images** in any size (automatically resized to 28×28)
- **Real-time predictions** with confidence percentage
- **Image preview** before and after prediction
- **Responsive React UI** with smooth animations
- **REST API** built with FastAPI for scalability
- **CNN model** trained on MNIST dataset with 7 epochs
- **Cross-origin support** for frontend-backend communication

## 🏗️ Project Structure

```
MNIST_Project_Alex/
├── app/
│   ├── main.py                 # Training script with validation & test evaluation
│   ├── load_model.py           # FastAPI backend server
│   └── requirements.txt         # Python dependencies
├── model/
│   ├── mnist_model.pth         # Trained PyTorch model weights
│   └── training_loss.png       # Training visualization
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # Main React component
│   │   ├── App.css             # Styling
│   │   └── index.jsx           # Entry point
│   ├── public/
│   │   └── index.html          # HTML template
│   └── package.json            # NPM dependencies
├── notebooks/
│   └── mnist.ipynb             # Jupyter notebook for exploration
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+
- pip (Python package manager)
- npm (Node package manager)

### 1. Backend Setup

```bash
cd app
pip install -r requirements.txt
python load_model.py
```

The API will run on `http://localhost:8000`

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI)

### 2. Frontend Setup

Open a **new terminal** in the `frontend` directory:

```bash
cd frontend
npm install
npm start
```

The app will run on `http://localhost:3000` and automatically open in your browser.

## 📖 How to Use

1. **Start both servers** (Backend on 8000, Frontend on 3000)
2. **Select an image** — Click "📁 Choose Image" to upload a handwritten digit
3. **View preview** — The selected image displays instantly
4. **Get prediction** — Click "Predict" button to analyze the digit
5. **See results** — View predicted digit and confidence percentage
6. **Try another** — Click "🔄 Try Another" to reset and upload a new image

## 🧠 Model Architecture

The project uses a **Convolutional Neural Network (CNN)** with the following layers:

```
Conv2d(1, 32, kernel_size=5) → ReLU → MaxPool2d
Conv2d(32, 64, kernel_size=5) → ReLU → MaxPool2d
Flatten → Linear(64×7×7, 128) → ReLU → Linear(128, 10)
```

**Training Details:**
- Dataset: MNIST (70,000 handwritten digit images)
- Train/Validation Split: 80/20
- Epochs: 7
- Batch Size: 64
- Optimizer: Adam (lr=0.001)

## 🏋️ Training the Model

To retrain the model with your own settings:

```bash
cd app
python main.py
```

This will:
- Load MNIST dataset
- Split into train/validation (80/20)
- Train for 7 epochs
- Display loss metrics
- Save model as `../model/mnist_model.pth`
- Generate `training_loss.png` visualization
- Evaluate on test dataset

## 📊 API Endpoints

### POST `/predict`
Upload an image and get digit prediction.

**Request:**
```bash
curl -F "file=@digit.png" http://localhost:8000/predict
```

**Response:**
```json
{
  "prediction": 7,
  "confidence": "99.85%",
  "probabilities": [0.001, 0.002, ..., 0.999, 0.003]
}
```

### GET `/`
Get API info.

```bash
curl http://localhost:8000/
```

## 🛠️ Installation from Scratch

### Backend Dependencies
Create `app/requirements.txt`:
```
torch==2.0.0
torchvision==0.15.0
fastapi==0.104.0
uvicorn==0.24.0
pillow==10.0.0
python-multipart==0.0.6
```

Install:
```bash
cd app
pip install -r requirements.txt
```

### Frontend Dependencies
```bash
cd frontend
npm install
```

Automatically installs:
- react, react-dom
- axios
- react-scripts

## 🐛 Troubleshooting

### "Error uploading image: Failed to fetch"
- **Check** if backend is running on port 8000
- **Run** `python load_model.py` in the app folder
- **Refresh** browser (Ctrl+Shift+R)

### "Model file not found"
- Ensure `model/mnist_model.pth` exists
- If missing, run `python main.py` to train the model

### Port already in use
- **Backend**: Change port in `load_model.py` → `uvicorn.run(app, port=8001)`
- **Frontend**: Run `npm start -- --port 3001`

### CORS errors
- Backend CORS is pre-configured for `http://localhost:3000`
- For production, update CORS origins in `load_model.py`

## 📈 Performance

- **Training Accuracy**: ~99%+
- **Test Accuracy**: ~98%+
- **Prediction Time**: <100ms per image
- **Model Size**: ~500KB

## 🔄 Workflow

```
1. Select Image (Frontend)
   ↓
2. Send to Backend (HTTP POST)
   ↓
3. Preprocess (Grayscale + Resize to 28×28)
   ↓
4. Forward through CNN
   ↓
5. Get Softmax Probabilities
   ↓
6. Display Prediction & Confidence (Frontend)
```

## 🚢 Deployment

### Docker Deployment (Coming Soon)
- Backend: Containerized FastAPI app
- Frontend: Static build served by nginx
- Memory efficient and portable

### Heroku/AWS Deployment
- Package both frontend and backend
- Update API endpoint in React app
- Deploy with CI/CD pipeline

## 📝 Technologies Used

**Backend:**
- PyTorch — Deep learning framework
- FastAPI — Modern async web framework
- Uvicorn — ASGI server
- Pillow — Image processing

**Frontend:**
- React 18 — UI library
- CSS3 — Styling with gradients & animations
- FileReader API — Client-side image handling

**ML/Data:**
- MNIST Dataset — 70,000 handwritten digits
- TorchVision — Dataset utilities

## 🎓 Learning Resources

- [PyTorch Docs](https://pytorch.org/docs)
- [FastAPI Tutorial](https://fastapi.tiangolo.com)
- [React Documentation](https://react.dev)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created as a full-stack machine learning project demonstrating end-to-end workflow from model training to deployment.

## 🤝 Contributing

Feel free to fork, modify, and improve this project! Some ideas:
- Add more digit recognition models (ResNet, VGG)
- Implement digit drawing canvas in frontend
- Add batch prediction
- Create mobile app version
- Deploy to cloud

---

**Questions?** Check the troubleshooting section or open an issue on GitHub.

Happy digit prediction! 🎉
