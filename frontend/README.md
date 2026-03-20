# MNIST Frontend (React)

React frontend for MNIST digit prediction.

## Setup

```bash
npm install
```

## Development

```bash
npm start
```

Runs the app at `http://localhost:3000`

## Build

```bash
npm build
```

## Configuration

Update the API endpoint in `src/App.jsx`:
```javascript
const response = await fetch('http://localhost:8000/predict', {
```

Change `http://localhost:8000` to your backend API URL.
