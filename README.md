# 🐟 HealthyFins - Complete Fish Health Management System 

[![Live Demo](https://img.shields.io/badge/demo-LIVE-brightgreen)](https://fish-app.vercel.app)
[![GitHub Repo](https://img.shields.io/badge/repo-HealthyFins-blue)](https://github.com/yourusername/healthyfins)
[![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-black?logo=flask)](https://flask.palletsprojects.com)
[![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-green)](https://arxiv.org/abs/1801.04381)
[![Supabase](https://img.shields.io/badge/Database-Supabase-green?logo=supabase)](https://supabase.com)
[![IoT](https://img.shields.io/badge/IoT-Raspberry%20Pi-red?logo=raspberrypi)](https://raspberrypi.org)
[![Frontend](https://img.shields.io/badge/Frontend-Vercel-black?logo=vercel)](https://vercel.com)
[![Backend](https://img.shields.io/badge/Backend-Render-blue?logo=render)](https://render.com)
[![Docker](https://img.shields.io/badge/Docker-✓-blue?logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE) 

---

## 🎯 Overview

**HealthyFins** is a comprehensive fish health management platform that combines **AI-powered disease detection**, **real-time water quality monitoring**, and **user history tracking** in one integrated system. 

Designed for fish farmers, aquarium hobbyists, and aquaculture researchers, it provides:
- 🔬 **Visual disease diagnosis** from photos (90%+ accuracy)
- 📊 **Water quality monitoring** (pH)
- 📱 **Historical data tracking** for all your tanks
- 🔔 **Alert system** for critical conditions
- 📈 **Trend analysis** and predictive insights

---
<br/>

## 🎥 HealthyFins Demo

https://github.com/user-attachments/assets/594a03e3-466a-4719-aee0-002150d44b39

<br/>

## ✨ Key Features

### 🖼️ **AI Disease Detection**
- Upload fish photos for instant diagnosis
- 7 disease classes: Healthy, White tail or rot, EUS, Fungus, Bacterial, gill disease,parasitic
- Confidence scores for each prediction
- Dual preprocessing methods for better accuracy

### 👤 **User History & Authentication**
- Secure signup/login with JWT authentication
- Personal dashboard with prediction history
- Save and review past diagnoses
- Track multiple tanks/fish populations

### 📊 **pH Monitoring & Water Quality**
- Real-time pH sensor monitoring
- Historical graphs and trends
- Alert thresholds 

### 🔧 **Hardware Integration**
- ESP system integration
- pH sensor (Analog pH Meter V2)
- Automated data collection every 15 minutes

### 🗄️ **Supabase Database**
- User profiles and authentication
- Prediction history storage
- Sensor readings time-series
- Tank/fish population management
- Real-time sync across devices

🧠**MobileNetV2 Architecture**
 The disease detection engine uses MobileNetV2, a lightweight convolutional neural network optimized for edge devices and fast inference.
 **Model Details**
- Base Architecture: MobileNetV2 (Transfer Learning)
- Input Size: 224 × 224 RGB images
- Framework: TensorFlow 2.15
- Output: 7 disease classes
- Accuracy: ~90% on validation set
**Why MobileNetV2?**
- Lightweight and fast for real-time prediction
- Low memory usage (ideal for deployment)
- High accuracy with limited training data
- Suitable for edge devices like Raspberry Pi  

## 💻 Tech Stack
- Frontend: HTML5, CSS3, Vanilla JavaScript hosted on Vercel with Chart.js for sensor data visualization
- Backend: Python 3.9, Flask 2.3.3, TensorFlow 2.15, MobileNetV2, JWT authentication hosted on Render
- Database: Supabase PostgreSQL with Row Level Security for user profiles, predictions, and sensor readings
- Hardware: ESP-8266 4 with pH sensor (Analog pH Meter V2) and temperature sensor (DS18B20)

## 📡 API Endpoints
- POST /predict - Upload fish image for disease diagnosis (returns disease name and confidence score)
- POST /api/auth/register - Create new user account (returns JWT token)
- POST /api/auth/login - Authenticate user (returns JWT token)
- GET /api/predictions - Get user's prediction history with pagination
- DELETE /api/predictions/{id} - Delete specific prediction from history
- POST /api/sensors/reading - Add pH/temperature reading from Raspberry Pi
- GET /api/sensors/history/{tank_id} - Get sensor data for charts
- GET /api/sensors/latest/{tank_id} - Get most recent sensor reading
- POST /api/tanks - Register new fish tank for monitoring
- GET /api/tanks - Get all tanks for authenticated user

🏗️**System Architecture**
 HealthyFins follows a modular full-stack architecture combining AI inference, cloud backend, real-time database, and IoT hardware integration.

- 🔄 High-Level Workflow
- User uploads fish image through frontend.
- Image is sent to Flask backend API.
- Backend preprocesses image and runs MobileNetV2 inference.
- Prediction happens through loaded model.
- Dashboard displays diagnosis history and analytics.
- Hardware collect water data periodically.
- Frontend fetches live and historical data for visualization. 

🚀 Deployment
- Frontend: Deployed on Vercel for fast static hosting and global CDN delivery
- Backend: Hosted on Render using Dockerized Flask API
- Database: Supabase PostgreSQL with Row Level Security enabled
- Model Storage: MobileNetV2 model loaded at runtime from backend server
- Vercel auto-builds frontend
- Render rebuilds backend container
- Backend connects to Supabase via environment variables
- Live system updates automatically  


## 📁 Project Structure
healthyfins/
- 📁 frontend/                             
- 📁 backend/                              
- 📁 model/                                
- 📁 training/                             
- README.md                                



## 👨‍💻 Contact
- GitHub: https://github.com/ajayraj30002
- LinkedIn: https://www.linkedin.com/in/ajay-raj-1ba689246


