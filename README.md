# 🐟 HealthyFins - Complete Fish Health Management System 

[![Live Demo](https://img.shields.io/badge/demo-LIVE-brightgreen)](https://fish-app.vercel.app)
[![GitHub Repo](https://img.shields.io/badge/repo-HealthyFins-blue)](https://github.com/ajayraj30002/healthyfins) 
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)](https://tensorflow.org)
[![Kafka](https://img.shields.io/badge/Apache_Kafka-Streaming-231F20?logo=apachekafka)](https://kafka.apache.org)
[![MQTT](https://img.shields.io/badge/MQTT-HiveMQ-yellow)](https://hivemq.com)
[![Supabase](https://img.shields.io/badge/Database-Supabase-green?logo=supabase)](https://supabase.com)
[![IoT](https://img.shields.io/badge/IoT-ESP8266-red?logo=espressif)](https://espressif.com)
[![Frontend](https://img.shields.io/badge/Frontend-Vercel-black?logo=vercel)](https://vercel.com)
[![Backend](https://img.shields.io/badge/Backend-Render-blue?logo=render)](https://render.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE) 

---

## 🎯 Overview

**HealthyFins** is an advanced, event-driven fish health management platform that combines **AI-powered disease detection**, **real-time decoupled IoT water quality monitoring**, and **user history tracking** in one enterprise-grade system. 

Designed for fish farmers, aquarium hobbyists, and aquaculture researchers, it provides:
- 🔬 **Visual disease diagnosis** from photos (90%+ accuracy)
- 🌊 **Real-time IoT streaming** of water quality (pH) via MQTT & Apache Kafka
- 📱 **Historical data tracking** for tanks mapped to hardware IDs
- 🔔 **Alert system** for critical conditions and urgent symptoms
- 📈 **Trend analysis**, search, and data export

---
<br/>

## 🎥 HealthyFins Demo

https://github.com/user-attachments/assets/594a03e3-466a-4719-aee0-002150d44b39

<br/>

## ✨ Key Features

### 🖼️ **AI Disease Detection (TensorFlow)**
- Upload fish photos for instant diagnosis using a custom-trained model.
- 7 disease classes: Healthy, White tail/rot, EUS, Fungus, Bacterial, Gill disease, Parasitic.
- Symptom extraction and confidence scores for each prediction.
- Resilient fallback to an enhanced image-feature analysis mode if model confidence is low.

### 🌊 **Event-Driven IoT Pipeline (Kafka & MQTT)**
- **Hardware:** ESP8266 continuously publishing analog pH readings.
- **Messaging:** Lightweight MQTT protocol handled by HiveMQ Cloud.
- **Streaming:** Apache Kafka (Aiven) handling high-throughput event streaming.
- **Decoupled:** Web API performance is completely isolated from hardware data streams.

### 👤 **User History & Authentication**
- Secure signup/login with JWT authentication.
- Personal dashboard with prediction history, statistics, and advanced search.
- Export prediction data to CSV or JSON formats.
- Profile management with hardware ID binding.

### 🗄️ **Supabase Database**
- User profiles and authentication bridging.
- Image prediction history storage.
- Real-time sync across devices.

---

## 🧠 MobileNetV2 Architecture
The disease detection engine uses MobileNetV2, a lightweight convolutional neural network optimized for fast inference and deployment in cloud containers.

**Model Details:**
- **Base Architecture:** MobileNetV2 (Transfer Learning)
- **Input Size:** 224 × 224 RGB images
- **Framework:** TensorFlow 2.15
- **Output:** 7 disease classes
- **Accuracy:** ~90% on the validation set
- **Custom Loader:** Implements custom `InputLayer` handling for robust deployment state restoration.

---

## 💻 Tech Stack
- **Frontend:** HTML5, CSS3, Vanilla JavaScript hosted on Vercel with Chart.js for data visualization.
- **Backend:** Python 3.9, **FastAPI** (Async), TensorFlow 2.15, `uvicorn`.
- **IoT / Streaming Pipeline:** 
  - **ESP8266** Microcontroller (C++)
  - **HiveMQ Cloud** (MQTT Broker)
  - **Aiven Cloud** (Apache Kafka Cluster)
  - `paho-mqtt` and `aiokafka` for Python integration.
- **Database:** Supabase PostgreSQL (via REST API).
- **Deployment:** Render (Backend monolithic micro-services), Vercel (Frontend).

---

## 🏗️ System Architecture

HealthyFins follows a decoupled, event-driven microservices architecture combining AI inference, an asynchronous backend, and a real-time IoT hardware stream.

**1. The IoT Stream (Real-Time)**
1. **ESP8266** reads analog sensor data and publishes a JSON payload to **HiveMQ** via MQTT.
2. An asynchronous background thread (`iot_bridge.py`) continuously pulls from HiveMQ and pushes the payload directly into an **Aiven Apache Kafka** cluster.
3. FastAPI runs an `AIOKafkaConsumer` task in the background, updating the server's in-memory state instantly without blocking web traffic.

**2. The Web Application (REST API)**
1. User uploads a fish image through the Vercel frontend.
2. The image hits the FastAPI `/predict` endpoint on Render.
3. The backend preprocesses the image (matching Colab training logic) and runs MobileNetV2 inference.
4. Results (predictions, confidence, symptoms) are saved to **Supabase** and returned to the client.

---

## 📡 API Endpoints

**Authentication & Profiles**
- `POST /register` - Create new user account (returns JWT token)
- `POST /login` - Authenticate user (returns JWT token)
- `GET /profile` - Retrieve user details and hardware ID
- `PUT /profile` - Update user details/hardware binding

**AI Inference**
- `POST /predict` - Upload fish image for disease diagnosis (Returns top 3 predictions + symptoms)

**Sensor Telemetry (HTTP Read)**
- `GET /ph-monitoring/latest` - Fetch live Kafka stream state for the user's bound hardware ID

**History & Analytics**
- `GET /history` - Get user's prediction history with pagination
- `GET /search` - Search history by disease name or symptoms
- `GET /stats` - Retrieve aggregate user statistics
- `DELETE /history/{id}` - Delete specific prediction
- `GET /export/history` - Export user data as JSON or CSV

---

🚀 Deployment
- **Frontend:** Deployed on Vercel for fast static hosting and global CDN delivery.
- **Backend:** Hosted on Render using Python Native Environment.
- **Database:** Supabase PostgreSQL with Row Level Security enabled.
- **Environment:** Managed via Render Environment Variables for secure secret injection (Kafka/MQTT keys).

---

## 📁 Project Structure

```text
healthyfins/
├── frontend/             # Vercel-deployed static assets (HTML/CSS/JS)
├── backend/              # FastAPI Python backend
│   ├── app.py            # Main API routing, Kafka consumer & AI execution
│   ├── iot_bridge.py     # Background worker translating MQTT to Kafka
│   ├── database.py       # Supabase REST API wrapper
│   ├── auth.py           # JWT generation and validation
│   └── models/           # Stored MobileNetV2 .h5 weights
├── esp8266/              # C++ code for hardware nodes
├── training/             # Jupyter Notebooks for model training
└── README.md                              
```

```
## 👨‍💻 Contact
- GitHub: https://github.com/ajayraj30002
- LinkedIn: https://www.linkedin.com/in/ajay-raj-3ee2
```


