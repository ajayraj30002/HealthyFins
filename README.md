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

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [User History](#-user-history--authentication)
- [pH Monitoring](#-ph-monitoring--hardware-integration)
- [Hardware Integration](#-hardware-integration-raspberry-pi--sensors)
- [Supabase Database](#-supabase-database-schema)
- [MobileNetV2 Architecture](#-mobilenetv2-architecture)
- [API Documentation](#-api-documentation)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Deployment](#-deployment)
- [Testing](#-testing)
- [Future Roadmap](#-future-roadmap)
- [Contributing](#-contributing)
- [Contact](#-contact)

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

---

## 💻 Tech Stack

### Frontend (Vercel)
```yaml
Core:
  - HTML5, CSS3, Vanilla JavaScript
  - Chart.js for data visualization
  - Responsive design (mobile-first)
  
Features:
  - Dashboard with real-time updates
  - History viewer with filters
  - Sensor data graphs
  - Profile management
