import json
import os
from datetime import datetime
from typing import Optional, List, Dict
import hashlib

class JSONDatabase:
    """Simple JSON-based database (No SQL needed!)"""
    
    def __init__(self, db_file='database.json'):
        self.db_file = db_file
        self.data = self._load_data()
    
    def _load_data(self):
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r') as f:
                try:
                    return json.load(f)
                except:
                    return {"users": {}, "history": {}}
        return {"users": {}, "history": {}}
    
    def _save_data(self):
        with open(self.db_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def create_user(self, email: str, password: str, name: str, hardware_id: str = None):
        """Create new user"""
        if email in self.data["users"]:
            return False, "Email already exists"
        
        user_id = hashlib.md5(email.encode()).hexdigest()[:8]
        
        self.data["users"][email] = {
            "id": user_id,
            "email": email,
            "name": name,
            "password": hashlib.sha256(password.encode()).hexdigest(),
            "hardware_id": hardware_id or "",
            "created_at": datetime.now().isoformat(),
            "last_login": datetime.now().isoformat()
        }
        
        # Initialize history for user
        self.data["history"][user_id] = []
        
        self._save_data()
        return True, {"user_id": user_id, "email": email, "name": name}
    
    def authenticate_user(self, email: str, password: str):
        """Authenticate user"""
        if email not in self.data["users"]:
            return False, "User not found"
        
        user = self.data["users"][email]
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        
        if user["password"] != hashed_pw:
            return False, "Invalid password"
        
        # Update last login
        user["last_login"] = datetime.now().isoformat()
        self._save_data()
        
        return True, {
            "user_id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "hardware_id": user["hardware_id"]
        }
    
    def add_prediction_history(self, user_id: str, image_name: str, prediction: str, confidence: float):
        """Add prediction to user history"""
        if user_id not in self.data["history"]:
            self.data["history"][user_id] = []
        
        history_entry = {
            "id": len(self.data["history"][user_id]) + 1,
            "timestamp": datetime.now().isoformat(),
            "image": image_name,
            "prediction": prediction,
            "confidence": confidence,
            "treatment": self._get_treatment(prediction)
        }
        
        self.data["history"][user_id].insert(0, history_entry)  # Add to beginning
        self._save_data()
        return True
    
    def get_user_history(self, user_id: str, limit: int = 50):
        """Get user's prediction history"""
        if user_id not in self.data["history"]:
            return []
        return self.data["history"][user_id][:limit]
    
    def update_user_profile(self, email: str, name: str = None, hardware_id: str = None):
        """Update user profile"""
        if email not in self.data["users"]:
            return False, "User not found"
        
        user = self.data["users"][email]
        if name:
            user["name"] = name
        if hardware_id:
            user["hardware_id"] = hardware_id
        
        self._save_data()
        return True, "Profile updated"
    
    def _get_treatment(self, disease: str) -> str:
        """Get treatment recommendation"""
        treatments = {
            "healthy": "✅ Fish appears healthy. Continue regular maintenance with weekly water changes.",
            "white spot": "🚨 Raise temperature to 30°C, add aquarium salt (1 tbsp/20L), use anti-parasitic medication for 10-14 days.",
            "fin rot": "⚠️ Improve water quality immediately, use antibacterial medication, remove sharp decorations.",
            "fungal": "⚠️ Use antifungal medication, improve water quality, consider salt bath treatment.",
            "parasite": "🚨 Use anti-parasitic medication, quarantine affected fish, clean tank thoroughly."
        }
        
        disease_lower = disease.lower()
        for key, value in treatments.items():
            if key in disease_lower:
                return value
        
        return "Consult a veterinarian for proper diagnosis and treatment."

# Global database instance
db = JSONDatabase()