import json
import os
import hashlib
from datetime import datetime

class JSONDatabase:
    def __init__(self):
        self.data = {"users": {}, "history": {}}
    
    def create_user(self, email, password, name, hardware_id=None):
        # Simple user creation
        user_id = hashlib.md5(email.encode()).hexdigest()[:8]
        self.data["users"][email] = {
            "id": user_id,
            "email": email,
            "name": name,
            "password": hashlib.sha256(password.encode()).hexdigest(),
            "hardware_id": hardware_id or "",
            "created_at": datetime.now().isoformat()
        }
        self.data["history"][user_id] = []
        return True, {"user_id": user_id, "email": email, "name": name}
    
    def authenticate_user(self, email, password):
        if email not in self.data["users"]:
            return False, "User not found"
        
        user = self.data["users"][email]
        hashed = hashlib.sha256(password.encode()).hexdigest()
        
        if user["password"] != hashed:
            return False, "Invalid password"
        
        return True, {
            "user_id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "hardware_id": user.get("hardware_id", "")
        }
    
    def add_prediction_history(self, user_id, image_name, prediction, confidence):
        if user_id not in self.data["history"]:
            self.data["history"][user_id] = []
        
        entry = {
            "id": len(self.data["history"][user_id]) + 1,
            "timestamp": datetime.now().isoformat(),
            "image": image_name,
            "prediction": prediction,
            "confidence": confidence,
            "treatment": "See treatment recommendations"
        }
        
        self.data["history"][user_id].insert(0, entry)
        return True
    
    def get_user_history(self, user_id, limit=50):
        if user_id not in self.data["history"]:
            return []
        return self.data["history"][user_id][:limit]
    
    def update_user_profile(self, email, name=None, hardware_id=None):
        if email not in self.data["users"]:
            return False, "User not found"
        
        user = self.data["users"][email]
        if name:
            user["name"] = name
        if hardware_id:
            user["hardware_id"] = hardware_id
        
        return True, "Profile updated"

db = JSONDatabase()
