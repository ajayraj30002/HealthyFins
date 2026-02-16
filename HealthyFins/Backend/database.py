# database.py - ULTRA SIMPLE VERSION
import os
import requests
import hashlib
from datetime import datetime
import uuid

class SupabaseDatabase:
    def __init__(self):
        # Hardcoded values - no environment variables needed!
        self.supabase_url = "https://bxfljshwfpgsnfyqemcd.supabase.co"
        self.supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ4Zmxqc2h3ZnBnc25meXFlbWNkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg0NjYxMDUsImV4cCI6MjA4NDA0MjEwNX0.M8qOkC-ajPfWgxG-PjCfY6UGLSSm5O2jmlQNTfaM3IQ"
        
        self.headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        
        print("✅ Database initialized")
    
    def create_user(self, email, password, name, hardware_id=None):
        """Create a new user"""
        try:
            # Simple insert
            user_id = str(uuid.uuid4())[:12]
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            user_data = {
                "id": user_id,
                "email": email,
                "name": name,
                "password_hash": password_hash,
                "hardware_id": hardware_id,
                "created_at": datetime.now().isoformat(),
                "last_login": datetime.now().isoformat(),
                "scan_count": 0,
                "is_active": True
            }
            
            response = requests.post(
                f"{self.supabase_url}/rest/v1/users",
                headers=self.headers,
                json=user_data
            )
            
            if response.status_code in [200, 201]:
                print(f"✅ User created: {email}")
                return True, {
                    "user_id": user_id,
                    "email": email,
                    "name": name,
                    "hardware_id": hardware_id
                }
            else:
                return False, f"Error: {response.text}"
                
        except Exception as e:
            return False, str(e)

# Create database instance
db = SupabaseDatabase()
