# database.py - COMPLETE VERSION WITH DEBUGGING
import os
import requests
import hashlib
from datetime import datetime
from typing import Optional, List, Dict
import uuid

class SupabaseDatabase:
    def __init__(self):
        # DEBUG: Print all environment variable names (not values for security)
        print("=" * 60)
        print("🔍 ENVIRONMENT VARIABLES DEBUG")
        print("=" * 60)
        
        # Get all environment variable keys
        env_keys = list(os.environ.keys())
        print(f"Total environment variables: {len(env_keys)}")
        print(f"First 10 env vars: {env_keys[:10]}")
        
        # Specifically check for Supabase variables
        print("\n🔍 Checking for Supabase variables:")
        
        # Try different possible names
        possible_url_names = ["SUPABASE_URL", "SUPABASEURL", "SUPABASE_URL", "SupabaseUrl", "supabase_url"]
        possible_key_names = ["SUPABASE_KEY", "SUPABASEKEY", "SUPABASE_KEY", "SupabaseKey", "supabase_key", "ANON_KEY", "SUPABASE_ANON_KEY"]
        
        self.supabase_url = None
        self.supabase_key = None
        
        # Try to find URL
        for name in possible_url_names:
            value = os.getenv(name)
            if value:
                print(f"✅ Found URL using: {name}")
                self.supabase_url = value
                break
            else:
                print(f"❌ Not found: {name}")
        
        # Try to find KEY
        for name in possible_key_names:
            value = os.getenv(name)
            if value:
                print(f"✅ Found KEY using: {name}")
                self.supabase_key = value
                break
            else:
                print(f"❌ Not found: {name}")
        
        # If still not found, try direct os.environ access
        if not self.supabase_url:
            for key in os.environ:
                if 'url' in key.lower() and 'supa' in key.lower():
                    self.supabase_url = os.environ[key]
                    print(f"✅ Found URL via search: {key}")
                    break
        
        if not self.supabase_key:
            for key in os.environ:
                if ('key' in key.lower() or 'anon' in key.lower()) and 'supa' in key.lower():
                    self.supabase_key = os.environ[key]
                    print(f"✅ Found KEY via search: {key}")
                    break
        
        # Final check
        print(f"\n🔍 FINAL VALUES:")
        print(f"SUPABASE_URL: {self.supabase_url[:30] if self.supabase_url else 'None'}...")
        print(f"SUPABASE_KEY: {self.supabase_key[:20] if self.supabase_key else 'None'}...")
        
        # HARDCODE FALLBACK - Use this if env vars still not found
        if not self.supabase_url or not self.supabase_key:
            print("\n⚠️ USING HARDCODED FALLBACK VALUES")
            self.supabase_url = "https://bxfljshwfpgsnfyqemcd.supabase.co"
            self.supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ4Zmxqc2h3ZnBnc25meXFlbWNkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg0NjYxMDUsImV4cCI6MjA4NDA0MjEwNX0.M8qOkC-ajPfWgxG-PjCfY6UGLSSm5O2jmlQNTfaM3IQ"
        
        # Headers for REST API
        self.headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        
        self.VALID_HARDWARE_IDS = [
            "FISHMON-001", "FISHMON-002", "FISHMON-003",
            "FISHMON-004", "FISHMON-005", "FISHMON-006",
            "AQUATECH-101", "AQUATECH-102", "AQUATECH-103",
            "HYDROPRO-201", "HYDROPRO-202"
        ]
        
        # Test connection
        try:
            if not self.supabase_url or not self.supabase_key:
                print("❌ Missing Supabase credentials!")
                return
                
            response = requests.get(
                f"{self.supabase_url}/rest/v1/users?limit=1",
                headers=self.headers
            )
            if response.status_code == 200:
                print("✅ Connected to Supabase via REST API")
            else:
                print(f"⚠️ Connection test failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
        print("=" * 60)
    
    def create_user(self, email: str, password: str, name: str, hardware_id: Optional[str] = None) -> tuple:
        """Create a new user"""
        try:
            # Check if user exists
            response = requests.get(
                f"{self.supabase_url}/rest/v1/users",
                headers=self.headers,
                params={"email": f"eq.{email}"}
            )
            
            if response.status_code == 200 and response.json():
                return False, "Email already exists"
            
            # Create user
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
                    "hardware_id": hardware_id,
                    "created_at": user_data["created_at"]
                }
            else:
                return False, f"Failed to create user: {response.text}"
                
        except Exception as e:
            print(f"❌ Create user error: {e}")
            return False, f"Registration failed: {str(e)}"
    
    def authenticate_user(self, email: str, password: str) -> tuple:
        """Authenticate user login"""
        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            response = requests.get(
                f"{self.supabase_url}/rest/v1/users",
                headers=self.headers,
                params={"email": f"eq.{email}"}
            )
            
            if response.status_code != 200:
                return False, "Invalid email or password"
            
            users = response.json()
            if not users:
                return False, "Invalid email or password"
            
            user = users[0]
            
            if user["password_hash"] != password_hash:
                return False, "Invalid email or password"
            
            # Update last login
            requests.patch(
                f"{self.supabase_url}/rest/v1/users",
                headers=self.headers,
                params={"id": f"eq.{user['id']}"},
                json={"last_login": datetime.now().isoformat()}
            )
            
            return True, {
                "user_id": user["id"],
                "email": user["email"],
                "name": user["name"],
                "hardware_id": user.get("hardware_id", ""),
                "created_at": user["created_at"],
                "scan_count": user.get("scan_count", 0),
                "last_login": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False, f"Authentication failed: {str(e)}"
    
    def add_prediction_history(self, user_id: str, image_name: str, 
                              prediction: str, confidence: float, 
                              image_url: Optional[str] = None,
                              symptoms: Optional[List[str]] = None,
                              model_type: str = "ai_model") -> bool:
        """Add a prediction to user's history"""
        try:
            entry_id = str(uuid.uuid4())
            
            history_entry = {
                "id": entry_id,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "image_name": image_name[:100],
                "image_url": image_url,
                "prediction": prediction,
                "confidence": float(confidence),
                "model_type": model_type,
                "symptoms": symptoms or [],
                "treatment_plan": self._generate_treatment_plan(prediction, confidence),
                "is_urgent": confidence > 70 and "healthy" not in prediction.lower()
            }
            
            response = requests.post(
                f"{self.supabase_url}/rest/v1/prediction_history",
                headers=self.headers,
                json=history_entry
            )
            
            if response.status_code in [200, 201]:
                # Update user's scan count
                user_response = requests.get(
                    f"{self.supabase_url}/rest/v1/users",
                    headers=self.headers,
                    params={"id": f"eq.{user_id}"}
                )
                
                if user_response.status_code == 200 and user_response.json():
                    current_count = user_response.json()[0].get("scan_count", 0)
                    new_count = current_count + 1
                    
                    requests.patch(
                        f"{self.supabase_url}/rest/v1/users",
                        headers=self.headers,
                        params={"id": f"eq.{user_id}"},
                        json={
                            "scan_count": new_count,
                            "last_scan": history_entry["timestamp"]
                        }
                    )
                
                print(f"✅ History saved: {entry_id}")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Save history error: {e}")
            return False
    
    def get_user_history(self, user_id: str, limit: int = 50, 
                        offset: int = 0, filter_disease: Optional[str] = None) -> List[Dict]:
        """Get user's prediction history"""
        try:
            params = {
                "user_id": f"eq.{user_id}",
                "order": "timestamp.desc",
                "limit": limit,
                "offset": offset
            }
            
            if filter_disease and filter_disease != "all":
                if filter_disease == "healthy":
                    params["prediction"] = "ilike.*healthy*"
                elif filter_disease == "bacterial":
                    params["prediction"] = "ilike.*bacterial*"
                elif filter_disease == "fungal":
                    params["prediction"] = "ilike.*fungal*"
                elif filter_disease == "parasitic":
                    params["prediction"] = "ilike.*parasitic*"
                elif filter_disease == "viral":
                    params["prediction"] = "ilike.*viral*"
            
            response = requests.get(
                f"{self.supabase_url}/rest/v1/prediction_history",
                headers=self.headers,
                params=params
            )
            
            return response.json() if response.status_code == 200 else []
            
        except Exception as e:
            print(f"❌ Get history error: {e}")
            return []
    
    def get_history_stats(self, user_id: str) -> Dict:
        """Get statistics about user's history"""
        try:
            # Get all history
            response = requests.get(
                f"{self.supabase_url}/rest/v1/prediction_history",
                headers=self.headers,
                params={"user_id": f"eq.{user_id}"}
            )
            
            if response.status_code != 200:
                return {"total": 0, "healthy": 0, "disease": 0, "disease_types": {}, "last_scan": None, "avg_confidence": 0}
            
            history = response.json()
            total = len(history)
            
            if total == 0:
                return {"total": 0, "healthy": 0, "disease": 0, "disease_types": {}, "last_scan": None, "avg_confidence": 0}
            
            # Calculate stats
            healthy = sum(1 for h in history if "healthy" in h["prediction"].lower())
            disease = total - healthy
            
            disease_types = {}
            last_scan = None
            total_confidence = 0
            
            for entry in history:
                pred = entry["prediction"].lower()
                if "healthy" not in pred:
                    if "bacterial" in pred:
                        disease_types["Bacterial"] = disease_types.get("Bacterial", 0) + 1
                    elif "fungal" in pred:
                        disease_types["Fungal"] = disease_types.get("Fungal", 0) + 1
                    elif "parasitic" in pred:
                        disease_types["Parasitic"] = disease_types.get("Parasitic", 0) + 1
                    elif "viral" in pred:
                        disease_types["Viral"] = disease_types.get("Viral", 0) + 1
                    else:
                        disease_types["Other"] = disease_types.get("Other", 0) + 1
                
                total_confidence += entry.get("confidence", 0)
                
                if not last_scan or entry["timestamp"] > last_scan:
                    last_scan = entry["timestamp"]
            
            return {
                "total": total,
                "healthy": healthy,
                "disease": disease,
                "disease_types": disease_types,
                "last_scan": last_scan,
                "avg_confidence": round(total_confidence / total, 2)
            }
            
        except Exception as e:
            print(f"❌ Get stats error: {e}")
            return {"total": 0, "healthy": 0, "disease": 0, "disease_types": {}, "last_scan": None, "avg_confidence": 0}
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile by ID"""
        try:
            response = requests.get(
                f"{self.supabase_url}/rest/v1/users",
                headers=self.headers,
                params={"id": f"eq.{user_id}"}
            )
            
            if response.status_code == 200 and response.json():
                user = response.json()[0]
                if "password_hash" in user:
                    del user["password_hash"]
                return user
            
            return None
            
        except Exception as e:
            print(f"❌ Get profile error: {e}")
            return None
    
    def update_user_profile(self, user_id: str, name: Optional[str] = None,
                           hardware_id: Optional[str] = None) -> tuple:
        """Update user profile"""
        try:
            update_data = {}
            if name:
                update_data["name"] = name
            if hardware_id is not None:
                update_data["hardware_id"] = hardware_id if hardware_id else None
            
            if not update_data:
                return False, "No data to update"
            
            response = requests.patch(
                f"{self.supabase_url}/rest/v1/users",
                headers=self.headers,
                params={"id": f"eq.{user_id}"},
                json=update_data
            )
            
            if response.status_code in [200, 204]:
                return True, "Profile updated successfully"
            else:
                return False, "Update failed"
            
        except Exception as e:
            print(f"❌ Update profile error: {e}")
            return False, f"Update failed: {str(e)}"
    
    def _generate_treatment_plan(self, prediction: str, confidence: float) -> str:
        """Generate treatment plan based on prediction"""
        plans = {
            "healthy": "✅ HEALTHY FISH - Maintain current water conditions. Regular monitoring recommended.\n\n1. Continue normal feeding schedule\n2. Weekly 20% water changes\n3. Monitor water parameters (pH 6.5-8.0, temp 24-28°C)\n4. Observe behavior daily",
            "bacterial": "🦠 BACTERIAL INFECTION - Immediate action required.\n\n1. Isolate affected fish immediately\n2. Antibiotic treatment (Kanamycin or Erythromycin)\n3. Salt bath: 1 tbsp per 5 gallons\n4. Increase water temperature to 28°C\n5. Daily 30% water changes\n6. Consult veterinarian if no improvement in 48 hours",
            "fungal": "🍄 FUNGAL INFECTION - Treatment needed.\n\n1. Antifungal medication (Methylene Blue)\n2. Salt bath: 2 tsp per gallon for 30 minutes\n3. Improve water quality immediately\n4. Remove any dead tissue carefully\n5. Increase aeration\n6. Treat for 7-10 days minimum",
            "parasitic": "🐛 PARASITIC INFECTION - Quarantine required.\n\n1. Anti-parasitic medication (Praziquantel)\n2. Formalin bath (follow instructions carefully)\n3. Raise temperature to 30°C gradually\n4. Vacuum substrate thoroughly\n5. Treat all fish in tank\n6. Repeat treatment after 7 days",
            "viral": "🦠 VIRAL INFECTION - Supportive care.\n\n1. No specific treatment available\n2. Maintain optimal water conditions\n3. Add aquarium salt (1 tsp per gallon)\n4. Provide high-quality food\n5. Reduce stress (dim lights, no handling)\n6. Watch for secondary infections",
            "default": "⚠️ UNKNOWN CONDITION - General care.\n\n1. Quarantine affected fish\n2. Improve water quality (test parameters)\n3. Consult aquatic veterinarian\n4. Take clear photos for diagnosis\n5. Monitor symptoms closely"
        }
        
        pred_lower = prediction.lower()
        
        if "healthy" in pred_lower:
            return plans["healthy"]
        elif "bacterial" in pred_lower:
            return plans["bacterial"]
        elif "fungal" in pred_lower:
            return plans["fungal"]
        elif "parasitic" in pred_lower:
            return plans["parasitic"]
        elif "viral" in pred_lower:
            return plans["viral"]
        else:
            return plans["default"]
    
    def get_hardware_ids(self) -> List[str]:
        """Get list of valid hardware IDs"""
        return self.VALID_HARDWARE_IDS.copy()
    
    def check_hardware_available(self, hardware_id: str) -> bool:
        """Check if hardware ID is available"""
        try:
            response = requests.get(
                f"{self.supabase_url}/rest/v1/users",
                headers=self.headers,
                params={"hardware_id": f"eq.{hardware_id}"}
            )
            return len(response.json()) == 0
        except:
            return True
    
    def search_history(self, user_id: str, query: str, limit: int = 20) -> List[Dict]:
        """Search in user's history"""
        try:
            response = requests.get(
                f"{self.supabase_url}/rest/v1/prediction_history",
                headers=self.headers,
                params={
                    "user_id": f"eq.{user_id}",
                    "prediction": f"ilike.*{query}*",
                    "order": "timestamp.desc",
                    "limit": limit
                }
            )
            return response.json() if response.status_code == 200 else []
        except:
            return []
    
    def delete_history_entry(self, user_id: str, entry_id: str) -> bool:
        """Delete a history entry"""
        try:
            response = requests.delete(
                f"{self.supabase_url}/rest/v1/prediction_history",
                headers=self.headers,
                params={
                    "id": f"eq.{entry_id}",
                    "user_id": f"eq.{user_id}"
                }
            )
            return response.status_code in [200, 204]
        except:
            return False
    
    def clear_user_history(self, user_id: str) -> bool:
        """Clear all history for a user"""
        try:
            response = requests.delete(
                f"{self.supabase_url}/rest/v1/prediction_history",
                headers=self.headers,
                params={"user_id": f"eq.{user_id}"}
            )
            
            # Reset scan count
            requests.patch(
                f"{self.supabase_url}/rest/v1/users",
                headers=self.headers,
                params={"id": f"eq.{user_id}"},
                json={"scan_count": 0, "last_scan": None}
            )
            
            return response.status_code in [200, 204]
        except:
            return False

# Create global instance
print("=" * 50)
print("🐟 INITIALIZING DATABASE")
print("=" * 50)

db = SupabaseDatabase()

print(f"📊 Database Type: Supabase REST API")
print(f"🔧 Valid Hardware IDs: {len(db.VALID_HARDWARE_IDS)}")
print("=" * 50)
