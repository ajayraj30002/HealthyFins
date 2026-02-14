# database.py - COMPLETE FIXED VERSION FOR RENDER DEPLOYMENT
import os
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
import json

# IMPORTANT: Don't import supabase at the top level
# We'll import it lazily inside methods

class SupabaseDatabase:
    def __init__(self):
        # Get Supabase credentials from environment variables
        self.supabase_url = os.getenv("SUPABASE_URL", "https://bxfljshwfpgsnfyqemcd.supabase.co")
        self.supabase_key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ4Zmxqc2h3ZnBnc25meXFlbWNkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg0NjYxMDUsImV4cCI6MjA4NDA0MjEwNX0.M8qOkC-ajPfWgxG-PjCfY6UGLSSm5O2jmlQNTfaM3IQ")
        self._supabase_client = None
        
        # Predefined hardware IDs (only these are allowed)
        self.VALID_HARDWARE_IDS = [
            "FISHMON-001", "FISHMON-002", "FISHMON-003",
            "FISHMON-004", "FISHMON-005", "FISHMON-006",
            "AQUATECH-101", "AQUATECH-102", "AQUATECH-103",
            "HYDROPRO-201", "HYDROPRO-202"
        ]
        
        # Print status (but DON'T connect yet)
        print("=" * 50)
        print("🐟 HEALTHYFINS DATABASE INITIALIZED")
        print(f"🔧 Supabase URL configured: {'Yes' if self.supabase_url else 'No'}")
        print(f"🔧 Supabase Key configured: {'Yes' if self.supabase_key else 'No'}")
        print(f"🔧 Valid Hardware IDs: {len(self.VALID_HARDWARE_IDS)}")
        print("=" * 50)
    
    @property
    def supabase(self):
        """Lazy-load Supabase client only when needed"""
        if self._supabase_client is None and self.supabase_url and self.supabase_key:
            try:
                # Import supabase only when needed
                import supabase
                self._supabase_client = supabase.create_client(self.supabase_url, self.supabase_key)
                print("✅ Supabase client created successfully")
            except Exception as e:
                print(f"⚠️ Supabase client creation failed: {e}")
                self._supabase_client = None
        return self._supabase_client
    
    # ========== USER MANAGEMENT ==========
    
    def create_user(self, email: str, password: str, name: str, hardware_id: Optional[str] = None) -> tuple:
        """Create a new user with validation"""
        try:
            print(f"📝 Attempting to create user: {email}")
            
            # Validate hardware_id if provided
            if hardware_id and hardware_id not in self.VALID_HARDWARE_IDS:
                return False, f"Invalid hardware ID. Must be one of: {', '.join(self.VALID_HARDWARE_IDS[:3])}..."
            
            # Hash password
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            user_id = hashlib.md5(email.encode()).hexdigest()[:12]
            
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
            
            # Try Supabase first
            if self.supabase:
                try:
                    # Check if email already exists
                    existing = self.supabase.table("users").select("*").eq("email", email).execute()
                    if existing.data:
                        return False, "Email already registered"
                    
                    # Check hardware_id if provided
                    if hardware_id:
                        existing_hw = self.supabase.table("users").select("*").eq("hardware_id", hardware_id).execute()
                        if existing_hw.data:
                            return False, "Hardware ID already registered to another user"
                    
                    # Save to Supabase
                    result = self.supabase.table("users").insert(user_data).execute()
                    if result.data:
                        print(f"✅ User created in Supabase: {email}")
                        return True, {
                            "user_id": user_id,
                            "email": email,
                            "name": name,
                            "hardware_id": hardware_id,
                            "created_at": user_data["created_at"]
                        }
                except Exception as e:
                    print(f"⚠️ Supabase insert failed, falling back to file storage: {e}")
                    # Fall through to file storage
            
            # File-based fallback (ensures registration always works)
            return self._save_user_to_file(user_data, email)
            
        except Exception as e:
            print(f"❌ Create user error: {e}")
            return False, f"Registration failed: {str(e)}"
    
    def _save_user_to_file(self, user_data: dict, email: str) -> tuple:
        """Fallback: Save user to JSON file"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs("data", exist_ok=True)
            
            # Load existing users
            users_file = "data/users.json"
            if os.path.exists(users_file):
                with open(users_file, 'r') as f:
                    users = json.load(f)
            else:
                users = {}
            
            # Check if email exists
            if email in users:
                return False, "Email already registered"
            
            # Save user
            users[email] = user_data
            with open(users_file, 'w') as f:
                json.dump(users, f, indent=2)
            
            print(f"✅ User created in file storage: {email}")
            return True, {
                "user_id": user_data["id"],
                "email": user_data["email"],
                "name": user_data["name"],
                "hardware_id": user_data.get("hardware_id"),
                "created_at": user_data["created_at"]
            }
        except Exception as e:
            print(f"❌ File save error: {e}")
            return False, f"Registration failed: {str(e)}"
    
    def authenticate_user(self, email: str, password: str) -> tuple:
        """Authenticate user login"""
        try:
            print(f"🔐 Authenticating user: {email}")
            
            # Hash the provided password
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            # Try Supabase first
            if self.supabase:
                try:
                    result = self.supabase.table("users").select("*").eq("email", email).execute()
                    if result.data:
                        user = result.data[0]
                        if user.get("password_hash") == password_hash:
                            # Update last login
                            self.supabase.table("users").update({
                                "last_login": datetime.now().isoformat()
                            }).eq("id", user["id"]).execute()
                            
                            return True, {
                                "user_id": user["id"],
                                "email": user["email"],
                                "name": user["name"],
                                "hardware_id": user.get("hardware_id", ""),
                                "created_at": user["created_at"],
                                "scan_count": user.get("scan_count", 0)
                            }
                except Exception as e:
                    print(f"⚠️ Supabase auth failed, trying file storage: {e}")
                    # Fall through to file storage
            
            # Try file storage
            users_file = "data/users.json"
            if os.path.exists(users_file):
                with open(users_file, 'r') as f:
                    users = json.load(f)
                
                if email in users:
                    user = users[email]
                    if user.get("password_hash") == password_hash:
                        return True, {
                            "user_id": user["id"],
                            "email": user["email"],
                            "name": user["name"],
                            "hardware_id": user.get("hardware_id", ""),
                            "created_at": user["created_at"],
                            "scan_count": user.get("scan_count", 0)
                        }
            
            return False, "Invalid email or password"
            
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False, f"Authentication failed: {str(e)}"
    
    # ========== PREDICTION HISTORY ==========
    
    def add_prediction_history(self, user_id: str, image_name: str, 
                              prediction: str, confidence: float, 
                              image_url: Optional[str] = None,
                              symptoms: Optional[List[str]] = None,
                              model_type: str = "ai_model") -> bool:
        """Add a prediction to user's history"""
        try:
            entry_id = f"{user_id}_{int(datetime.now().timestamp() * 1000)}"
            
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
            
            # Try Supabase
            if self.supabase:
                try:
                    result = self.supabase.table("prediction_history").insert(history_entry).execute()
                    if result.data:
                        # Update user's scan count
                        self.supabase.table("users").update({
                            "scan_count": self.supabase.table("users").select("scan_count").eq("id", user_id).execute().data[0].get("scan_count", 0) + 1
                        }).eq("id", user_id).execute()
                        return True
                except Exception as e:
                    print(f"⚠️ Supabase history save failed: {e}")
            
            # File fallback
            return self._save_history_to_file(user_id, history_entry)
            
        except Exception as e:
            print(f"❌ Save history error: {e}")
            return False
    
    def _save_history_to_file(self, user_id: str, history_entry: dict) -> bool:
        """Save history to file as fallback"""
        try:
            os.makedirs("data", exist_ok=True)
            history_file = f"data/history_{user_id}.json"
            
            history = []
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
            
            history.insert(0, history_entry)
            
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            return True
        except Exception as e:
            print(f"❌ File history save error: {e}")
            return False
    
    def get_user_history(self, user_id: str, limit: int = 50, 
                        offset: int = 0, filter_disease: Optional[str] = None) -> List[Dict]:
        """Get user's prediction history"""
        try:
            # Try Supabase first
            if self.supabase:
                try:
                    query = self.supabase.table("prediction_history").select("*").eq("user_id", user_id)
                    
                    if filter_disease and filter_disease != "all":
                        if filter_disease == "healthy":
                            query = query.ilike("prediction", "%healthy%")
                        else:
                            query = query.ilike("prediction", f"%{filter_disease}%")
                    
                    result = query.order("timestamp", desc=True).range(offset, offset + limit - 1).execute()
                    if result.data:
                        return result.data
                except Exception as e:
                    print(f"⚠️ Supabase history fetch failed: {e}")
            
            # File fallback
            history_file = f"data/history_{user_id}.json"
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                if filter_disease and filter_disease != "all":
                    history = [h for h in history if filter_disease.lower() in h["prediction"].lower()]
                
                return history[offset:offset + limit]
            
            return []
                
        except Exception as e:
            print(f"❌ Get history error: {e}")
            return []
    
    def get_history_stats(self, user_id: str) -> Dict:
        """Get statistics about user's history"""
        try:
            history = self.get_user_history(user_id, limit=1000)
            
            total = len(history)
            healthy = sum(1 for h in history if "healthy" in h["prediction"].lower())
            disease = total - healthy
            
            disease_types = {}
            for h in history:
                if "healthy" not in h["prediction"].lower():
                    pred = h["prediction"].lower()
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
            
            last_scan = history[0]["timestamp"] if history else None
            
            return {
                "total": total,
                "healthy": healthy,
                "disease": disease,
                "disease_types": disease_types,
                "last_scan": last_scan
            }
                
        except Exception as e:
            print(f"❌ Get stats error: {e}")
            return {"total": 0, "healthy": 0, "disease": 0, "disease_types": {}, "last_scan": None}
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile by ID"""
        try:
            if self.supabase:
                try:
                    result = self.supabase.table("users").select("*").eq("id", user_id).execute()
                    if result.data:
                        user = result.data[0]
                        if "password_hash" in user:
                            del user["password_hash"]
                        return user
                except Exception as e:
                    print(f"⚠️ Supabase profile fetch failed: {e}")
            
            # File fallback
            users_file = "data/users.json"
            if os.path.exists(users_file):
                with open(users_file, 'r') as f:
                    users = json.load(f)
                
                for user in users.values():
                    if user["id"] == user_id:
                        user_copy = user.copy()
                        if "password_hash" in user_copy:
                            del user_copy["password_hash"]
                        return user_copy
            
            return None
            
        except Exception as e:
            print(f"❌ Get profile error: {e}")
            return None
    
    def _generate_treatment_plan(self, prediction: str, confidence: float) -> str:
        """Generate treatment plan based on prediction"""
        plans = {
            "healthy": "✅ HEALTHY FISH - Maintain current water conditions. Regular monitoring recommended.",
            "bacterial": "🦠 BACTERIAL INFECTION - Immediate action required. Isolate affected fish, use antibiotics.",
            "fungal": "🍄 FUNGAL INFECTION - Treatment needed. Use antifungal medication, improve water quality.",
            "parasitic": "🐛 PARASITIC INFECTION - Quarantine required. Use anti-parasitic medication.",
            "viral": "🦠 VIRAL INFECTION - Supportive care only. Maintain optimal water conditions."
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
            return "⚠️ UNKNOWN CONDITION - Consult with aquatic veterinarian."
    
    def get_hardware_ids(self) -> List[str]:
        """Get list of valid hardware IDs"""
        return self.VALID_HARDWARE_IDS.copy()
    
    def check_hardware_available(self, hardware_id: str) -> bool:
        """Check if hardware ID is available"""
        try:
            if self.supabase:
                result = self.supabase.table("users").select("hardware_id").eq("hardware_id", hardware_id).execute()
                return len(result.data) == 0
            return True
        except:
            return True

# Create global instance (NO CONNECTION ATTEMPTED HERE)
db = SupabaseDatabase()
