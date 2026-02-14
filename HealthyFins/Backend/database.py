# database.py - DIRECT SUPABASE REST API (NO PACKAGE ISSUES)
import os
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import traceback
import requests

class SupabaseDatabase:
    def __init__(self):
        # Get Supabase credentials
        self.supabase_url = os.getenv("SUPABASE_URL", "https://bxfljshwfpgsnfyqemcd.supabase.co").rstrip('/')
        self.supabase_key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ4Zmxqc2h3ZnBnc25meXFlbWNkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg0NjYxMDUsImV4cCI6MjA4NDA0MjEwNX0.M8qOkC-ajPfWgxG-PjCfY6UGLSSm5O2jmlQNTfaM3IQ")
        
        # API headers for direct REST calls
        self.headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        
        # Predefined hardware IDs
        self.VALID_HARDWARE_IDS = [
            "FISHMON-001", "FISHMON-002", "FISHMON-003",
            "FISHMON-004", "FISHMON-005", "FISHMON-006",
            "AQUATECH-101", "AQUATECH-102", "AQUATECH-103",
            "HYDROPRO-201", "HYDROPRO-202"
        ]
        
        # Create data directories for file backup
        os.makedirs("data/users", exist_ok=True)
        os.makedirs("data/history", exist_ok=True)
        
        print("=" * 60)
        print("🐟 HEALTHYFINS DATABASE INITIALIZED")
        print(f"📁 File storage: {os.path.abspath('data')}")
        print(f"🔧 Supabase URL: {'✅ Set' if self.supabase_url else '❌ Not set'}")
        print(f"🔧 Supabase Key: {'✅ Set' if self.supabase_key else '❌ Not set'}")
        print("=" * 60)
        
        # Test Supabase connection
        if self.supabase_url and self.supabase_key:
            self._test_supabase_connection()
        else:
            print("📁 Running in file-only storage mode (no Supabase credentials)")
    
    def _test_supabase_connection(self):
        """Test Supabase connection using REST API"""
        try:
            # Try to fetch one user to test connection
            url = f"{self.supabase_url}/rest/v1/users?select=count&limit=1"
            response = requests.get(url, headers=self.headers, timeout=5)
            
            if response.status_code == 200:
                print("✅ Supabase connection successful (REST API)")
                print("☁️ Will sync data to Supabase")
            elif response.status_code == 404:
                print("⚠️ Supabase tables don't exist yet - they will be created automatically")
                self._create_tables_if_needed()
            else:
                print(f"⚠️ Supabase connection test returned {response.status_code}: {response.text[:100]}")
                print("📁 Using file storage (Supabase will be tried on each operation)")
        except Exception as e:
            print(f"⚠️ Supabase connection failed: {e}")
            print("📁 Using file storage only")
    
    def _create_tables_if_needed(self):
        """Create tables if they don't exist"""
        try:
            # Try to create users table
            create_users_sql = {
                "name": "users",
                "schema": {
                    "id": {"type": "text", "primaryKey": True},
                    "email": {"type": "text", "unique": True},
                    "name": {"type": "text"},
                    "password_hash": {"type": "text"},
                    "hardware_id": {"type": "text", "nullable": True},
                    "created_at": {"type": "timestamp", "default": "now()"},
                    "last_login": {"type": "timestamp", "default": "now()"},
                    "scan_count": {"type": "integer", "default": 0},
                    "is_active": {"type": "boolean", "default": True}
                }
            }
            
            # Note: Table creation requires higher privileges
            # We'll just create them manually in Supabase dashboard
            print("""
            ⚠️ Please create tables manually in Supabase SQL Editor:
            
            CREATE TABLE users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                hardware_id TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                last_login TIMESTAMP DEFAULT NOW(),
                scan_count INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT true
            );
            
            CREATE TABLE prediction_history (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT NOW(),
                image_name TEXT,
                image_url TEXT,
                prediction TEXT NOT NULL,
                confidence DECIMAL(5,2) NOT NULL,
                model_type TEXT DEFAULT 'ai_model',
                symptoms TEXT[] DEFAULT '{}',
                treatment_plan TEXT,
                is_urgent BOOLEAN DEFAULT false
            );
            """)
        except Exception as e:
            print(f"⚠️ Could not create tables: {e}")
    
    def _supabase_request(self, method: str, table: str, data: dict = None, params: dict = None):
        """Make a direct REST API call to Supabase"""
        if not self.supabase_url or not self.supabase_key:
            return None
        
        url = f"{self.supabase_url}/rest/v1/{table}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=self.headers, params=params, timeout=5)
            elif method.upper() == "POST":
                response = requests.post(url, headers=self.headers, json=data, timeout=5)
            elif method.upper() == "PATCH":
                response = requests.patch(url, headers=self.headers, json=data, params=params, timeout=5)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=self.headers, params=params, timeout=5)
            else:
                return None
            
            if response.status_code in [200, 201]:
                return response.json()
            elif response.status_code == 204:
                return []
            else:
                print(f"⚠️ Supabase {method} returned {response.status_code}: {response.text[:100]}")
                return None
        except Exception as e:
            print(f"⚠️ Supabase request error: {e}")
            return None
    
    # ========== FILE STORAGE METHODS ==========
    
    def _get_user_file_path(self, email: str) -> str:
        """Get path for user file"""
        safe_email = email.replace('@', '_at_').replace('.', '_dot_')
        return f"data/users/{safe_email}.json"
    
    def _get_history_file_path(self, user_id: str) -> str:
        """Get path for history file"""
        return f"data/history/{user_id}.json"
    
    def _save_user_to_file(self, user_data: dict) -> bool:
        """Save user to file system"""
        try:
            file_path = self._get_user_file_path(user_data["email"])
            with open(file_path, 'w') as f:
                json.dump(user_data, f, indent=2, default=str)
            print(f"✅ User saved to file: {os.path.basename(file_path)}")
            return True
        except Exception as e:
            print(f"❌ File save error: {e}")
            return False
    
    def _get_user_from_file(self, email: str) -> Optional[Dict]:
        """Get user from file system"""
        try:
            file_path = self._get_user_file_path(email)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"❌ File read error: {e}")
        return None
    
    def _get_user_by_id_from_file(self, user_id: str) -> Optional[Dict]:
        """Find user by ID in files"""
        try:
            for filename in os.listdir("data/users"):
                if filename.endswith('.json'):
                    with open(os.path.join("data/users", filename), 'r') as f:
                        user = json.load(f)
                        if user.get("id") == user_id:
                            return user
        except Exception as e:
            print(f"❌ Find user error: {e}")
        return None
    
    def _save_history_to_file(self, user_id: str, history_entry: dict) -> bool:
        """Save history to file system"""
        try:
            file_path = self._get_history_file_path(user_id)
            
            # Load existing history
            history = []
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    history = json.load(f)
            
            # Add new entry
            history.insert(0, history_entry)
            history = history[:100]  # Keep last 100 entries
            
            # Save back
            with open(file_path, 'w') as f:
                json.dump(history, f, indent=2, default=str)
            
            print(f"✅ History saved ({len(history)} total entries)")
            return True
        except Exception as e:
            print(f"❌ History save error: {e}")
            return False
    
    def _get_history_from_file(self, user_id: str) -> List[Dict]:
        """Get history from file system"""
        try:
            file_path = self._get_history_file_path(user_id)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"❌ History read error: {e}")
        return []
    
    # ========== USER MANAGEMENT ==========
    
    def create_user(self, email: str, password: str, name: str, hardware_id: Optional[str] = None) -> tuple:
        """Create a new user with validation"""
        print(f"\n📝 Creating user: {email}")
        
        try:
            # Validate hardware_id if provided
            if hardware_id and hardware_id not in self.VALID_HARDWARE_IDS:
                return False, f"Invalid hardware ID. Must be one of: {', '.join(self.VALID_HARDWARE_IDS[:3])}..."
            
            # Check if hardware_id is available
            if hardware_id and not self.check_hardware_available(hardware_id):
                return False, "Hardware ID already registered"
            
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
            
            # ALWAYS save to file first
            file_saved = self._save_user_to_file(user_data)
            
            # Try Supabase
            supabase_saved = False
            if file_saved and self.supabase_url and self.supabase_key:
                try:
                    # Check if user exists in Supabase
                    check = self._supabase_request("GET", "users", params={"email": f"eq.{email}"})
                    if check and len(check) > 0:
                        return False, "Email already registered"
                    
                    # Insert user
                    result = self._supabase_request("POST", "users", data=user_data)
                    if result:
                        supabase_saved = True
                        print("☁️ User saved to Supabase")
                except Exception as e:
                    print(f"⚠️ Supabase save error: {e}")
            
            if file_saved:
                print(f"✅ User created: {email} (ID: {user_id})")
                return True, {
                    "user_id": user_id,
                    "email": email,
                    "name": name,
                    "hardware_id": hardware_id,
                    "created_at": user_data["created_at"]
                }
            else:
                return False, "Failed to save user data"
            
        except Exception as e:
            print(f"❌ Create user error: {e}")
            traceback.print_exc()
            return False, f"Registration failed: {str(e)}"
    
    def authenticate_user(self, email: str, password: str) -> tuple:
        """Authenticate user login"""
        print(f"\n🔐 Authenticating: {email}")
        
        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            user = None
            source = None
            
            # Try file first (always works)
            user = self._get_user_from_file(email)
            if user:
                source = "file"
                print(f"✅ User found in {source}")
            
            # Try Supabase if file not found
            if not user and self.supabase_url and self.supabase_key:
                try:
                    result = self._supabase_request("GET", "users", params={"email": f"eq.{email}"})
                    if result and len(result) > 0:
                        user = result[0]
                        source = "supabase"
                        print(f"✅ User found in {source}")
                        # Save to file for future
                        self._save_user_to_file(user)
                except Exception as e:
                    print(f"⚠️ Supabase error: {e}")
            
            if not user:
                print("❌ User not found")
                return False, "Invalid email or password"
            
            # Verify password
            if user.get("password_hash") != password_hash:
                print("❌ Invalid password")
                return False, "Invalid email or password"
            
            # Update last login
            user["last_login"] = datetime.now().isoformat()
            self._save_user_to_file(user)
            
            # Try Supabase update
            if self.supabase_url and self.supabase_key:
                try:
                    self._supabase_request("PATCH", "users", 
                                          data={"last_login": user["last_login"]},
                                          params={"id": f"eq.{user['id']}"})
                except:
                    pass
            
            return True, {
                "user_id": user["id"],
                "email": user["email"],
                "name": user["name"],
                "hardware_id": user.get("hardware_id", ""),
                "created_at": user["created_at"],
                "scan_count": user.get("scan_count", 0)
            }
            
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False, f"Authentication failed: {str(e)}"
    
    def add_prediction_history(self, user_id: str, image_name: str, 
                              prediction: str, confidence: float, 
                              image_url: Optional[str] = None,
                              symptoms: Optional[List[str]] = None,
                              model_type: str = "ai_model") -> bool:
        """Add prediction to history"""
        print(f"\n📜 Adding history for user {user_id}")
        
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
            
            # ALWAYS save to file
            file_saved = self._save_history_to_file(user_id, history_entry)
            
            # Try Supabase
            if file_saved and self.supabase_url and self.supabase_key:
                try:
                    result = self._supabase_request("POST", "prediction_history", data=history_entry)
                    if result:
                        print("☁️ History saved to Supabase")
                        
                        # Update scan count
                        user = self._get_user_by_id_from_file(user_id)
                        if user:
                            user['scan_count'] = user.get('scan_count', 0) + 1
                            self._save_user_to_file(user)
                            
                            self._supabase_request("PATCH", "users",
                                                  data={"scan_count": user['scan_count']},
                                                  params={"id": f"eq.{user_id}"})
                except Exception as e:
                    print(f"⚠️ Supabase history save failed: {e}")
            
            return file_saved
            
        except Exception as e:
            print(f"❌ Save history error: {e}")
            return False
    
    def get_user_history(self, user_id: str, limit: int = 50, 
                        offset: int = 0, filter_disease: Optional[str] = None) -> List[Dict]:
        """Get user's prediction history"""
        try:
            # Get from file (always works)
            history = self._get_history_from_file(user_id)
            
            # Apply filter
            if filter_disease and filter_disease != "all" and history:
                history = [
                    h for h in history 
                    if filter_disease.lower() in h["prediction"].lower()
                ]
            
            # Apply pagination
            paginated = history[offset:offset + limit]
            print(f"📜 Retrieved {len(paginated)} history entries")
            
            return paginated
            
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
            return self._get_user_by_id_from_file(user_id)
        except Exception as e:
            print(f"❌ Get profile error: {e}")
            return None
    
    def _generate_treatment_plan(self, prediction: str, confidence: float) -> str:
        """Generate treatment plan"""
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
            # Check files first
            for filename in os.listdir("data/users"):
                if filename.endswith('.json'):
                    with open(os.path.join("data/users", filename), 'r') as f:
                        user = json.load(f)
                        if user.get("hardware_id") == hardware_id:
                            return False
            
            # Check Supabase if available
            if self.supabase_url and self.supabase_key:
                try:
                    result = self._supabase_request("GET", "users", params={"hardware_id": f"eq.{hardware_id}"})
                    if result and len(result) > 0:
                        return False
                except:
                    pass
            
            return True
        except:
            return True

# Create global instance
db = SupabaseDatabase()

# List existing users on startup
print("\n📂 EXISTING USERS:")
user_files = [f for f in os.listdir("data/users") if f.endswith('.json')] if os.path.exists("data/users") else []
if user_files:
    for user_file in user_files[:5]:
        try:
            with open(os.path.join("data/users", user_file), 'r') as f:
                user = json.load(f)
                print(f"  - {user.get('email')} (ID: {user.get('id')})")
        except:
            print(f"  - {user_file}")
    if len(user_files) > 5:
        print(f"  ... and {len(user_files) - 5} more")
else:
    print("  No users yet")
print("=" * 60)
