# database.py - COMPLETE WORKING VERSION WITH SUPABASE
import os
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import traceback

# CORRECT IMPORTS for supabase
from supabase_py import create_client, Client
from postgrest.exceptions import APIError

class SupabaseDatabase:
    def __init__(self):
        # Get Supabase credentials
        self.supabase_url = os.getenv("SUPABASE_URL", "https://bxfljshwfpgsnfyqemcd.supabase.co")
        self.supabase_key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ4Zmxqc2h3ZnBnc25meXFlbWNkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg0NjYxMDUsImV4cCI6MjA4NDA0MjEwNX0.M8qOkC-ajPfWgxG-PjCfY6UGLSSm5O2jmlQNTfaM3IQ")
        self._supabase_client = None
        
        # Predefined hardware IDs
        self.VALID_HARDWARE_IDS = [
            "FISHMON-001", "FISHMON-002", "FISHMON-003",
            "FISHMON-004", "FISHMON-005", "FISHMON-006",
            "AQUATECH-101", "AQUATECH-102", "AQUATECH-103",
            "HYDROPRO-201", "HYDROPRO-202"
        ]
        
        # Create backup directories
        os.makedirs("backup/users", exist_ok=True)
        os.makedirs("backup/history", exist_ok=True)
        
        print("=" * 60)
        print("🐟 HEALTHYFINS DATABASE INITIALIZED")
        print(f"🔧 Supabase URL: {'✅ Configured' if self.supabase_url else '❌ Missing'}")
        print(f"🔧 Supabase Key: {'✅ Configured' if self.supabase_key else '❌ Missing'}")
        print(f"📁 Backup folder: {os.path.abspath('backup')}")
        print("=" * 60)
        
        # Try to connect
        if self.supabase_url and self.supabase_key:
            self._init_supabase()
    
    def _init_supabase(self):
        """Initialize Supabase client"""
        try:
            self._supabase_client = create_client(self.supabase_url, self.supabase_key)
            print("✅ Supabase client initialized")
            
            # Test connection
            try:
                test = self._supabase_client.table('users').select('*').limit(1).execute()
                print("✅ Supabase connection test passed")
            except Exception as e:
                print(f"⚠️ Supabase connection test failed: {e}")
                self._supabase_client = None
        except Exception as e:
            print(f"⚠️ Supabase init failed: {e}")
            self._supabase_client = None
    
    @property
    def supabase(self):
        return self._supabase_client
    
    # ========== BACKUP METHODS ==========
    
    def _backup_user(self, user_data: dict) -> bool:
        """Save user to backup file"""
        try:
            safe_email = user_data['email'].replace('@', '_at_').replace('.', '_dot_')
            file_path = f"backup/users/{safe_email}.json"
            with open(file_path, 'w') as f:
                json.dump(user_data, f, indent=2, default=str)
            return True
        except:
            return False
    
    def _get_user_from_backup(self, email: str) -> Optional[Dict]:
        """Get user from backup"""
        try:
            safe_email = email.replace('@', '_at_').replace('.', '_dot_')
            file_path = f"backup/users/{safe_email}.json"
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
        except:
            pass
        return None
    
    def _backup_history(self, user_id: str, history: List) -> bool:
        """Save history to backup"""
        try:
            file_path = f"backup/history/{user_id}.json"
            with open(file_path, 'w') as f:
                json.dump(history, f, indent=2, default=str)
            return True
        except:
            return False
    
    # ========== USER MANAGEMENT ==========
    
    def create_user(self, email: str, password: str, name: str, hardware_id: Optional[str] = None) -> tuple:
        """Create a new user with validation"""
        print(f"\n📝 Creating user: {email}")
        
        try:
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
            
            # ALWAYS backup to file first
            self._backup_user(user_data)
            print("✅ User backed up to file")
            
            # Try Supabase
            supabase_success = False
            if self.supabase:
                try:
                    # Check if user exists
                    existing = self.supabase.table('users').select('*').eq('email', email).execute()
                    if existing.data:
                        return False, "Email already registered"
                    
                    # Check hardware_id
                    if hardware_id:
                        hw_check = self.supabase.table('users').select('*').eq('hardware_id', hardware_id).execute()
                        if hw_check.data:
                            return False, "Hardware ID already registered"
                    
                    # Insert user
                    result = self.supabase.table('users').insert(user_data).execute()
                    if result.data:
                        supabase_success = True
                        print("✅ User saved to Supabase")
                except Exception as e:
                    print(f"⚠️ Supabase save failed: {e}")
            
            # Return success (we have backup)
            print(f"✅ User created: {email} (ID: {user_id})")
            return True, {
                "user_id": user_id,
                "email": email,
                "name": name,
                "hardware_id": hardware_id,
                "created_at": user_data["created_at"]
            }
            
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
            
            # Try Supabase first
            if self.supabase:
                try:
                    result = self.supabase.table('users').select('*').eq('email', email).execute()
                    if result.data:
                        user = result.data[0]
                        source = "Supabase"
                        print(f"✅ User found in {source}")
                except Exception as e:
                    print(f"⚠️ Supabase query failed: {e}")
            
            # Try backup if Supabase failed
            if not user:
                user = self._get_user_from_backup(email)
                if user:
                    source = "backup"
                    print(f"✅ User found in {source}")
            
            if not user:
                print("❌ User not found")
                return False, "Invalid email or password"
            
            # Verify password
            if user.get("password_hash") != password_hash:
                print("❌ Invalid password")
                return False, "Invalid email or password"
            
            # Update last login
            user["last_login"] = datetime.now().isoformat()
            self._backup_user(user)
            
            if self.supabase:
                try:
                    self.supabase.table('users').update({"last_login": user["last_login"]}).eq('id', user['id']).execute()
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
            
            # Get existing history
            history = self.get_user_history(user_id)
            history.insert(0, history_entry)
            
            # Backup to file
            self._backup_history(user_id, history)
            
            # Try Supabase
            if self.supabase:
                try:
                    self.supabase.table('prediction_history').insert(history_entry).execute()
                    
                    # Update scan count
                    self.supabase.table('users').update({
                        'scan_count': self.supabase.table('users').select('scan_count').eq('id', user_id).execute().data[0].get('scan_count', 0) + 1
                    }).eq('id', user_id).execute()
                except Exception as e:
                    print(f"⚠️ Supabase history save failed: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Save history error: {e}")
            return False
    
    def get_user_history(self, user_id: str, limit: int = 50, 
                        offset: int = 0, filter_disease: Optional[str] = None) -> List[Dict]:
        """Get user's prediction history"""
        try:
            history = []
            
            # Try Supabase first
            if self.supabase:
                try:
                    query = self.supabase.table('prediction_history').select('*').eq('user_id', user_id)
                    
                    if filter_disease and filter_disease != "all":
                        if filter_disease == "healthy":
                            query = query.ilike('prediction', '%healthy%')
                        else:
                            query = query.ilike('prediction', f'%{filter_disease}%')
                    
                    result = query.order('timestamp', desc=True).range(offset, offset + limit - 1).execute()
                    if result.data:
                        history = result.data
                except Exception as e:
                    print(f"⚠️ Supabase history fetch failed: {e}")
            
            # Try backup if Supabase failed
            if not history:
                try:
                    file_path = f"backup/history/{user_id}.json"
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as f:
                            all_history = json.load(f)
                            
                            if filter_disease and filter_disease != "all":
                                all_history = [
                                    h for h in all_history 
                                    if filter_disease.lower() in h['prediction'].lower()
                                ]
                            
                            history = all_history[offset:offset + limit]
                except Exception as e:
                    print(f"⚠️ Backup history fetch failed: {e}")
            
            return history
            
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
            # Try Supabase
            if self.supabase:
                try:
                    result = self.supabase.table('users').select('*').eq('id', user_id).execute()
                    if result.data:
                        user = result.data[0]
                        if "password_hash" in user:
                            del user["password_hash"]
                        return user
                except:
                    pass
            
            # Try backup
            for filename in os.listdir("backup/users"):
                if filename.endswith('.json'):
                    with open(os.path.join("backup/users", filename), 'r') as f:
                        user = json.load(f)
                        if user.get("id") == user_id:
                            if "password_hash" in user:
                                del user["password_hash"]
                            return user
            
            return None
            
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
            if self.supabase:
                result = self.supabase.table('users').select('hardware_id').eq('hardware_id', hardware_id).execute()
                if result.data:
                    return False
            
            # Check backups too
            for filename in os.listdir("backup/users"):
                if filename.endswith('.json'):
                    with open(os.path.join("backup/users", filename), 'r') as f:
                        user = json.load(f)
                        if user.get("hardware_id") == hardware_id:
                            return False
            
            return True
        except:
            return True

# Create global instance
db = SupabaseDatabase()

# Display status
print("\n📊 DATABASE STATUS:")
print(f"🔌 Supabase: {'🟢 Connected' if db.supabase else '🔴 Disconnected'}")
print(f"💾 Backup folder: {'🟢 Ready'}")
print("=" * 60)
