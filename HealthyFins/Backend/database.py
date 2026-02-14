# database.py - BUILD-SAFE VERSION (won't fail during deployment)
import os
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any

# Import supabase lazily (only when needed)
import supabase

class SupabaseDatabase:
    def __init__(self):
        # Store credentials but DON'T connect during __init__
        self.supabase_url = os.getenv("SUPABASE_URL", "")
        self.supabase_key = os.getenv("SUPABASE_KEY", "")
        self._client = None  # Lazy-loaded client
        
        # Predefined hardware IDs
        self.VALID_HARDWARE_IDS = [
            "FISHMON-001", "FISHMON-002", "FISHMON-003",
            "FISHMON-004", "FISHMON-005", "FISHMON-006",
            "AQUATECH-101", "AQUATECH-102", "AQUATECH-103",
            "HYDROPRO-201", "HYDROPRO-202"
        ]
        
        print("=" * 60)
        print("🐟 HEALTHYFINS DATABASE INITIALIZED (lazy connection)")
        print("=" * 60)
    
    @property
    def client(self):
        """Lazy-load the Supabase client only when needed"""
        if self._client is None:
            if not self.supabase_url or not self.supabase_key:
                print("❌ Supabase credentials missing!")
                print("Set SUPABASE_URL and SUPABASE_KEY in environment variables")
                # Don't raise error here - let it fail gracefully at runtime
                return None
            
            try:
                print("🔗 Connecting to Supabase...")
                self._client = supabase.create_client(self.supabase_url, self.supabase_key)
                print("✅ Supabase connection established")
            except Exception as e:
                print(f"❌ Supabase connection failed: {e}")
                self._client = None
        
        return self._client
    
    # ========== USER MANAGEMENT ==========
    
    def create_user(self, email: str, password: str, name: str, hardware_id: Optional[str] = None) -> tuple:
        """Create a new user with validation"""
        try:
            print(f"📝 Attempting to create user: {email}")
            
            client = self.client
            if not client:
                return False, "Database connection not available"
            
            # Validate hardware_id if provided
            if hardware_id and hardware_id not in self.VALID_HARDWARE_IDS:
                return False, f"Invalid hardware ID. Must be one of: {', '.join(self.VALID_HARDWARE_IDS[:3])}..."
            
            # Check if hardware_id is already in use
            if hardware_id:
                existing_hw = client.table("users").select("*").eq("hardware_id", hardware_id).execute()
                if existing_hw.data:
                    return False, "Hardware ID already registered to another user"
            
            # Check if email already exists
            existing = client.table("users").select("*").eq("email", email).execute()
            if existing.data:
                return False, "Email already registered"
            
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
            
            # Save to Supabase
            result = client.table("users").insert(user_data).execute()
            
            if result.data:
                print(f"✅ User created successfully: {email}")
                return True, {
                    "user_id": user_id,
                    "email": email,
                    "name": name,
                    "hardware_id": hardware_id,
                    "created_at": user_data["created_at"]
                }
            else:
                return False, "Failed to save user to database"
            
        except Exception as e:
            print(f"❌ Create user error: {e}")
            return False, f"Registration failed: {str(e)}"
    
    def authenticate_user(self, email: str, password: str) -> tuple:
        """Authenticate user login"""
        try:
            print(f"🔐 Authenticating user: {email}")
            
            client = self.client
            if not client:
                return False, "Database connection not available"
            
            # Hash the provided password
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            # Query Supabase
            result = client.table("users").select("*").eq("email", email).execute()
            
            if not result.data:
                print(f"❌ User not found: {email}")
                return False, "Invalid email or password"
            
            user = result.data[0]
            
            # Verify password
            if user.get("password_hash") != password_hash:
                print(f"❌ Invalid password for: {email}")
                return False, "Invalid email or password"
            
            # Check if user is active
            if not user.get("is_active", True):
                return False, "Account is deactivated"
            
            # Update last login
            update_data = {"last_login": datetime.now().isoformat()}
            client.table("users").update(update_data).eq("id", user["id"]).execute()
            
            return True, {
                "user_id": user["id"],
                "email": user["email"],
                "name": user["name"],
                "hardware_id": user.get("hardware_id", ""),
                "created_at": user["created_at"],
                "scan_count": user.get("scan_count", 0),
                "last_login": update_data["last_login"]
            }
            
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
            client = self.client
            if not client:
                return False
            
            # Generate unique ID
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
            
            # Save to Supabase
            result = client.table("prediction_history").insert(history_entry).execute()
            
            if result.data:
                # Update user's scan count
                user_data = client.table("users").select("scan_count").eq("id", user_id).execute()
                if user_data.data:
                    current_count = user_data.data[0].get("scan_count", 0)
                    new_count = current_count + 1
                    client.table("users").update({
                        "scan_count": new_count,
                        "last_scan": history_entry["timestamp"]
                    }).eq("id", user_id).execute()
                
                return True
            else:
                return False
            
        except Exception as e:
            print(f"❌ Save history error: {e}")
            return False
    
    def get_user_history(self, user_id: str, limit: int = 50, 
                        offset: int = 0, filter_disease: Optional[str] = None) -> List[Dict]:
        """Get user's prediction history"""
        try:
            client = self.client
            if not client:
                return []
            
            # Build query
            query = client.table("prediction_history").select("*").eq("user_id", user_id)
            
            if filter_disease and filter_disease != "all":
                if filter_disease == "healthy":
                    query = query.ilike("prediction", "%healthy%")
                else:
                    query = query.ilike("prediction", f"%{filter_disease}%")
            
            # Execute query with ordering and pagination
            result = query.order("timestamp", desc=True).range(offset, offset + limit - 1).execute()
            
            return result.data if result.data else []
                
        except Exception as e:
            print(f"❌ Get history error: {e}")
            return []
    
    # ========== USER PROFILE ==========
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile by ID"""
        try:
            client = self.client
            if not client:
                return None
            
            result = client.table("users").select("*").eq("id", user_id).execute()
            
            if result.data:
                user = result.data[0]
                # Remove sensitive data
                if "password_hash" in user:
                    del user["password_hash"]
                return user
            
            return None
            
        except Exception as e:
            print(f"❌ Get profile error: {e}")
            return None
    
    def update_user_profile(self, user_id: str, name: Optional[str] = None,
                           hardware_id: Optional[str] = None) -> tuple:
        """Update user profile information"""
        try:
            client = self.client
            if not client:
                return False, "Database connection not available"
            
            # Validate hardware_id if provided
            if hardware_id and hardware_id not in self.VALID_HARDWARE_IDS:
                return False, f"Invalid hardware ID. Must be one of: {', '.join(self.VALID_HARDWARE_IDS[:3])}..."
            
            # Check if hardware_id is already in use (by another user)
            if hardware_id:
                existing = client.table("users").select("id, hardware_id").eq("hardware_id", hardware_id).execute()
                if existing.data and existing.data[0]["id"] != user_id:
                    return False, "Hardware ID already registered to another user"
            
            update_data = {}
            if name:
                update_data["name"] = name
            if hardware_id is not None:
                update_data["hardware_id"] = hardware_id if hardware_id else None
            
            if not update_data:
                return False, "No data to update"
            
            result = client.table("users").update(update_data).eq("id", user_id).execute()
            
            if result.data:
                return True, "Profile updated successfully"
            else:
                return False, "User not found"
            
        except Exception as e:
            print(f"❌ Update profile error: {e}")
            return False, f"Update failed: {str(e)}"
    
    # ========== HELPER METHODS ==========
    
    def _generate_treatment_plan(self, prediction: str, confidence: float) -> str:
        """Generate treatment plan based on prediction"""
        plans = {
            "healthy": "✅ HEALTHY FISH - Maintain current water conditions. Regular monitoring recommended.",
            "bacterial": "🦠 BACTERIAL INFECTION - Immediate action required.\n\n1. Isolate affected fish\n2. Antibiotic treatment\n3. Salt bath\n4. Consult veterinarian",
            "fungal": "🍄 FUNGAL INFECTION - Treatment needed.\n\n1. Antifungal medication\n2. Salt bath\n3. Improve water quality",
            "parasitic": "🐛 PARASITIC INFECTION - Quarantine required.\n\n1. Anti-parasitic medication\n2. Raise temperature\n3. Vacuum substrate",
            "viral": "🦠 VIRAL INFECTION - Supportive care.\n\n1. Maintain optimal water\n2. Add aquarium salt\n3. Reduce stress"
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
    
    def get_history_stats(self, user_id: str) -> Dict:
        """Get statistics about user's history"""
        try:
            client = self.client
            if not client:
                return {"total": 0, "healthy": 0, "disease": 0, "disease_types": {}, "last_scan": None}
            
            # Get all entries for this user
            all_entries = client.table("prediction_history").select("prediction, timestamp").eq("user_id", user_id).execute()
            
            total_count = len(all_entries.data)
            healthy_count = 0
            disease_count = 0
            disease_types = {}
            last_scan = None
            
            for entry in all_entries.data:
                pred = entry["prediction"].lower()
                if "healthy" in pred:
                    healthy_count += 1
                else:
                    disease_count += 1
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
                
                if not last_scan or entry["timestamp"] > last_scan:
                    last_scan = entry["timestamp"]
            
            return {
                "total": total_count,
                "healthy": healthy_count,
                "disease": disease_count,
                "disease_types": disease_types,
                "last_scan": last_scan
            }
                
        except Exception as e:
            print(f"❌ Get stats error: {e}")
            return {"total": 0, "healthy": 0, "disease": 0, "disease_types": {}, "last_scan": None}
    
    def get_hardware_ids(self) -> List[str]:
        """Get list of valid hardware IDs"""
        return self.VALID_HARDWARE_IDS.copy()
    
    def check_hardware_available(self, hardware_id: str) -> bool:
        """Check if hardware ID is available"""
        try:
            client = self.client
            if not client:
                return True
            
            result = client.table("users").select("hardware_id").eq("hardware_id", hardware_id).execute()
            return len(result.data) == 0
        except:
            return True

# Create global instance
db = SupabaseDatabase()
