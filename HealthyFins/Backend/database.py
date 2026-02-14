# database.py - COMPLETE UPDATED VERSION FOR SUPABASE 2.4.0
import os
# CORRECT IMPORT for supabase 2.4.0
import supabase
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any

class SupabaseDatabase:
    def __init__(self):
        # Get Supabase credentials from environment variables
        self.supabase_url = os.getenv("SUPABASE_URL", "https://bxfljshwfpgsnfyqemcd.supabase.co")
        self.supabase_key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ4Zmxqc2h3ZnBnc25meXFlbWNkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg0NjYxMDUsImV4cCI6MjA4NDA0MjEwNX0.M8qOkC-ajPfWgxG-PjCfY6UGLSSm5O2jmlQNTfaM3IQ")
        
        # Predefined hardware IDs (only these are allowed)
        self.VALID_HARDWARE_IDS = [
            "FISHMON-001", "FISHMON-002", "FISHMON-003",
            "FISHMON-004", "FISHMON-005", "FISHMON-006",
            "AQUATECH-101", "AQUATECH-102", "AQUATECH-103",
            "HYDROPRO-201", "HYDROPRO-202"
        ]
        
        print("=" * 60)
        print("🐟 HEALTHYFINS DATABASE INITIALIZATION")
        print("=" * 60)
        
        if not self.supabase_url or not self.supabase_key:
            print("❌ CRITICAL: Supabase credentials missing!")
            print("Please set SUPABASE_URL and SUPABASE_KEY in Render environment variables")
            raise ValueError("Database configuration missing")
        
        try:
            print(f"🔗 Connecting to Supabase...")
            self.supabase = supabase.create_client(self.supabase_url, self.supabase_key)
            
            # Test connection
            test_result = self.supabase.table("users").select("*", count="exact").limit(1).execute()
            print(f"✅ Supabase connection successful")
            print(f"📊 Current users in database: {test_result.count}")
            
        except Exception as e:
            print(f"❌ FATAL: Cannot connect to Supabase: {e}")
            raise e
        
        print("=" * 60)
    
    # ========== USER MANAGEMENT ==========
    
    def create_user(self, email: str, password: str, name: str, hardware_id: Optional[str] = None) -> tuple:
        """Create a new user with validation - ALWAYS saves to Supabase"""
        try:
            print(f"📝 Attempting to create user: {email}")
            
            # Validate hardware_id if provided
            if hardware_id and hardware_id not in self.VALID_HARDWARE_IDS:
                return False, f"Invalid hardware ID. Must be one of: {', '.join(self.VALID_HARDWARE_IDS[:3])}..."
            
            # Check if hardware_id is already in use
            if hardware_id:
                existing_hw = self.supabase.table("users").select("*").eq("hardware_id", hardware_id).execute()
                if existing_hw.data:
                    return False, "Hardware ID already registered to another user"
            
            # Check if email already exists
            existing = self.supabase.table("users").select("*").eq("email", email).execute()
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
            
            # Save to Supabase - THIS IS THE CRITICAL PART
            print(f"💾 Saving user to Supabase: {email}")
            result = self.supabase.table("users").insert(user_data).execute()
            
            if result.data:
                print(f"✅ User created successfully in Supabase: {email}")
                return True, {
                    "user_id": user_id,
                    "email": email,
                    "name": name,
                    "hardware_id": hardware_id,
                    "created_at": user_data["created_at"]
                }
            else:
                print(f"❌ Failed to insert user into Supabase: {email}")
                return False, "Failed to save user to database"
            
        except Exception as e:
            print(f"❌ Create user error: {e}")
            return False, f"Registration failed: {str(e)}"
    
    def authenticate_user(self, email: str, password: str) -> tuple:
        """Authenticate user login - ALWAYS checks Supabase"""
        try:
            print(f"🔐 Authenticating user: {email}")
            
            # Hash the provided password
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            # ALWAYS query Supabase
            result = self.supabase.table("users").select("*").eq("email", email).execute()
            
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
            self.supabase.table("users").update(update_data).eq("id", user["id"]).execute()
            
            print(f"✅ User authenticated: {email}")
            
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
        """Add a prediction to user's history - ALWAYS saves to Supabase"""
        try:
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
            
            # ALWAYS save to Supabase
            print(f"💾 Saving prediction to Supabase for user {user_id}")
            result = self.supabase.table("prediction_history").insert(history_entry).execute()
            
            if result.data:
                # Update user's scan count
                user_data = self.supabase.table("users").select("scan_count").eq("id", user_id).execute()
                if user_data.data:
                    current_count = user_data.data[0].get("scan_count", 0)
                    new_count = current_count + 1
                    self.supabase.table("users").update({
                        "scan_count": new_count,
                        "last_scan": history_entry["timestamp"]
                    }).eq("id", user_id).execute()
                
                print(f"✅ History saved to Supabase: {entry_id}")
                return True
            else:
                print(f"❌ Failed to save history to Supabase")
                return False
            
        except Exception as e:
            print(f"❌ Save history error: {e}")
            return False
    
    def get_user_history(self, user_id: str, limit: int = 50, 
                        offset: int = 0, filter_disease: Optional[str] = None) -> List[Dict]:
        """Get user's prediction history - ALWAYS from Supabase"""
        try:
            print(f"📜 Fetching history for user {user_id}")
            
            # Build query
            query = self.supabase.table("prediction_history").select("*").eq("user_id", user_id)
            
            if filter_disease and filter_disease != "all":
                if filter_disease == "healthy":
                    query = query.ilike("prediction", "%healthy%")
                else:
                    query = query.ilike("prediction", f"%{filter_disease}%")
            
            # Execute query with ordering and pagination
            result = query.order("timestamp", desc=True).range(offset, offset + limit - 1).execute()
            
            print(f"✅ Found {len(result.data)} history entries")
            return result.data if result.data else []
                
        except Exception as e:
            print(f"❌ Get history error: {e}")
            return []
    
    # ========== USER PROFILE ==========
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile by ID - ALWAYS from Supabase"""
        try:
            result = self.supabase.table("users").select("*").eq("id", user_id).execute()
            
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
            # Validate hardware_id if provided
            if hardware_id and hardware_id not in self.VALID_HARDWARE_IDS:
                return False, f"Invalid hardware ID. Must be one of: {', '.join(self.VALID_HARDWARE_IDS[:3])}..."
            
            # Check if hardware_id is already in use (by another user)
            if hardware_id:
                existing = self.supabase.table("users").select("id, hardware_id").eq("hardware_id", hardware_id).execute()
                if existing.data and existing.data[0]["id"] != user_id:
                    return False, "Hardware ID already registered to another user"
            
            update_data = {}
            if name:
                update_data["name"] = name
            if hardware_id is not None:
                update_data["hardware_id"] = hardware_id if hardware_id else None
            
            if not update_data:
                return False, "No data to update"
            
            result = self.supabase.table("users").update(update_data).eq("id", user_id).execute()
            
            if result.data:
                print(f"✅ Profile updated for user {user_id}")
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
            "healthy": "✅ HEALTHY FISH - Maintain current water conditions. Regular monitoring recommended.\n\n1. Continue normal feeding schedule\n2. Weekly 20% water changes\n3. Monitor water parameters (pH 6.5-8.0, temp 24-28°C)\n4. Observe behavior daily",
            "bacterial": "🦠 BACTERIAL INFECTION - Immediate action required.\n\n1. Isolate affected fish immediately\n2. Antibiotic treatment (Kanamycin or Erythromycin)\n3. Salt bath: 1 tbsp per 5 gallons\n4. Increase water temperature to 28°C\n5. Daily 30% water changes\n6. Consult veterinarian if no improvement in 48 hours",
            "fungal": "🍄 FUNGAL INFECTION - Treatment needed.\n\n1. Antifungal medication (Methylene Blue)\n2. Salt bath: 2 tsp per gallon for 30 minutes\n3. Improve water quality immediately\n4. Remove any dead tissue carefully\n5. Increase aeration\n6. Treat for 7-10 days minimum",
            "parasitic": "🐛 PARASITIC INFECTION - Quarantine required.\n\n1. Anti-parasitic medication (Praziquantel)\n2. Formalin bath (follow instructions carefully)\n3. Raise temperature to 30°C gradually\n4. Vacuum substrate thoroughly\n5. Treat all fish in tank\n6. Repeat treatment after 7 days",
            "viral": "🦠 VIRAL INFECTION - Supportive care.\n\n1. No specific treatment available\n2. Maintain optimal water conditions\n3. Add aquarium salt (1 tsp per gallon)\n4. Provide high-quality food\n5. Reduce stress (dim lights, no handling)\n6. Watch for secondary infections"
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
            return "⚠️ UNKNOWN CONDITION - Consult with aquatic veterinarian for proper diagnosis and treatment."
    
    def get_history_stats(self, user_id: str) -> Dict:
        """Get statistics about user's history"""
        try:
            # Get all entries for this user
            all_entries = self.supabase.table("prediction_history").select("prediction, timestamp").eq("user_id", user_id).execute()
            
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
                    # Extract main disease type
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
                
                # Track last scan
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
        """Check if hardware ID is available (not already assigned)"""
        try:
            result = self.supabase.table("users").select("hardware_id").eq("hardware_id", hardware_id).execute()
            return len(result.data) == 0
        except:
            return True
    
    def get_all_users_count(self) -> int:
        """Get total number of users"""
        try:
            result = self.supabase.table("users").select("*", count="exact").execute()
            return result.count
        except:
            return 0

# Create global instance
db = SupabaseDatabase()
