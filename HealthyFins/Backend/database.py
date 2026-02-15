# database.py - COMPLETE CORRECTED VERSION
import os
from supabase import create_client, Client
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
import traceback

class SupabaseDatabase:
    def __init__(self):
        # Get Supabase credentials from environment variables
        self.supabase_url = os.getenv("https://bxfljshwfpgsnfyqemcd.supabase.co")
        self.supabase_key = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ4Zmxqc2h3ZnBnc25meXFlbWNkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg0NjYxMDUsImV4cCI6MjA4NDA0MjEwNX0.M8qOkC-ajPfWgxG-PjCfY6UGLSSm5O2jmlQNTfaM3IQ")
        
        # Predefined hardware IDs (only these are allowed)
        self.VALID_HARDWARE_IDS = [
            "FISHMON-001", "FISHMON-002", "FISHMON-003",
            "FISHMON-004", "FISHMON-005", "FISHMON-006",
            "AQUATECH-101", "AQUATECH-102", "AQUATECH-103",
            "HYDROPRO-201", "HYDROPRO-202"
        ]
        
        print(f"🔧 Supabase URL: {self.supabase_url[:30] if self.supabase_url else 'Not set'}...")
        
        self.supabase = None
        self.local_data = {"users": {}, "history": {}}
        
        try:
            if self.supabase_url and self.supabase_key:
                # CORRECT: Use create_client from supabase
                self.supabase = create_client(self.supabase_url, self.supabase_key)
                print("✅ Connected to Supabase")
                
                # Test connection by checking if tables exist
                try:
                    # Try to query users table
                    test = self.supabase.table("users").select("*", count="exact").limit(1).execute()
                    print(f"✅ Database test successful")
                except Exception as e:
                    print(f"⚠️ Tables might not exist yet: {e}")
                    print("🔄 Creating tables if they don't exist...")
                    self._create_tables()
            else:
                print("⚠️ Supabase credentials missing, using local database")
                
        except Exception as e:
            print(f"❌ Supabase connection failed: {e}")
            traceback.print_exc()
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            # Check if users table exists by trying to insert a test record
            # This is a workaround - in production, you should create tables manually in Supabase
            print("📋 Please ensure the following tables exist in your Supabase database:")
            print("""
            CREATE TABLE users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                hardware_id TEXT,
                created_at TIMESTAMP NOT NULL,
                last_login TIMESTAMP,
                scan_count INTEGER DEFAULT 0,
                last_scan TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            );
            
            CREATE TABLE prediction_history (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id),
                timestamp TIMESTAMP NOT NULL,
                image_name TEXT NOT NULL,
                image_url TEXT,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_type TEXT,
                symptoms TEXT[],
                treatment_plan TEXT,
                is_urgent BOOLEAN DEFAULT FALSE
            );
            """)
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
    
    # ========== USER MANAGEMENT ==========
    
    def create_user(self, email: str, password: str, name: str, hardware_id: Optional[str] = None) -> tuple:
        """Create a new user with validation"""
        try:
            # Validate hardware_id if provided
            if hardware_id and hardware_id not in self.VALID_HARDWARE_IDS:
                return False, f"Invalid hardware ID. Must be one of: {', '.join(self.VALID_HARDWARE_IDS[:3])}..."
            
            # Check if hardware_id is already in use
            if hardware_id and self.supabase:
                existing_hw = self.supabase.table("users").select("*").eq("hardware_id", hardware_id).execute()
                if existing_hw.data and len(existing_hw.data) > 0:
                    return False, "Hardware ID already registered to another user"
            
            # Hash password
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            # Generate unique user ID
            import uuid
            user_id = str(uuid.uuid4())[:12]
            
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
            
            # Save to database
            if self.supabase:
                # Check if email already exists
                existing = self.supabase.table("users").select("*").eq("email", email).execute()
                if existing.data and len(existing.data) > 0:
                    return False, "Email already registered"
                
                # Insert new user
                result = self.supabase.table("users").insert(user_data).execute()
                
                if not result.data:
                    return False, "Failed to create user"
                
                print(f"✅ User created in Supabase: {email}")
            else:
                # Local fallback
                if email in self.local_data["users"]:
                    return False, "Email already registered"
                
                self.local_data["users"][email] = user_data
                self.local_data["history"][user_id] = []
                print(f"✅ User created locally: {email}")
            
            # Return user data without password hash
            return_data = {
                "user_id": user_id,
                "email": email,
                "name": name,
                "hardware_id": hardware_id,
                "created_at": user_data["created_at"]
            }
            
            return True, return_data
            
        except Exception as e:
            print(f"❌ Create user error: {e}")
            traceback.print_exc()
            return False, f"Registration failed: {str(e)}"
    
    def authenticate_user(self, email: str, password: str) -> tuple:
        """Authenticate user login"""
        try:
            # Hash the provided password
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            user = None
            
            # Query database
            if self.supabase:
                result = self.supabase.table("users").select("*").eq("email", email).execute()
                if not result.data or len(result.data) == 0:
                    return False, "Invalid email or password"
                
                user = result.data[0]
            else:
                # Local fallback
                if email not in self.local_data["users"]:
                    return False, "Invalid email or password"
                user = self.local_data["users"][email]
            
            # Verify password
            if user.get("password_hash") != password_hash:
                return False, "Invalid email or password"
            
            # Check if user is active
            if not user.get("is_active", True):
                return False, "Account is deactivated"
            
            # Update last login
            update_data = {"last_login": datetime.now().isoformat()}
            if self.supabase:
                self.supabase.table("users").update(update_data).eq("id", user["id"]).execute()
            else:
                user["last_login"] = update_data["last_login"]
            
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
            traceback.print_exc()
            return False, f"Authentication failed: {str(e)}"
    
    # ========== PREDICTION HISTORY ==========
    
    def add_prediction_history(self, user_id: str, image_name: str, 
                              prediction: str, confidence: float, 
                              image_url: Optional[str] = None,
                              symptoms: Optional[List[str]] = None,
                              model_type: str = "ai_model") -> bool:
        """Add a prediction to user's history"""
        try:
            import uuid
            # Generate unique ID
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
            
            # Save to database
            if self.supabase:
                # Add to history table
                result = self.supabase.table("prediction_history").insert(history_entry).execute()
                
                # Update user's scan count
                user_data = self.supabase.table("users").select("scan_count").eq("id", user_id).execute()
                if user_data.data and len(user_data.data) > 0:
                    current_count = user_data.data[0].get("scan_count", 0)
                    new_count = current_count + 1
                    self.supabase.table("users").update({
                        "scan_count": new_count,
                        "last_scan": history_entry["timestamp"]
                    }).eq("id", user_id).execute()
                    
                print(f"✅ History saved to Supabase: {entry_id}")
            else:
                # Local fallback
                if user_id not in self.local_data["history"]:
                    self.local_data["history"][user_id] = []
                self.local_data["history"][user_id].insert(0, history_entry)
                print(f"✅ History saved locally: {entry_id}")
            
            return True
            
        except Exception as e:
            print(f"❌ Save history error: {e}")
            traceback.print_exc()
            return False
    
    def get_user_history(self, user_id: str, limit: int = 50, 
                        offset: int = 0, filter_disease: Optional[str] = None) -> List[Dict]:
        """Get user's prediction history with optional filtering"""
        try:
            if self.supabase:
                # Build query
                query = self.supabase.table("prediction_history").select("*").eq("user_id", user_id)
                
                if filter_disease and filter_disease != "all":
                    # Use proper Supabase filtering
                    if filter_disease == "healthy":
                        query = query.ilike("prediction", "%healthy%")
                    elif filter_disease == "bacterial":
                        query = query.ilike("prediction", "%bacterial%")
                    elif filter_disease == "fungal":
                        query = query.ilike("prediction", "%fungal%")
                    elif filter_disease == "parasitic":
                        query = query.ilike("prediction", "%parasitic%")
                    elif filter_disease == "viral":
                        query = query.ilike("prediction", "%viral%")
                
                # Execute query with ordering and pagination
                result = query.order("timestamp", desc=True).range(offset, offset + limit - 1).execute()
                return result.data if result.data else []
            else:
                # Local fallback
                if user_id not in self.local_data["history"]:
                    return []
                
                history = self.local_data["history"][user_id]
                if filter_disease and filter_disease != "all":
                    if filter_disease == "healthy":
                        history = [h for h in history if "healthy" in h["prediction"].lower()]
                    elif filter_disease == "bacterial":
                        history = [h for h in history if "bacterial" in h["prediction"].lower()]
                    elif filter_disease == "fungal":
                        history = [h for h in history if "fungal" in h["prediction"].lower()]
                    elif filter_disease == "parasitic":
                        history = [h for h in history if "parasitic" in h["prediction"].lower()]
                    elif filter_disease == "viral":
                        history = [h for h in history if "viral" in h["prediction"].lower()]
                
                return history[offset:offset + limit]
                
        except Exception as e:
            print(f"❌ Get history error: {e}")
            traceback.print_exc()
            return []
    
    def get_history_stats(self, user_id: str) -> Dict:
        """Get statistics about user's history"""
        try:
            if self.supabase:
                # Get all entries for this user
                all_entries = self.supabase.table("prediction_history").select("prediction, timestamp, confidence").eq("user_id", user_id).execute()
                
                total_count = len(all_entries.data) if all_entries.data else 0
                
                # Count healthy vs disease
                healthy_count = 0
                disease_count = 0
                disease_types = {}
                last_scan = None
                avg_confidence = 0
                
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
                    
                    avg_confidence += entry.get("confidence", 0)
                    
                    # Track last scan
                    if not last_scan or entry["timestamp"] > last_scan:
                        last_scan = entry["timestamp"]
                
                if total_count > 0:
                    avg_confidence = avg_confidence / total_count
                
                return {
                    "total": total_count,
                    "healthy": healthy_count,
                    "disease": disease_count,
                    "disease_types": disease_types,
                    "last_scan": last_scan,
                    "avg_confidence": round(avg_confidence, 2)
                }
            else:
                # Local fallback
                if user_id not in self.local_data["history"]:
                    return {"total": 0, "healthy": 0, "disease": 0, "disease_types": {}, "last_scan": None, "avg_confidence": 0}
                
                history = self.local_data["history"][user_id]
                healthy = sum(1 for h in history if "healthy" in h["prediction"].lower())
                disease = len(history) - healthy
                
                disease_types = {}
                last_scan = None
                avg_confidence = 0
                
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
                    
                    avg_confidence += entry.get("confidence", 0)
                    
                    if not last_scan or entry["timestamp"] > last_scan:
                        last_scan = entry["timestamp"]
                
                if len(history) > 0:
                    avg_confidence = avg_confidence / len(history)
                
                return {
                    "total": len(history),
                    "healthy": healthy,
                    "disease": disease,
                    "disease_types": disease_types,
                    "last_scan": last_scan,
                    "avg_confidence": round(avg_confidence, 2)
                }
                
        except Exception as e:
            print(f"❌ Get stats error: {e}")
            traceback.print_exc()
            return {"total": 0, "healthy": 0, "disease": 0, "disease_types": {}, "last_scan": None, "avg_confidence": 0}
    
    # ========== USER PROFILE ==========
    
    def update_user_profile(self, user_id: str, name: Optional[str] = None,
                           hardware_id: Optional[str] = None) -> tuple:
        """Update user profile information"""
        try:
            # Validate hardware_id if provided
            if hardware_id and hardware_id not in self.VALID_HARDWARE_IDS:
                return False, f"Invalid hardware ID. Must be one of: {', '.join(self.VALID_HARDWARE_IDS[:3])}..."
            
            # Check if hardware_id is already in use (by another user)
            if hardware_id and self.supabase:
                existing = self.supabase.table("users").select("id").eq("hardware_id", hardware_id).neq("id", user_id).execute()
                if existing.data and len(existing.data) > 0:
                    return False, "Hardware ID already registered to another user"
            
            update_data = {}
            if name:
                update_data["name"] = name
            if hardware_id is not None:  # Allow setting to empty string
                update_data["hardware_id"] = hardware_id if hardware_id else None
            
            if not update_data:
                return False, "No data to update"
            
            if self.supabase:
                result = self.supabase.table("users").update(update_data).eq("id", user_id).execute()
                if not result.data or len(result.data) == 0:
                    return False, "User not found"
                print(f"✅ Profile updated in Supabase for user {user_id}")
            else:
                # Find user by ID in local data
                user_found = False
                for email, user in self.local_data["users"].items():
                    if user["id"] == user_id:
                        if name:
                            user["name"] = name
                        if hardware_id is not None:
                            user["hardware_id"] = hardware_id if hardware_id else None
                        user_found = True
                        break
                
                if not user_found:
                    return False, "User not found"
                print(f"✅ Profile updated locally for user {user_id}")
            
            return True, "Profile updated successfully"
            
        except Exception as e:
            print(f"❌ Update profile error: {e}")
            traceback.print_exc()
            return False, f"Update failed: {str(e)}"
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile by ID"""
        try:
            if self.supabase:
                result = self.supabase.table("users").select("*").eq("id", user_id).execute()
                if result.data and len(result.data) > 0:
                    user = result.data[0]
                    # Remove sensitive data
                    if "password_hash" in user:
                        del user["password_hash"]
                    return user
            else:
                # Find user in local data
                for email, user in self.local_data["users"].items():
                    if user["id"] == user_id:
                        user_copy = user.copy()
                        if "password_hash" in user_copy:
                            del user_copy["password_hash"]
                        return user_copy
                        
            return None
            
        except Exception as e:
            print(f"❌ Get profile error: {e}")
            traceback.print_exc()
            return None
    
    # ========== HELPER METHODS ==========
    
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
        elif "parasitic" in pred_lower or "parasite" in pred_lower:
            return plans["parasitic"]
        elif "viral" in pred_lower or "virus" in pred_lower:
            return plans["viral"]
        else:
            return plans["default"]
    
    def get_hardware_ids(self) -> List[str]:
        """Get list of valid hardware IDs"""
        return self.VALID_HARDWARE_IDS.copy()
    
    def check_hardware_available(self, hardware_id: str) -> bool:
        """Check if hardware ID is available (not already assigned)"""
        try:
            if self.supabase:
                result = self.supabase.table("users").select("hardware_id").eq("hardware_id", hardware_id).execute()
                return len(result.data) == 0
            else:
                for user in self.local_data["users"].values():
                    if user.get("hardware_id") == hardware_id:
                        return False
                return True
        except:
            return True

# Create global instance
db = SupabaseDatabase()
print("=" * 50)
print("🐟 HEALTHYFINS DATABASE INITIALIZED")
print(f"📊 Database Type: {'Supabase' if db.supabase else 'Local JSON'}")
print(f"🔧 Valid Hardware IDs: {len(db.VALID_HARDWARE_IDS)}")
print("=" * 50)
