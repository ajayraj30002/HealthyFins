# database.py - OPTIMIZED FOR EFFICIENT STORAGE
import os
from supabase import create_client, Client
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import traceback
import uuid

class SupabaseDatabase:
    def __init__(self):
        self.supabase_url = os.getenv("https://bxfljshwfpgsnfyqemcd.supabase.co")
        self.supabase_key = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ4Zmxqc2h3ZnBnc25meXFlbWNkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg0NjYxMDUsImV4cCI6MjA4NDA0MjEwNX0.M8qOkC-ajPfWgxG-PjCfY6UGLSSm5O2jmlQNTfaM3IQ")
        
        self.VALID_HARDWARE_IDS = [
            "FISHMON-001", "FISHMON-002", "FISHMON-003",
            "FISHMON-004", "FISHMON-005", "FISHMON-006",
            "AQUATECH-101", "AQUATECH-102", "AQUATECH-103",
            "HYDROPRO-201", "HYDROPRO-202"
        ]
        
        self.supabase = None
        self.local_data = {"users": {}, "history": {}}
        
        # Cache for frequently accessed data (lasts 1 hour)
        self.cache = {
            "users": {},
            "hardware_availability": {},
            "stats": {}
        }
        self.cache_timestamp = {}
        
        try:
            if self.supabase_url and self.supabase_key:
                self.supabase = create_client(self.supabase_url, self.supabase_key)
                print("✅ Connected to Supabase")
        except Exception as e:
            print(f"❌ Supabase connection failed: {e}")
    
    # ========== CACHE MANAGEMENT ==========
    def _get_from_cache(self, key: str, cache_type: str, max_age_minutes: int = 60):
        """Get data from cache if not expired"""
        cache_key = f"{cache_type}:{key}"
        if cache_key in self.cache_timestamp:
            age = datetime.now() - self.cache_timestamp[cache_key]
            if age < timedelta(minutes=max_age_minutes):
                return self.cache.get(cache_key)
        return None
    
    def _set_in_cache(self, key: str, cache_type: str, data: Any):
        """Store data in cache"""
        cache_key = f"{cache_type}:{key}"
        self.cache[cache_key] = data
        self.cache_timestamp[cache_key] = datetime.now()
    
    def _clear_cache(self, cache_type: Optional[str] = None):
        """Clear cache"""
        if cache_type:
            keys_to_delete = [k for k in self.cache.keys() if k.startswith(f"{cache_type}:")]
            for key in keys_to_delete:
                self.cache.pop(key, None)
                self.cache_timestamp.pop(key, None)
        else:
            self.cache.clear()
            self.cache_timestamp.clear()
    
    # ========== USER MANAGEMENT ==========
    def create_user(self, email: str, password: str, name: str, hardware_id: Optional[str] = None) -> tuple:
        """Create a new user with validation"""
        try:
            # Validate hardware_id if provided
            if hardware_id and hardware_id not in self.VALID_HARDWARE_IDS:
                return False, f"Invalid hardware ID"
            
            # Check if hardware_id is already in use
            if hardware_id and self.supabase:
                existing_hw = self.supabase.table("users").select("id").eq("hardware_id", hardware_id).execute()
                if existing_hw.data and len(existing_hw.data) > 0:
                    return False, "Hardware ID already registered"
            
            # Hash password
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            # Generate UUID
            user_id = str(uuid.uuid4())
            
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
            
            if self.supabase:
                # Check if email already exists
                existing = self.supabase.table("users").select("id").eq("email", email).execute()
                if existing.data and len(existing.data) > 0:
                    return False, "Email already registered"
                
                # Insert new user
                result = self.supabase.table("users").insert(user_data).execute()
                
                if not result.data:
                    return False, "Failed to create user"
                
                # Clear cache for this email
                self._clear_cache("user")
                
                print(f"✅ User created: {email}")
            else:
                # Local fallback
                if email in self.local_data["users"]:
                    return False, "Email already registered"
                self.local_data["users"][email] = user_data
                self.local_data["history"][user_id] = []
            
            return True, {
                "user_id": user_id,
                "email": email,
                "name": name,
                "hardware_id": hardware_id,
                "created_at": user_data["created_at"]
            }
            
        except Exception as e:
            print(f"❌ Create user error: {e}")
            return False, f"Registration failed"
    
    def authenticate_user(self, email: str, password: str) -> tuple:
        """Authenticate user login"""
        try:
            # Check cache first
            cached_user = self._get_from_cache(email, "user", 5)  # 5 minute cache for auth
            if cached_user:
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                if cached_user.get("password_hash") == password_hash:
                    return True, cached_user
            
            # Hash the provided password
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            user = None
            
            if self.supabase:
                result = self.supabase.table("users").select("*").eq("email", email).execute()
                if not result.data or len(result.data) == 0:
                    return False, "Invalid email or password"
                
                user = result.data[0]
            else:
                if email not in self.local_data["users"]:
                    return False, "Invalid email or password"
                user = self.local_data["users"][email]
            
            # Verify password
            if user.get("password_hash") != password_hash:
                return False, "Invalid email or password"
            
            # Check if user is active
            if not user.get("is_active", True):
                return False, "Account is deactivated"
            
            # Update last login (async - don't wait for it)
            update_data = {"last_login": datetime.now().isoformat()}
            if self.supabase:
                try:
                    self.supabase.table("users").update(update_data).eq("id", user["id"]).execute()
                except:
                    pass  # Non-critical, continue
            
            user_data = {
                "user_id": user["id"],
                "email": user["email"],
                "name": user["name"],
                "hardware_id": user.get("hardware_id", ""),
                "created_at": user["created_at"],
                "scan_count": user.get("scan_count", 0),
                "last_login": update_data["last_login"]
            }
            
            # Cache the user data
            self._set_in_cache(email, "user", {**user_data, "password_hash": user["password_hash"]})
            
            return True, user_data
            
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False, f"Authentication failed"
    
    # ========== PREDICTION HISTORY ==========
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
                "confidence": round(float(confidence), 2),  # Store rounded
                "model_type": model_type,
                "symptoms": symptoms or [],
                "treatment_plan": self._generate_treatment_plan(prediction, confidence),
                "is_urgent": confidence > 70 and "healthy" not in prediction.lower()
            }
            
            if self.supabase:
                # Batch insert if possible (but single is fine)
                result = self.supabase.table("prediction_history").insert(history_entry).execute()
                
                # Update user's scan count (use RPC for better performance)
                try:
                    self.supabase.rpc('increment_scan_count', {'user_id': user_id}).execute()
                except:
                    # Fallback to direct update
                    user_data = self.supabase.table("users").select("scan_count").eq("id", user_id).execute()
                    if user_data.data:
                        new_count = user_data.data[0].get("scan_count", 0) + 1
                        self.supabase.table("users").update({
                            "scan_count": new_count,
                            "last_scan": history_entry["timestamp"]
                        }).eq("id", user_id).execute()
                
                # Clear stats cache for this user
                self._clear_cache(f"stats:{user_id}")
                
                print(f"✅ History saved: {entry_id}")
            else:
                if user_id not in self.local_data["history"]:
                    self.local_data["history"][user_id] = []
                self.local_data["history"][user_id].insert(0, history_entry)
                # Keep only last 100 entries in local mode
                self.local_data["history"][user_id] = self.local_data["history"][user_id][:100]
            
            return True
            
        except Exception as e:
            print(f"❌ Save history error: {e}")
            return False
    
    def get_user_history(self, user_id: str, limit: int = 50, 
                        offset: int = 0, filter_disease: Optional[str] = None) -> List[Dict]:
        """Get user's prediction history with pagination"""
        try:
            # Check cache for recent history
            cache_key = f"{user_id}:{limit}:{offset}:{filter_disease}"
            cached = self._get_from_cache(cache_key, "history", 1)  # 1 minute cache
            if cached:
                return cached
            
            if self.supabase:
                query = self.supabase.table("prediction_history").select("*").eq("user_id", user_id)
                
                if filter_disease and filter_disease != "all":
                    query = query.ilike("prediction", f"%{filter_disease}%")
                
                # Get paginated results
                result = query.order("timestamp", desc=True).range(offset, offset + limit - 1).execute()
                
                history = result.data if result.data else []
                
                # Cache the result
                self._set_in_cache(cache_key, "history", history)
                
                return history
            else:
                if user_id not in self.local_data["history"]:
                    return []
                
                history = self.local_data["history"][user_id]
                if filter_disease and filter_disease != "all":
                    history = [h for h in history if filter_disease.lower() in h["prediction"].lower()]
                
                return history[offset:offset + limit]
                
        except Exception as e:
            print(f"❌ Get history error: {e}")
            return []
    
    def get_history_stats(self, user_id: str) -> Dict:
        """Get cached statistics about user's history"""
        try:
            # Check cache first (5 minute cache for stats)
            cache_key = f"stats:{user_id}"
            cached = self._get_from_cache(cache_key, "stats", 5)
            if cached:
                return cached
            
            if self.supabase:
                # Use Supabase's aggregate functions for better performance
                # Get total count
                count_result = self.supabase.table("prediction_history")\
                    .select("*", count="exact")\
                    .eq("user_id", user_id)\
                    .execute()
                
                total_count = count_result.count if hasattr(count_result, 'count') else 0
                
                if total_count == 0:
                    stats = {"total": 0, "healthy": 0, "disease": 0, "disease_types": {}, "last_scan": None, "avg_confidence": 0}
                    self._set_in_cache(cache_key, "stats", stats)
                    return stats
                
                # Get healthy count
                healthy_result = self.supabase.table("prediction_history")\
                    .select("*", count="exact")\
                    .eq("user_id", user_id)\
                    .ilike("prediction", "%healthy%")\
                    .execute()
                
                healthy_count = healthy_result.count if hasattr(healthy_result, 'count') else 0
                
                # Get last scan
                last_scan_result = self.supabase.table("prediction_history")\
                    .select("timestamp")\
                    .eq("user_id", user_id)\
                    .order("timestamp", desc=True)\
                    .limit(1)\
                    .execute()
                
                last_scan = last_scan_result.data[0]["timestamp"] if last_scan_result.data else None
                
                # Get average confidence
                avg_result = self.supabase.table("prediction_history")\
                    .select("confidence")\
                    .eq("user_id", user_id)\
                    .execute()
                
                avg_confidence = 0
                if avg_result.data:
                    confidences = [entry["confidence"] for entry in avg_result.data]
                    avg_confidence = sum(confidences) / len(confidences)
                
                # Get disease types (simplified - get recent 100 and analyze)
                recent = self.supabase.table("prediction_history")\
                    .select("prediction")\
                    .eq("user_id", user_id)\
                    .limit(100)\
                    .execute()
                
                disease_types = {}
                for entry in recent.data:
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
                
                stats = {
                    "total": total_count,
                    "healthy": healthy_count,
                    "disease": total_count - healthy_count,
                    "disease_types": disease_types,
                    "last_scan": last_scan,
                    "avg_confidence": round(avg_confidence, 2)
                }
                
                # Cache the stats
                self._set_in_cache(cache_key, "stats", stats)
                
                return stats
            else:
                # Local fallback (keep as is)
                if user_id not in self.local_data["history"]:
                    return {"total": 0, "healthy": 0, "disease": 0, "disease_types": {}, "last_scan": None, "avg_confidence": 0}
                
                history = self.local_data["history"][user_id]
                healthy = sum(1 for h in history if "healthy" in h["prediction"].lower())
                
                stats = {
                    "total": len(history),
                    "healthy": healthy,
                    "disease": len(history) - healthy,
                    "disease_types": {},
                    "last_scan": history[0]["timestamp"] if history else None,
                    "avg_confidence": sum(h.get("confidence", 0) for h in history) / len(history) if history else 0
                }
                
                return stats
                
        except Exception as e:
            print(f"❌ Get stats error: {e}")
            return {"total": 0, "healthy": 0, "disease": 0, "disease_types": {}, "last_scan": None, "avg_confidence": 0}
