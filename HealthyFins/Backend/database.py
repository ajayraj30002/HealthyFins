# database.py - Supabase Integration (COMPLETE VERSION)
import os
from supabase import create_client, Client
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any

class SupabaseDatabase:
    def __init__(self):
        # Get Supabase credentials
        self.supabase_url = os.getenv("SUPABASE_URL", "https://bxfljshwfpgsnfyqemcd.supabase.co")
        self.supabase_key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ4Zmxqc2h3ZnBnc25meXFlbWNkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg0NjYxMDUsImV4cCI6MjA4NDA0MjEwNX0.M8qOkC-ajPfWgxG-PjCfY6UGLSSm5O2jmlQNTfaM3IQ")
        
        # Predefined hardware IDs (only these are allowed)
        self.VALID_HARDWARE_IDS = [
            "FISHMON-001", "FISHMON-002", "FISHMON-003",
            "FISHMON-004", "FISHMON-005", "FISHMON-006",
            "AQUATECH-101", "AQUATECH-102", "AQUATECH-103",
            "HYDROPRO-201", "HYDROPRO-202"
        ]
        
        try:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            print("✅ Connected to Supabase")
        except Exception as e:
            print(f"❌ Supabase connection failed: {e}")
            self.supabase = None
            # Fallback to local JSON storage
            self.local_data = {"users": {}, "history": {}}
    
    # ========== USER MANAGEMENT ==========
    
    def create_user(self, email: str, password: str, name: str, hardware_id: Optional[str] = None) -> tuple:
        """Create a new user with validation"""
        try:
            # Validate hardware_id if provided
            if hardware_id and hardware_id not in self.VALID_HARDWARE_IDS:
                return False, f"Invalid hardware ID. Must be one of: {', '.join(self.VALID_HARDWARE_IDS[:3])}..."
            
            # Check if email already exists
            if self.supabase:
                existing = self.supabase.table("users").select("*").eq("email", email).execute()
                if existing.data:
                    return False, "Email already registered"
            else:
                if email in self.local_data["users"]:
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
            
            # Save to database
            if self.supabase:
                result = self.supabase.table("users").insert(user_data).execute()
            else:
                # Local fallback
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
            return False, f"Registration failed: {str(e)}"
    
    def authenticate_user(self, email: str, password: str) -> tuple:
        """Authenticate user login"""
        try:
            # Hash the provided password
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            # Query database
            if self.supabase:
                result = self.supabase.table("users").select("*").eq("email", email).execute()
                if not result.data:
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
            
            # Update last login
            if self.supabase:
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
            history_entry = {
                "id": f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "image_name": image_name[:100],
                "image_url": image_url,
                "prediction": prediction,
                "confidence": confidence,
                "model_type": model_type,
                "symptoms": symptoms or [],
                "treatment_plan": self._generate_treatment_plan(prediction, confidence),
                "is_urgent": confidence > 70 and "healthy" not in prediction.lower()
            }
            
            # Save to database
            if self.supabase:
                # Add to history table
                self.supabase.table("prediction_history").insert(history_entry).execute()
                
                # Update user's scan count
                user = self.supabase.table("users").select("scan_count").eq("id", user_id).execute()
                if user.data:
                    new_count = user.data[0].get("scan_count", 0) + 1
                    self.supabase.table("users").update({
                        "scan_count": new_count
                    }).eq("id", user_id).execute()
            else:
                # Local fallback
                if user_id not in self.local_data["history"]:
                    self.local_data["history"][user_id] = []
                self.local_data["history"][user_id].insert(0, history_entry)
            
            print(f"✅ History saved for user {user_id}: {prediction} ({confidence}%)")
            return True
            
        except Exception as e:
            print(f"❌ Save history error: {e}")
            return False
    
    def get_user_history(self, user_id: str, limit: int = 50, 
                        offset: int = 0, filter_disease: Optional[str] = None) -> List[Dict]:
        """Get user's prediction history with optional filtering"""
        try:
            if self.supabase:
                query = self.supabase.table("prediction_history").select("*").eq("user_id", user_id)
                
                if filter_disease and filter_disease != "all":
                    query = query.eq("prediction", filter_disease)
                
                query = query.order("timestamp", desc=True).range(offset, offset + limit - 1)
                result = query.execute()
                return result.data
            else:
                # Local fallback
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
        """Get statistics about user's history"""
        try:
            if self.supabase:
                # Get total count
                total = self.supabase.table("prediction_history").select("*", count="exact").eq("user_id", user_id).execute()
                
                # Get by disease type
                diseases = self.supabase.table("prediction_history").select("prediction").eq("user_id", user_id).execute()
                
                # Count healthy vs disease
                healthy_count = 0
                disease_count = 0
                disease_types = {}
                
                for entry in diseases.data:
                    pred = entry["prediction"].lower()
                    if "healthy" in pred:
                        healthy_count += 1
                    else:
                        disease_count += 1
                        disease_types[pred] = disease_types.get(pred, 0) + 1
                
                return {
                    "total": total.count,
                    "healthy": healthy_count,
                    "disease": disease_count,
                    "disease_types": disease_types,
                    "last_scan": self.get_last_scan(user_id)
                }
            else:
                # Local fallback
                if user_id not in self.local_data["history"]:
                    return {"total": 0, "healthy": 0, "disease": 0, "disease_types": {}, "last_scan": None}
                
                history = self.local_data["history"][user_id]
                healthy = sum(1 for h in history if "healthy" in h["prediction"].lower())
                disease = len(history) - healthy
                
                disease_types = {}
                for entry in history:
                    if "healthy" not in entry["prediction"].lower():
                        pred = entry["prediction"]
                        disease_types[pred] = disease_types.get(pred, 0) + 1
                
                return {
                    "total": len(history),
                    "healthy": healthy,
                    "disease": disease,
                    "disease_types": disease_types,
                    "last_scan": history[0]["timestamp"] if history else None
                }
                
        except Exception as e:
            print(f"❌ Get stats error: {e}")
            return {"total": 0, "healthy": 0, "disease": 0, "disease_types": {}, "last_scan": None}
    
    # ========== USER PROFILE ==========
    
    def update_user_profile(self, user_id: str, name: Optional[str] = None,
                           hardware_id: Optional[str] = None) -> tuple:
        """Update user profile information"""
        try:
            # Validate hardware_id if provided
            if hardware_id and hardware_id not in self.VALID_HARDWARE_IDS:
                return False, f"Invalid hardware ID. Must be one of: {', '.join(self.VALID_HARDWARE_IDS[:3])}..."
            
            update_data = {}
            if name:
                update_data["name"] = name
            if hardware_id:
                update_data["hardware_id"] = hardware_id
            
            if not update_data:
                return False, "No data to update"
            
            if self.supabase:
                result = self.supabase.table("users").update(update_data).eq("id", user_id).execute()
                if not result.data:
                    return False, "User not found"
            else:
                # Find user by ID in local data
                user_found = False
                for email, user in self.local_data["users"].items():
                    if user["id"] == user_id:
                        if name:
                            user["name"] = name
                        if hardware_id:
                            user["hardware_id"] = hardware_id
                        user_found = True
                        break
                
                if not user_found:
                    return False, "User not found"
            
            return True, "Profile updated successfully"
            
        except Exception as e:
            print(f"❌ Update profile error: {e}")
            return False, f"Update failed: {str(e)}"
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile by ID"""
        try:
            if self.supabase:
                result = self.supabase.table("users").select("*").eq("id", user_id).execute()
                if result.data:
                    user = result.data[0]
                    # Remove sensitive data
                    user.pop("password_hash", None)
                    return user
            else:
                # Find user in local data
                for email, user in self.local_data["users"].items():
                    if user["id"] == user_id:
                        user_copy = user.copy()
                        user_copy.pop("password_hash", None)
                        return user_copy
                        
            return None
            
        except Exception as e:
            print(f"❌ Get profile error: {e}")
            return None
    
    # ========== HELPER METHODS ==========
    
    def _generate_treatment_plan(self, prediction: str, confidence: float) -> str:
        """Generate treatment plan based on prediction"""
        plans = {
            "healthy": "Maintain current water conditions. Regular monitoring recommended.",
            "bacterial": "Antibiotic treatment recommended. Isolate affected fish.",
            "fungal": "Antifungal medication required. Improve water quality.",
            "parasitic": "Anti-parasitic treatment needed. Quarantine tank recommended.",
            "viral": "Supportive care. No specific treatment available. Focus on stress reduction."
        }
        
        pred_lower = prediction.lower()
        for key, plan in plans.items():
            if key in pred_lower:
                return plan
        
        return "Consult with aquatic veterinarian for proper diagnosis and treatment."
    
    def get_last_scan(self, user_id: str) -> Optional[str]:
        """Get timestamp of last scan"""
        try:
            if self.supabase:
                result = self.supabase.table("prediction_history").select("timestamp").eq("user_id", user_id).order("timestamp", desc=True).limit(1).execute()
                return result.data[0]["timestamp"] if result.data else None
            else:
                if user_id in self.local_data["history"] and self.local_data["history"][user_id]:
                    return self.local_data["history"][user_id][0]["timestamp"]
                return None
        except:
            return None
    
    def search_history(self, user_id: str, query: str, limit: int = 20) -> List[Dict]:
        """Search in user's history"""
        try:
            if self.supabase:
                # Search in prediction field
                result = self.supabase.table("prediction_history").select("*").eq("user_id", user_id).ilike("prediction", f"%{query}%").order("timestamp", desc=True).limit(limit).execute()
                return result.data
            else:
                if user_id not in self.local_data["history"]:
                    return []
                
                query_lower = query.lower()
                return [
                    h for h in self.local_data["history"][user_id]
                    if query_lower in h["prediction"].lower() or 
                       query_lower in str(h.get("symptoms", [""])[0]).lower()
                ][:limit]
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def delete_history_entry(self, user_id: str, entry_id: str) -> bool:
        """Delete a specific history entry"""
        try:
            if self.supabase:
                result = self.supabase.table("prediction_history").delete().eq("id", entry_id).eq("user_id", user_id).execute()
                return len(result.data) > 0
            else:
                if user_id in self.local_data["history"]:
                    self.local_data["history"][user_id] = [
                        h for h in self.local_data["history"][user_id]
                        if h["id"] != entry_id
                    ]
                    return True
                return False
        except Exception as e:
            print(f"❌ Delete entry error: {e}")
            return False
    
    def clear_user_history(self, user_id: str) -> bool:
        """Clear all history for a user"""
        try:
            if self.supabase:
                self.supabase.table("prediction_history").delete().eq("user_id", user_id).execute()
                # Reset scan count
                self.supabase.table("users").update({"scan_count": 0}).eq("id", user_id).execute()
            else:
                self.local_data["history"][user_id] = []
                
            print(f"✅ History cleared for user {user_id}")
            return True
        except Exception as e:
            print(f"❌ Clear history error: {e}")
            return False
    
    def get_hardware_ids(self) -> List[str]:
        """Get list of valid hardware IDs"""
        return self.VALID_HARDWARE_IDS
    
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

