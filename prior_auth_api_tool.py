"""
Prior Authorization API Tool
Simulates API calls to check if medical procedures require prior authorization.
"""

import os
import json
import time
import random
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AuthStatus(str, Enum):
    REQUIRED = "required"
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"

class AuthType(str, Enum):
    SURGERY = "surgery"
    IMAGING = "imaging"
    SPECIALIST = "specialist"
    MEDICATION = "medication"
    THERAPY = "therapy"
    LAB_TEST = "lab_test"
    EMERGENCY = "emergency"

@dataclass
class PriorAuthRequest:
    patient_id: str
    patient_name: str
    provider_id: str
    provider_name: str
    treatment_type: str
    diagnosis_code: str
    diagnosis_description: str
    insurance_id: str
    insurance_type: str
    cost_estimate: float
    urgency: str = "routine"  # routine, urgent, emergency
    clinical_notes: str = ""

@dataclass
class PriorAuthResponse:
    request_id: str
    status: AuthStatus
    auth_required: bool
    auth_number: Optional[str] = None
    expiration_date: Optional[datetime] = None
    approval_date: Optional[datetime] = None
    denial_reason: Optional[str] = None
    required_documents: List[str] = None
    estimated_processing_time: Optional[int] = None  # days
    clinical_criteria: Optional[str] = None
    cost_estimate: Optional[float] = None
    coverage_percentage: Optional[float] = None
    patient_responsibility: Optional[float] = None
    api_response_time: float = 0.0
    error_message: Optional[str] = None

class PriorAuthAPITool:
    """Simulates Prior Authorization API calls with configurable error simulation."""
    
    def __init__(self, induce_prior_auth_error: bool = False):
        self.name = "prior_auth_api_tool"
        self.description = "Check if medical procedures require prior authorization and submit authorization requests"
        self.induce_prior_auth_error = induce_prior_auth_error
        
        # Simulated database of existing authorizations
        self.existing_auths = self._initialize_existing_auths()
        
        # API configuration
        self.base_url = "https://api.insurance.com/prior-auth/v1"
        self.api_key = os.getenv("PRIOR_AUTH_API_KEY", "mock_api_key_12345")
        self.timeout = 30  # seconds
        
        # Error simulation settings
        self.error_probability = 0.05  # 5% chance of random error
        self.timeout_probability = 0.02  # 2% chance of timeout
        
        print(f"Prior Auth API Tool initialized. Error simulation: {self.induce_prior_auth_error}")
    
    def _initialize_existing_auths(self) -> Dict[str, PriorAuthResponse]:
        """Initialize some existing prior authorizations for realistic simulation."""
        existing = {}
        
        # Sample existing authorizations
        sample_auths = [
            {
                "request_id": "PA-2024-001",
                "patient_id": "P12345",
                "status": AuthStatus.APPROVED,
                "auth_number": "AUTH-2024-001",
                "expiration_date": datetime.now() + timedelta(days=30),
                "approval_date": datetime.now() - timedelta(days=5),
                "treatment_type": "surgery",
                "diagnosis": "heart_disease"
            },
            {
                "request_id": "PA-2024-002", 
                "patient_id": "P12346",
                "status": AuthStatus.PENDING,
                "expiration_date": datetime.now() + timedelta(days=15),
                "treatment_type": "imaging",
                "diagnosis": "cancer"
            },
            {
                "request_id": "PA-2024-003",
                "patient_id": "P12347", 
                "status": AuthStatus.DENIED,
                "denial_reason": "Treatment not medically necessary",
                "treatment_type": "therapy",
                "diagnosis": "chronic_pain"
            }
        ]
        
        for auth in sample_auths:
            existing[auth["request_id"]] = PriorAuthResponse(
                request_id=auth["request_id"],
                status=auth["status"],
                auth_required=True,
                auth_number=auth.get("auth_number"),
                expiration_date=auth.get("expiration_date"),
                approval_date=auth.get("approval_date"),
                denial_reason=auth.get("denial_reason"),
                required_documents=[],
                estimated_processing_time=5,
                clinical_criteria="Standard medical necessity criteria",
                cost_estimate=5000.0,
                coverage_percentage=80.0,
                patient_responsibility=1000.0
            )
        
        return existing
    
    def _simulate_api_latency(self) -> float:
        """Simulate realistic API response time."""
        # Base latency: 200-800ms
        base_latency = random.uniform(0.2, 0.8)
        
        # Add occasional spikes
        if random.random() < 0.1:  # 10% chance of slow response
            base_latency += random.uniform(1.0, 3.0)
        
        return base_latency
    
    def _simulate_network_error(self) -> bool:
        """Simulate network errors."""
        if self.induce_prior_auth_error:
            return True
        
        # Random errors
        if random.random() < self.error_probability:
            return True
        
        return False
    
    def _simulate_timeout(self) -> bool:
        """Simulate API timeouts."""
        if self.induce_prior_auth_error:
            return random.choice([True, False])  # 50% chance if error flag is set
        
        return random.random() < self.timeout_probability
    
    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        random_suffix = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))
        return f"PA-{timestamp}-{random_suffix}"
    
    def _determine_auth_requirement(self, treatment_type: str, diagnosis: str, 
                                  insurance_type: str, cost: float) -> bool:
        """Determine if prior authorization is required based on business rules."""
        # High-cost procedures typically require auth
        if cost > 10000:
            return True
        
        # Certain treatment types always require auth
        if treatment_type in ['surgery', 'imaging', 'specialist']:
            return True
        
        # Certain diagnoses require auth
        if diagnosis in ['cancer', 'heart_disease']:
            return True
        
        # Medicare/Medicaid have different rules
        if insurance_type in ['medicare', 'medicaid']:
            if cost > 5000:
                return True
        
        # Private insurance is more lenient
        if insurance_type == 'private':
            if cost > 15000:
                return True
        
        return False
    
    def _determine_approval_status(self, treatment_type: str, diagnosis: str,
                                 insurance_type: str, cost: float) -> AuthStatus:
        """Determine approval status based on business rules."""
        # Emergency procedures are usually approved
        if treatment_type == 'emergency':
            return AuthStatus.APPROVED
        
        # High-cost procedures are often denied
        if cost > 50000:
            return AuthStatus.DENIED
        
        # Certain combinations are typically approved
        if diagnosis == 'cancer' and treatment_type in ['surgery', 'imaging']:
            return AuthStatus.APPROVED
        
        if diagnosis == 'heart_disease' and treatment_type == 'surgery':
            return AuthStatus.APPROVED
        
        # Random approval with some bias
        approval_chance = 0.7  # 70% base approval rate
        
        # Adjust based on factors
        if insurance_type == 'medicare':
            approval_chance += 0.1
        elif insurance_type == 'uninsured':
            approval_chance -= 0.3
        
        if cost > 20000:
            approval_chance -= 0.2
        
        return AuthStatus.APPROVED if random.random() < approval_chance else AuthStatus.DENIED
    
    def check_prior_auth_requirement(self, patient_id: str, patient_name: str,
                                   treatment_type: str, diagnosis: str,
                                   insurance_type: str, cost: float,
                                   provider_id: str = "PROV001",
                                   provider_name: str = "Dr. Smith") -> Dict[str, Any]:
        """
        Check if a procedure requires prior authorization.
        
        Args:
            patient_id: Patient identifier
            patient_name: Patient name
            treatment_type: Type of treatment/procedure
            diagnosis: Medical diagnosis
            insurance_type: Type of insurance
            cost: Estimated cost
            provider_id: Provider identifier
            provider_name: Provider name
            
        Returns:
            Dictionary with prior authorization information
        """
        start_time = time.time()
        
        try:
            # Simulate API latency
            api_latency = self._simulate_api_latency()
            time.sleep(api_latency)
            
            # Check for simulated errors
            if self._simulate_network_error():
                return {
                    "success": False,
                    "error": "NETWORK_ERROR",
                    "error_message": "Unable to connect to Prior Authorization API. Please try again later.",
                    "api_response_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
            
            if self._simulate_timeout():
                return {
                    "success": False,
                    "error": "TIMEOUT_ERROR", 
                    "error_message": "Request timed out. Please try again.",
                    "api_response_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Determine if authorization is required
            auth_required = self._determine_auth_requirement(
                treatment_type, diagnosis, insurance_type, cost
            )
            
            # Generate response
            response = {
                "success": True,
                "request_id": self._generate_request_id(),
                "patient_id": patient_id,
                "patient_name": patient_name,
                "provider_id": provider_id,
                "provider_name": provider_name,
                "treatment_type": treatment_type,
                "diagnosis": diagnosis,
                "insurance_type": insurance_type,
                "cost_estimate": cost,
                "auth_required": auth_required,
                "status": AuthStatus.REQUIRED if auth_required else AuthStatus.NOT_REQUIRED,
                "api_response_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add additional details if auth is required
            if auth_required:
                coverage_percentage = random.uniform(60, 90)
                patient_responsibility = cost * (1 - coverage_percentage / 100)
                response.update({
                    "estimated_processing_time": random.randint(3, 10),  # days
                    "required_documents": [
                        "Medical necessity letter from provider",
                        "Clinical documentation",
                        "Insurance card copy",
                        "Treatment plan"
                    ],
                    "clinical_criteria": "Standard medical necessity criteria apply",
                    "coverage_percentage": coverage_percentage,
                    "patient_responsibility": patient_responsibility
                })
            
            return response
            
        except Exception as e:
            return {
                "success": False,
                "error": "INTERNAL_ERROR",
                "error_message": f"Internal server error: {str(e)}",
                "api_response_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def submit_prior_auth_request(self, patient_id: str, patient_name: str,
                                treatment_type: str, diagnosis: str,
                                insurance_type: str, cost: float,
                                clinical_notes: str = "",
                                urgency: str = "routine",
                                provider_id: str = "PROV001",
                                provider_name: str = "Dr. Smith") -> Dict[str, Any]:
        """
        Submit a prior authorization request.
        
        Args:
            patient_id: Patient identifier
            patient_name: Patient name
            treatment_type: Type of treatment/procedure
            diagnosis: Medical diagnosis
            insurance_type: Type of insurance
            cost: Estimated cost
            clinical_notes: Clinical justification
            urgency: Urgency level (routine, urgent, emergency)
            provider_id: Provider identifier
            provider_name: Provider name
            
        Returns:
            Dictionary with prior authorization request result
        """
        start_time = time.time()
        
        try:
            # Simulate API latency
            api_latency = self._simulate_api_latency()
            time.sleep(api_latency)
            
            # Check for simulated errors
            if self._simulate_network_error():
                return {
                    "success": False,
                    "error": "NETWORK_ERROR",
                    "error_message": "Unable to submit prior authorization request. Please try again later.",
                    "api_response_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
            
            if self._simulate_timeout():
                return {
                    "success": False,
                    "error": "TIMEOUT_ERROR",
                    "error_message": "Request timed out. Please try again.",
                    "api_response_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Generate request ID
            request_id = self._generate_request_id()
            
            # Determine approval status
            approval_status = self._determine_approval_status(
                treatment_type, diagnosis, insurance_type, cost
            )
            
            # Generate response
            response = {
                "success": True,
                "request_id": request_id,
                "patient_id": patient_id,
                "patient_name": patient_name,
                "provider_id": provider_id,
                "provider_name": provider_name,
                "treatment_type": treatment_type,
                "diagnosis": diagnosis,
                "insurance_type": insurance_type,
                "cost_estimate": cost,
                "clinical_notes": clinical_notes,
                "urgency": urgency,
                "status": approval_status,
                "submission_date": datetime.now().isoformat(),
                "api_response_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add status-specific details
            if approval_status == AuthStatus.APPROVED:
                coverage_percentage = random.uniform(70, 95)
                patient_responsibility = cost * (1 - coverage_percentage / 100)
                response.update({
                    "auth_number": f"AUTH-{request_id}",
                    "approval_date": datetime.now().isoformat(),
                    "expiration_date": (datetime.now() + timedelta(days=90)).isoformat(),
                    "coverage_percentage": coverage_percentage,
                    "patient_responsibility": patient_responsibility,
                    "approval_notes": "Approved based on medical necessity"
                })
            elif approval_status == AuthStatus.DENIED:
                response.update({
                    "denial_reason": random.choice([
                        "Treatment not medically necessary",
                        "Alternative treatment available",
                        "Insufficient clinical documentation",
                        "Treatment not covered under plan",
                        "Cost exceeds plan limits"
                    ]),
                    "appeal_deadline": (datetime.now() + timedelta(days=30)).isoformat(),
                    "appeal_process": "Submit appeal within 30 days with additional documentation"
                })
            elif approval_status == AuthStatus.PENDING:
                response.update({
                    "estimated_processing_time": random.randint(3, 15),  # days
                    "review_status": "Under clinical review",
                    "next_update": (datetime.now() + timedelta(days=2)).isoformat()
                })
            
            return response
            
        except Exception as e:
            return {
                "success": False,
                "error": "INTERNAL_ERROR",
                "error_message": f"Internal server error: {str(e)}",
                "api_response_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_auth_status(self, request_id: str) -> Dict[str, Any]:
        """
        Get the status of an existing prior authorization request.
        
        Args:
            request_id: Prior authorization request ID
            
        Returns:
            Dictionary with current status information
        """
        start_time = time.time()
        
        try:
            # Simulate API latency
            api_latency = self._simulate_api_latency()
            time.sleep(api_latency)
            
            # Check for simulated errors
            if self._simulate_network_error():
                return {
                    "success": False,
                    "error": "NETWORK_ERROR",
                    "error_message": "Unable to retrieve authorization status. Please try again later.",
                    "api_response_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Check if request exists in our simulated database
            if request_id in self.existing_auths:
                auth = self.existing_auths[request_id]
                return {
                    "success": True,
                    "request_id": request_id,
                    "status": auth.status,
                    "auth_number": auth.auth_number,
                    "expiration_date": auth.expiration_date.isoformat() if auth.expiration_date else None,
                    "approval_date": auth.approval_date.isoformat() if auth.approval_date else None,
                    "denial_reason": auth.denial_reason,
                    "api_response_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": "NOT_FOUND",
                    "error_message": f"Prior authorization request {request_id} not found.",
                    "api_response_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": "INTERNAL_ERROR",
                "error_message": f"Internal server error: {str(e)}",
                "api_response_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def _run(self, action: str, **kwargs) -> str:
        """
        Main entry point for the tool.
        
        Args:
            action: Action to perform ('check', 'submit', 'status')
            **kwargs: Additional parameters based on action
            
        Returns:
            JSON string with API response
        """
        try:
            if action == "check":
                return json.dumps(
                    self.check_prior_auth_requirement(**kwargs),
                    indent=2,
                    default=str
                )
            elif action == "submit":
                return json.dumps(
                    self.submit_prior_auth_request(**kwargs),
                    indent=2,
                    default=str
                )
            elif action == "status":
                return json.dumps(
                    self.get_auth_status(**kwargs),
                    indent=2,
                    default=str
                )
            else:
                return json.dumps({
                    "success": False,
                    "error": "INVALID_ACTION",
                    "error_message": f"Invalid action: {action}. Valid actions: check, submit, status"
                }, indent=2)
                
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": "TOOL_ERROR",
                "error_message": f"Tool execution error: {str(e)}"
            }, indent=2)

# Example usage and testing
def main():
    """Test the Prior Authorization API Tool."""
    print("Testing Prior Authorization API Tool")
    print("=" * 50)
    
    # Test with normal operation
    print("\n1. Testing normal operation:")
    tool = PriorAuthAPITool(induce_prior_auth_error=False)
    
    # Test check requirement
    print("\nChecking prior auth requirement...")
    result = tool.check_prior_auth_requirement(
        patient_id="P12345",
        patient_name="John Doe",
        treatment_type="surgery",
        diagnosis="heart_disease",
        insurance_type="private",
        cost=25000.0
    )
    print(json.dumps(result, indent=2))
    
    # Test submit request
    print("\nSubmitting prior auth request...")
    result = tool.submit_prior_auth_request(
        patient_id="P12345",
        patient_name="John Doe",
        treatment_type="surgery",
        diagnosis="heart_disease",
        insurance_type="private",
        cost=25000.0,
        clinical_notes="Patient requires cardiac surgery for severe heart disease"
    )
    print(json.dumps(result, indent=2))
    
    # Test with error simulation
    print("\n2. Testing error simulation:")
    error_tool = PriorAuthAPITool(induce_prior_auth_error=True)
    
    print("\nChecking prior auth requirement (with error simulation)...")
    result = error_tool.check_prior_auth_requirement(
        patient_id="P12345",
        patient_name="John Doe",
        treatment_type="surgery",
        diagnosis="heart_disease",
        insurance_type="private",
        cost=25000.0
    )
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main() 