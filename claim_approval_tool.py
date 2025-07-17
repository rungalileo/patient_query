"""
Claim Approval Tool
Uses a logistic regression classifier to determine if a medical claim should be approved or denied.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ClaimApprovalInput(BaseModel):
    patient_name: str = Field(description="Name of the patient")
    treatment_type: str = Field(description="Type of treatment or procedure")
    cost: float = Field(description="Cost of the treatment in dollars")
    diagnosis: str = Field(description="Medical diagnosis or condition")
    age: int = Field(description="Patient age")
    insurance_type: str = Field(description="Type of insurance (e.g., 'private', 'medicare', 'medicaid')")

class ClaimApprovalTool:
    """Claim Approval Tool using logistic regression classifier for medical claim decisions."""
    
    def __init__(self):
        self.name = "claim_approval_tool"
        self.description = "Determine if a medical claim should be approved or denied based on patient and treatment data"
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = ['age', 'cost', 'treatment_type_encoded', 'diagnosis_encoded', 'insurance_type_encoded']
        self._generate_training_data()
        self._train_model()
    
    def _generate_training_data(self):
        """Generate mock training data for claim approval."""
        np.random.seed(42)  # For reproducibility
        
        # Generate synthetic data
        n_samples = 1000
        
        # Patient ages (18-85)
        ages = np.random.normal(45, 15, n_samples)
        ages = np.clip(ages, 18, 85)
        
        # Treatment costs (100-50000)
        costs = np.random.exponential(5000, n_samples)
        costs = np.clip(costs, 100, 50000)
        
        # Treatment types
        treatment_types = [
            'surgery', 'medication', 'therapy', 'imaging', 'lab_test',
            'emergency_room', 'specialist_consultation', 'preventive_care'
        ]
        
        # Diagnoses
        diagnoses = [
            'hypertension', 'diabetes', 'heart_disease', 'cancer', 'asthma',
            'depression', 'arthritis', 'infection', 'injury', 'chronic_pain'
        ]
        
        # Insurance types
        insurance_types = ['private', 'medicare', 'medicaid', 'uninsured']
        
        # Generate random data
        data = []
        for i in range(n_samples):
            treatment_type = np.random.choice(treatment_types)
            diagnosis = np.random.choice(diagnoses)
            insurance_type = np.random.choice(insurance_types)
            
            # Create features that influence approval
            age = ages[i]
            cost = costs[i]
            
            # Approval logic based on business rules
            approved = self._business_logic_approval(
                age, cost, treatment_type, diagnosis, insurance_type
            )
            
            data.append({
                'age': age,
                'cost': cost,
                'treatment_type': treatment_type,
                'diagnosis': diagnosis,
                'insurance_type': insurance_type,
                'approved': approved
            })
        
        self.training_data = pd.DataFrame(data)
        print(f"Generated {len(self.training_data)} training samples")
    
    def _business_logic_approval(self, age: float, cost: float, treatment_type: str, 
                                diagnosis: str, insurance_type: str) -> bool:
        """Business logic to determine approval based on rules."""
        # Base approval probability
        approval_prob = 0.7
        
        # Age factors
        if age < 18 or age > 80:
            approval_prob -= 0.1
        
        # Cost factors
        if cost > 20000:
            approval_prob -= 0.3
        elif cost > 10000:
            approval_prob -= 0.1
        elif cost < 1000:
            approval_prob += 0.1
        
        # Treatment type factors
        if treatment_type in ['surgery', 'emergency_room']:
            approval_prob += 0.2
        elif treatment_type == 'preventive_care':
            approval_prob += 0.3
        elif treatment_type == 'lab_test':
            approval_prob += 0.1
        
        # Diagnosis factors
        if diagnosis in ['cancer', 'heart_disease']:
            approval_prob += 0.2
        elif diagnosis in ['infection', 'injury']:
            approval_prob += 0.1
        elif diagnosis == 'chronic_pain':
            approval_prob -= 0.1
        
        # Insurance factors
        if insurance_type == 'uninsured':
            approval_prob -= 0.2
        elif insurance_type == 'medicare':
            approval_prob += 0.1
        
        # Add some randomness
        approval_prob += np.random.normal(0, 0.1)
        approval_prob = np.clip(approval_prob, 0, 1)
        
        return approval_prob > 0.5
    
    def _train_model(self):
        """Train the logistic regression model."""
        print("Training claim approval model...")
        
        # Prepare features
        X = self.training_data.drop('approved', axis=1)
        y = self.training_data['approved']
        
        # Encode categorical variables
        for col in ['treatment_type', 'diagnosis', 'insurance_type']:
            le = LabelEncoder()
            X[f'{col}_encoded'] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
        # Select features
        X = X[self.feature_columns]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.3f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
    
    def _prepare_features(self, patient_name: str, treatment_type: str, cost: float,
                         diagnosis: str, age: int, insurance_type: str) -> np.ndarray:
        """Prepare features for prediction."""
        # Encode categorical variables
        treatment_type_encoded = self.label_encoders['treatment_type'].transform([treatment_type])[0]
        diagnosis_encoded = self.label_encoders['diagnosis'].transform([diagnosis])[0]
        insurance_type_encoded = self.label_encoders['insurance_type'].transform([insurance_type])[0]
        
        # Create feature array
        features = np.array([[
            age, cost, treatment_type_encoded, diagnosis_encoded, insurance_type_encoded
        ]])
        
        return features
    
    def _run(self, patient_name: str, treatment_type: str, cost: float,
             diagnosis: str, age: int, insurance_type: str) -> str:
        """Run the claim approval tool."""
        try:
            if treatment_type not in self.label_encoders['treatment_type'].classes_:
                return f"Error: Unknown treatment type '{treatment_type}'. Valid types: {list(self.label_encoders['treatment_type'].classes_)}"
            
            if diagnosis not in self.label_encoders['diagnosis'].classes_:
                return f"Error: Unknown diagnosis '{diagnosis}'. Valid diagnoses: {list(self.label_encoders['diagnosis'].classes_)}"
            
            if insurance_type not in self.label_encoders['insurance_type'].classes_:
                return f"Error: Unknown insurance type '{insurance_type}'. Valid types: {list(self.label_encoders['insurance_type'].classes_)}"
            
            features = self._prepare_features(
                patient_name, treatment_type, cost, diagnosis, age, insurance_type
            )
            
            features_scaled = self.scaler.transform(features)
            
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            result = {
                "patient_name": patient_name,
                "treatment_type": treatment_type,
                "cost": cost,
                "diagnosis": diagnosis,
                "age": age,
                "insurance_type": insurance_type,
                "decision": "Approved" if prediction else "Denied",
                "confidence": float(max(probability)),
                "approval_probability": float(probability[1]) if len(probability) > 1 else float(probability[0])
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error processing claim approval: {str(e)}"
    
    async def _arun(self, patient_name: str, treatment_type: str, cost: float,
                    diagnosis: str, age: int, insurance_type: str) -> str:
        """Async version of the run method."""
        return self._run(patient_name, treatment_type, cost, diagnosis, age, insurance_type)
    
    def get_claim_summary(self, patient_name: str, treatment_type: str, cost: float,
                         diagnosis: str, age: int, insurance_type: str) -> Dict[str, Any]:
        """Get a summary of the claim for display purposes."""
        result = self._run(patient_name, treatment_type, cost, diagnosis, age, insurance_type)
        
        try:
            result_dict = json.loads(result)
            return result_dict
        except:
            return {"error": result} 