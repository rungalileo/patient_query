"""
Mock patient data for the patient query Q&A chatbot.
This contains sample patient records with medical history, allergies, and conditions.
"""

MOCK_PATIENT_RECORDS = [
    {
        "patient_id": "P001",
        "name": "Atin Sanyal",
        "age": 35,
        "gender": "Male",
        "medical_history": [
            "Hypertension (diagnosed 2020)",
            "Type 2 Diabetes (diagnosed 2019)",
            "Seasonal allergies (pollen, dust)",
            "Previous surgery: Appendectomy (2015)"
        ],
        "current_medications": [
            "Metformin 500mg twice daily",
            "Lisinopril 10mg once daily",
            "Cetirizine 10mg as needed for allergies"
        ],
        "allergies": [
            "Penicillin - Severe allergic reaction",
            "Sulfa drugs - Rash and itching",
            "Peanuts - Anaphylaxis risk"
        ],
        "conditions": [
            "Hypertension",
            "Type 2 Diabetes",
            "Seasonal Allergies"
        ],
        "lab_results": {
            "blood_pressure": "140/90 mmHg",
            "blood_sugar": "180 mg/dL (fasting)",
            "cholesterol": "220 mg/dL"
        },
        "last_visit": "2024-01-15",
        "next_appointment": "2024-04-15"
    },
    {
        "patient_id": "P002",
        "name": "Sarah Johnson",
        "age": 28,
        "gender": "Female",
        "medical_history": [
            "Asthma (diagnosed 2018)",
            "Migraine headaches",
            "Anxiety disorder"
        ],
        "current_medications": [
            "Albuterol inhaler as needed",
            "Sumatriptan 50mg for migraines",
            "Sertraline 50mg once daily"
        ],
        "allergies": [
            "Aspirin - Stomach irritation",
            "Latex - Skin rash"
        ],
        "conditions": [
            "Asthma",
            "Migraine",
            "Anxiety"
        ],
        "lab_results": {
            "blood_pressure": "120/80 mmHg",
            "peak_flow": "450 L/min",
            "heart_rate": "72 bpm"
        },
        "last_visit": "2024-02-20",
        "next_appointment": "2024-05-20"
    },
    {
        "patient_id": "P003",
        "name": "Michael Chen",
        "age": 45,
        "gender": "Male",
        "medical_history": [
            "Heart disease (diagnosed 2021)",
            "High cholesterol",
            "Sleep apnea"
        ],
        "current_medications": [
            "Atorvastatin 20mg once daily",
            "Metoprolol 50mg twice daily",
            "CPAP machine for sleep apnea"
        ],
        "allergies": [
            "Codeine - Respiratory depression",
            "Shellfish - Hives and swelling"
        ],
        "conditions": [
            "Coronary Artery Disease",
            "Hyperlipidemia",
            "Obstructive Sleep Apnea"
        ],
        "lab_results": {
            "blood_pressure": "135/85 mmHg",
            "cholesterol": "180 mg/dL",
            "ejection_fraction": "55%"
        },
        "last_visit": "2024-03-10",
        "next_appointment": "2024-06-10"
    },
    {
        "patient_id": "P004",
        "name": "Emily Rodriguez",
        "age": 32,
        "gender": "Female",
        "medical_history": [
            "Hypothyroidism (diagnosed 2017)",
            "Depression",
            "Iron deficiency anemia"
        ],
        "current_medications": [
            "Levothyroxine 75mcg once daily",
            "Fluoxetine 20mg once daily",
            "Ferrous sulfate 325mg twice daily"
        ],
        "allergies": [
            "Iodine - Thyroid swelling",
            "Soy - Digestive issues"
        ],
        "conditions": [
            "Hypothyroidism",
            "Major Depressive Disorder",
            "Iron Deficiency Anemia"
        ],
        "lab_results": {
            "tsh": "2.5 mIU/L",
            "hemoglobin": "11.5 g/dL",
            "ferritin": "25 ng/mL"
        },
        "last_visit": "2024-01-30",
        "next_appointment": "2024-04-30"
    }
]

MEDICATION_DATABASE = {
    "Aspirin": {
        "description": "Pain reliever and blood thinner",
        "contraindications": [
            "Allergy to aspirin or NSAIDs",
            "Active bleeding or ulcers",
            "Children under 18 with viral infections (Reye's syndrome risk)",
            "Pregnancy (especially third trimester)"
        ],
        "interactions": [
            "Blood thinners (increased bleeding risk)",
            "ACE inhibitors (reduced effectiveness)",
            "Diuretics (reduced effectiveness)"
        ]
    },
    "Ibuprofen": {
        "description": "Non-steroidal anti-inflammatory drug (NSAID)",
        "contraindications": [
            "Allergy to NSAIDs",
            "Active stomach ulcers",
            "Severe kidney disease",
            "Heart failure"
        ],
        "interactions": [
            "Blood thinners",
            "ACE inhibitors",
            "Diuretics"
        ]
    },
    "Acetaminophen": {
        "description": "Pain reliever and fever reducer",
        "contraindications": [
            "Severe liver disease",
            "Alcohol abuse (increased liver damage risk)"
        ],
        "interactions": [
            "Warfarin (increased bleeding risk)",
            "Alcohol (increased liver damage risk)"
        ]
    },
    "Amoxicillin": {
        "description": "Antibiotic for bacterial infections",
        "contraindications": [
            "Allergy to penicillin or cephalosporins",
            "Mononucleosis (increased rash risk)"
        ],
        "interactions": [
            "Birth control pills (reduced effectiveness)",
            "Methotrexate (increased toxicity)"
        ]
    },
    "Lisinopril": {
        "description": "ACE inhibitor for high blood pressure",
        "contraindications": [
            "Pregnancy (especially second and third trimesters)",
            "Angioedema history",
            "Severe kidney disease"
        ],
        "interactions": [
            "Potassium supplements (increased potassium levels)",
            "Lithium (increased lithium levels)",
            "NSAIDs (reduced effectiveness)"
        ]
    },
    "Metformin": {
        "description": "Oral diabetes medication",
        "contraindications": [
            "Severe kidney disease",
            "Metabolic acidosis",
            "Heart failure requiring hospitalization"
        ],
        "interactions": [
            "Alcohol (increased lactic acidosis risk)",
            "Contrast dye (temporary discontinuation needed)"
        ]
    },
    "Atorvastatin": {
        "description": "Cholesterol-lowering medication",
        "contraindications": [
            "Active liver disease",
            "Pregnancy",
            "Breastfeeding"
        ],
        "interactions": [
            "Grapefruit juice (increased drug levels)",
            "Certain antibiotics",
            "Warfarin (increased bleeding risk)"
        ]
    }
}

# Common symptoms and their potential causes
SYMPTOM_DATABASE = {
    "runny nose": {
        "common_causes": [
            "Common cold",
            "Seasonal allergies",
            "Sinus infection",
            "Flu"
        ],
        "recommended_actions": [
            "Rest and hydration",
            "Saline nasal spray",
            "Over-the-counter decongestants (if no contraindications)",
            "Consult doctor if symptoms persist beyond 10 days"
        ]
    },
    "fever": {
        "common_causes": [
            "Viral infection",
            "Bacterial infection",
            "Inflammatory conditions"
        ],
        "recommended_actions": [
            "Rest and hydration",
            "Acetaminophen or ibuprofen (if no contraindications)",
            "Consult doctor if fever >103°F or persists >3 days"
        ]
    },
    "headache": {
        "common_causes": [
            "Tension headache",
            "Migraine",
            "Dehydration",
            "Stress"
        ],
        "recommended_actions": [
            "Rest in quiet, dark room",
            "Hydration",
            "Over-the-counter pain relievers (if no contraindications)",
            "Consult doctor if severe or accompanied by other symptoms"
        ]
    },
    "cough": {
        "common_causes": [
            "Upper respiratory infection",
            "Allergies",
            "Post-nasal drip",
            "Bronchitis"
        ],
        "recommended_actions": [
            "Honey for soothing (adults only)",
            "Humidifier",
            "Over-the-counter cough suppressants (if no contraindications)",
            "Consult doctor if cough persists >2 weeks"
        ]
    }
}

def get_patient_by_name(name):
    """Get patient record by name (case-insensitive)."""
    for patient in MOCK_PATIENT_RECORDS:
        if patient["name"].lower() == name.lower():
            return patient
    return None

def get_medication_info(medication_name):
    """Get medication information by name (case-insensitive)."""
    for med_name, info in MEDICATION_DATABASE.items():
        if med_name.lower() == medication_name.lower():
            return info
    return None

def get_symptom_info(symptom):
    """Get symptom information (case-insensitive)."""
    for symptom_name, info in SYMPTOM_DATABASE.items():
        if symptom_name.lower() == symptom.lower():
            return info
    return None 