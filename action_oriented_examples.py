"""
Action-Oriented Example Queries for Medical Agent
These queries explicitly request filing claims and processing prior authorizations.
"""

# Example queries that explicitly request action

ACTION_ORIENTED_QUERIES = {
    "claim_filing": [
        "File a claim for Sarah Johnson's $2,500 MRI scan for her migraine diagnosis",
        "Submit an insurance claim for Michael Chen's $18,000 heart surgery",
        "Process a claim for Emily Rodriguez's $800 depression therapy sessions",
        "File a claim for Atin Sanyal's $1,200 diabetes medication",
        "Submit insurance claim for $5,000 specialist consultation for Sarah Johnson's asthma"
    ],
    
    "prior_authorization": [
        "Request prior authorization for Sarah Johnson's migraine medication",
        "Submit prior authorization request for Michael Chen's cardiac surgery",
        "File prior auth for Emily Rodriguez's depression treatment",
        "Request pre-authorization for Atin Sanyal's diabetes medication",
        "Submit prior authorization for $15,000 surgery for John Smith's heart disease"
    ],
    
    "claim_and_auth": [
        "File a claim for Sarah Johnson's $3,000 migraine treatment and request prior authorization",
        "Submit insurance claim for Michael Chen's heart surgery and get prior auth approval",
        "Process claim for Emily Rodriguez's therapy and request pre-authorization",
        "File claim for Atin Sanyal's medication and submit prior auth request",
        "Submit claim for $20,000 surgery and request prior authorization for John Smith"
    ],
    
    "urgent_requests": [
        "URGENT: File emergency claim for Sarah Johnson's asthma attack treatment",
        "EMERGENCY: Submit claim for Michael Chen's cardiac emergency surgery",
        "URGENT: Process claim for Emily Rodriguez's crisis intervention therapy",
        "EMERGENCY: File claim for Atin Sanyal's diabetic emergency medication",
        "URGENT: Submit claim for $25,000 emergency surgery for John Smith"
    ],
    
    "follow_up_actions": [
        "Check status of Sarah Johnson's claim for migraine medication",
        "Follow up on Michael Chen's prior authorization for heart surgery",
        "Track Emily Rodriguez's claim for depression treatment",
        "Monitor Atin Sanyal's claim status for diabetes medication",
        "Check approval status for John Smith's surgery claim"
    ]
}

# Example workflow descriptions
WORKFLOW_EXAMPLES = {
    "standard_claim_workflow": {
        "query": "File a claim for Sarah Johnson's $2,500 MRI scan for her migraine diagnosis",
        "expected_workflow": [
            "1. Intent Classification → claim_approval",
            "2. Extract claim information (patient, treatment, cost, diagnosis)",
            "3. Process claim through Claim Approval Tool",
            "4. If approved → Check if prior authorization is required",
            "5. If prior auth required → Submit prior authorization request",
            "6. Combine responses with claim decision and auth requirements"
        ],
        "expected_output": {
            "claim_decision": "Approved/Denied with confidence score",
            "prior_auth": "Required/Not required with processing details",
            "next_steps": "Required documents and timeline"
        }
    },
    
    "emergency_workflow": {
        "query": "URGENT: File emergency claim for Sarah Johnson's asthma attack treatment",
        "expected_workflow": [
            "1. Intent Classification → claim_approval (with urgency detection)",
            "2. Extract claim information with emergency flag",
            "3. Process claim with emergency priority",
            "4. Emergency claims typically auto-approved",
            "5. Prior auth may be waived for emergencies",
            "6. Fast-track response with emergency protocols"
        ],
        "expected_output": {
            "claim_decision": "Emergency approval with expedited processing",
            "prior_auth": "Waived due to emergency",
            "next_steps": "Immediate processing instructions"
        }
    }
}

# Test queries for different scenarios
TEST_SCENARIOS = {
    "high_cost_claim": "Submit a claim for $50,000 experimental cancer treatment for Sarah Johnson",
    "routine_medication": "File claim for Sarah Johnson's monthly asthma inhaler prescription",
    "preventive_care": "Submit claim for Atin Sanyal's annual diabetes screening",
    "specialist_consultation": "File claim for Michael Chen's cardiologist consultation",
    "therapy_session": "Submit claim for Emily Rodriguez's weekly therapy session",
    "surgery_with_complications": "File claim for $35,000 surgery with post-operative complications for John Smith"
}

def print_example_queries():
    """Print all example queries organized by category."""
    print("🎯 ACTION-ORIENTED EXAMPLE QUERIES")
    print("=" * 60)
    
    for category, queries in ACTION_ORIENTED_QUERIES.items():
        print(f"\n📋 {category.upper().replace('_', ' ')}:")
        for i, query in enumerate(queries, 1):
            print(f"  {i}. {query}")
    
    print(f"\n🔄 WORKFLOW EXAMPLES:")
    for workflow_name, details in WORKFLOW_EXAMPLES.items():
        print(f"\n  {workflow_name}:")
        print(f"    Query: {details['query']}")
        print(f"    Expected Workflow:")
        for step in details['expected_workflow']:
            print(f"      {step}")

if __name__ == "__main__":
    print_example_queries() 