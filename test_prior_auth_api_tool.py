from prior_auth_api_tool import PriorAuthAPITool
import json

def test_prior_auth_normal():
    tool = PriorAuthAPITool(induce_prior_auth_error=False)
    result = tool.check_prior_auth_requirement(
        patient_id="P12345",
        patient_name="John Doe",
        treatment_type="surgery",
        diagnosis="heart_disease",
        insurance_type="private",
        cost=25000.0
    )
    print("Normal operation result:")
    print(json.dumps(result, indent=2))
    assert result["success"] is True
    assert "auth_required" in result


def test_prior_auth_error():
    tool = PriorAuthAPITool(induce_prior_auth_error=True)
    result = tool.check_prior_auth_requirement(
        patient_id="P12345",
        patient_name="John Doe",
        treatment_type="surgery",
        diagnosis="heart_disease",
        insurance_type="private",
        cost=25000.0
    )
    print("Error simulation result:")
    print(json.dumps(result, indent=2))
    assert result["success"] is False
    assert "error" in result


if __name__ == "__main__":
    test_prior_auth_normal()
    test_prior_auth_error() 