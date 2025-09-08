#!/usr/bin/env python3
"""
Example script demonstrating Galileo logging with evaluation scenario data.
This script shows how to use the healthcare agent's Galileo implementation
with structured evaluation data from sampleEvalScenario.json.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv, find_dotenv

# Import Galileo logger and healthcare agent components
from galileo import GalileoLogger
from healthcare_agent import MedicalAgent, initialize_galileo

# Load environment variables
# 1) load global/shared first
load_dotenv(os.path.expanduser("~/.config/secrets/myapps.env"), override=False)
# 2) then load per-app .env (if present) to override selectively
load_dotenv(find_dotenv(usecwd=True), override=True)

class GalileoEvalRunner:
    """
    Runner class that demonstrates Galileo logging with evaluation scenarios.
    """
    
    def __init__(self, project_name: str = None, logstream_name: str = None):
        """Initialize the evaluation runner with Galileo logging."""
        self.project_name = project_name or os.getenv("GALILEO_PROJECT", "healthcare-agent-eval")
        self.logstream_name = logstream_name or os.getenv("GALILEO_LOG_STREAM", "eval-scenarios")
        
        # Initialize Galileo
        self._initialize_galileo()
        
        # Initialize the medical agent
        initialize_galileo(self.project_name, self.logstream_name)
        self.agent = MedicalAgent()
        
    def _initialize_galileo(self):
        """Initialize Galileo logger for evaluation tracking."""
        api_key = os.getenv("GALILEO_API_KEY")
        
        if not api_key:
            print("⚠️  Warning: GALILEO_API_KEY not found. Logging will be disabled.")
            self.galileo_logger = None
            return
            
        try:
            self.galileo_logger = GalileoLogger(
                project=self.project_name,
                log_stream=self.logstream_name
            )
            print(f"✅ Galileo logger initialized:")
            print(f"   Project: {self.project_name}")
            print(f"   Log Stream: {self.logstream_name}")
        except Exception as e:
            print(f"❌ Failed to initialize Galileo logger: {e}")
            self.galileo_logger = None
    
    def load_evaluation_scenario(self, scenario_file: str) -> Dict[str, Any]:
        """Load evaluation scenario from JSON file."""
        try:
            with open(scenario_file, 'r', encoding='utf-8') as f:
                scenario = json.load(f)
            print(f"📄 Loaded evaluation scenario: {scenario.get('name', 'Unknown')}")
            return scenario
        except FileNotFoundError:
            print(f"❌ Scenario file not found: {scenario_file}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in scenario file: {e}")
            raise
    
    def log_evaluation_start(self, scenario: Dict[str, Any]) -> str:
        """Log the start of an evaluation scenario."""
        if not self.galileo_logger:
            return None
            
        trace_id = f"eval_{scenario.get('id', 'unknown')}_{int(time.time())}"
        
        # Start a trace for the entire evaluation
        self.galileo_logger.start_trace(
            input=scenario['datapoint']['message'],
            name=f"Evaluation: {scenario.get('name', 'Unknown Scenario')}",
            tags=[
                "evaluation",
                "scenario_test",
                scenario.get('metadata', {}).get('test_type', 'unknown'),
                scenario.get('metadata', {}).get('test_complexity', 'unknown')
            ]
        )
        
        # Log scenario metadata
        self.galileo_logger.add_tool_span(
            input=json.dumps(scenario['metadata'], indent=2),
            output=f"Starting evaluation for scenario: {scenario.get('name')}",
            name="Evaluation Metadata",
            duration_ns=1000000,  # 1ms
            metadata={
                "scenario_id": scenario.get('id'),
                "configuration_id": scenario.get('configuration_id'),
                "test_type": scenario.get('metadata', {}).get('test_type'),
                "test_complexity": scenario.get('metadata', {}).get('test_complexity'),
                "num_orders": scenario.get('metadata', {}).get('num_orders'),
                "user_type": scenario.get('metadata', {}).get('user_type')
            }
        )
        
        return trace_id
    
    def run_scenario_with_logging(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run an evaluation scenario with comprehensive Galileo logging.
        
        Args:
            scenario: The loaded evaluation scenario
            
        Returns:
            Dictionary containing the results and evaluation metrics
        """
        print(f"\n🚀 Running evaluation scenario: {scenario.get('name')}")
        print(f"📝 Message: {scenario['datapoint']['message']}")
        
        # Start evaluation logging
        trace_id = self.log_evaluation_start(scenario)
        start_time = time.time()
        
        try:
            # Log expected tool calls
            self._log_expected_goals(scenario.get('metrics', {}).get('goals', []))
            
            # Process the query through the agent
            user_message = scenario['datapoint']['message']
            result = self.agent.process_query(user_message)
            
            # Log the results
            evaluation_results = self._evaluate_results(scenario, result)
            
            # Log evaluation results
            self._log_evaluation_results(scenario, result, evaluation_results)
            
            # Conclude the trace
            if self.galileo_logger:
                total_duration = time.time() - start_time
                self.galileo_logger.conclude(
                    output=json.dumps({
                        "final_response": result.get('final_response'),
                        "evaluation_results": evaluation_results,
                        "scenario_passed": evaluation_results.get('overall_success', False)
                    }, indent=2),
                    duration_ns=int(total_duration * 1000000000),
                    status_code=200 if evaluation_results.get('overall_success') else 400
                )
                self.galileo_logger.flush()
                print(f"📊 Flushed evaluation results to Galileo project: {self.project_name}")
            
            return {
                "scenario": scenario,
                "agent_result": result,
                "evaluation": evaluation_results,
                "trace_id": trace_id,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            # Log error and conclude trace
            if self.galileo_logger:
                self.galileo_logger.conclude(
                    output=json.dumps({"error": str(e)}),
                    duration_ns=int((time.time() - start_time) * 1000000000),
                    status_code=500
                )
                self.galileo_logger.flush()
            
            print(f"❌ Error running scenario: {e}")
            return {
                "scenario": scenario,
                "error": str(e),
                "trace_id": trace_id,
                "duration": time.time() - start_time
            }
    
    def _log_expected_goals(self, goals: List[Dict[str, Any]]):
        """Log the expected tool calls for comparison."""
        if not self.galileo_logger or not goals:
            return
            
        goals_summary = []
        for goal in goals:
            goal_info = {
                "order": goal.get('order'),
                "type": goal.get('type'),
                "tool_id": goal.get('value', {}).get('tool_id'),
                "inputs": goal.get('value', {}).get('inputs', [])
            }
            goals_summary.append(goal_info)
        
        self.galileo_logger.add_tool_span(
            input="Expected agent behavior",
            output=json.dumps(goals_summary, indent=2),
            name="Expected Tool Sequence",
            duration_ns=1000000,
            metadata={
                "type": "expected_behavior",
                "num_expected_tools": len(goals),
                "tool_sequence": [g.get('value', {}).get('tool_id') for g in goals]
            }
        )
    
    def _evaluate_results(self, scenario: Dict[str, Any], agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the agent's performance against the scenario expectations.
        This is a simplified evaluation - in practice you'd want more sophisticated comparison.
        """
        evaluation_questions = scenario.get('metrics', {}).get('evaluation', [])
        expected_goals = scenario.get('metrics', {}).get('goals', [])
        
        # Simple evaluation based on whether agent completed successfully
        success_indicators = {
            "completed_successfully": bool(agent_result.get('final_response') and not agent_result.get('error')),
            "has_intent_classification": bool(agent_result.get('intent_result')),
            "processing_time_reasonable": agent_result.get('metadata', {}).get('processing_time', 0) < 30.0,
            "no_errors": not bool(agent_result.get('error'))
        }
        
        # Calculate overall success
        overall_success = all(success_indicators.values())
        
        return {
            "success_indicators": success_indicators,
            "overall_success": overall_success,
            "expected_questions": len(evaluation_questions),
            "expected_tools": len(expected_goals),
            "agent_metadata": agent_result.get('metadata', {}),
            "evaluation_timestamp": datetime.now().isoformat()
        }
    
    def _log_evaluation_results(self, scenario: Dict[str, Any], agent_result: Dict[str, Any], 
                               evaluation_results: Dict[str, Any]):
        """Log the evaluation results to Galileo."""
        if not self.galileo_logger:
            return
            
        # Log the evaluation outcome
        self.galileo_logger.add_tool_span(
            input=json.dumps({
                "scenario_name": scenario.get('name'),
                "user_message": scenario['datapoint']['message'],
                "expected_evaluations": scenario.get('metrics', {}).get('evaluation', [])
            }, indent=2),
            output=json.dumps(evaluation_results, indent=2),
            name="Scenario Evaluation Results",
            duration_ns=1000000,
            metadata={
                "overall_success": str(evaluation_results.get('overall_success')),
                "scenario_id": scenario.get('id'),
                "test_type": scenario.get('metadata', {}).get('test_type'),
                "agent_response_length": len(agent_result.get('final_response', '')),
                "has_error": str(bool(agent_result.get('error')))
            }
        )


def main():
    """
    Main function demonstrating how to run evaluation scenarios with Galileo logging.
    """
    print("🔬 Galileo Evaluation Example")
    print("=" * 50)
    
    # Initialize the evaluation runner
    runner = GalileoEvalRunner(
        project_name="healthcare-agent-evaluation",
        logstream_name=f"eval-run-{int(time.time())}"
    )
    
    # Load the sample evaluation scenario
    try:
        scenario = runner.load_evaluation_scenario("sampleEvalScenario.json")
    except Exception as e:
        print(f"Failed to load scenario: {e}")
        return
    
    # Run the scenario with logging
    results = runner.run_scenario_with_logging(scenario)
    
    # Display results
    print("\n📊 Evaluation Results:")
    print("=" * 50)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
    else:
        evaluation = results["evaluation"]
        print(f"✅ Overall Success: {evaluation['overall_success']}")
        print(f"⏱️  Duration: {results['duration']:.2f} seconds")
        print(f"🎯 Success Indicators:")
        
        for indicator, passed in evaluation["success_indicators"].items():
            status = "✅" if passed else "❌"
            print(f"   {status} {indicator.replace('_', ' ').title()}")
        
        print(f"\n💬 Agent Response:")
        print(f"   {results['agent_result'].get('final_response', 'No response')}")
    
    print(f"\n🔗 Trace ID: {results.get('trace_id', 'Not available')}")
    print("\n🎉 Evaluation complete! Check your Galileo dashboard for detailed logs.")


if __name__ == "__main__":
    main()
