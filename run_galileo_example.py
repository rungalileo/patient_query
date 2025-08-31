#!/usr/bin/env python3
"""
Simple script to run the Galileo evaluation example.
This demonstrates how to import and use the sampleEvalScenario.json with Galileo logging.
"""

import os
import sys
from galileo_eval_example import GalileoEvalRunner

def main():
    """Run a simple demonstration of Galileo logging with the evaluation scenario."""
    
    print("🔬 Running Galileo Evaluation Example")
    print("=" * 60)
    
    # Check if required environment variables are set
    required_env_vars = ["GALILEO_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print("⚠️  Warning: Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nTo enable Galileo logging, please set these in your .env file:")
        print("   GALILEO_API_KEY=your_api_key_here")
        print("   GALILEO_PROJECT=your_project_name (optional)")
        print("   GALILEO_LOG_STREAM=your_log_stream_name (optional)")
        print("\nContinuing with logging disabled...")
    
    # Initialize the runner with custom project/logstream names
    runner = GalileoEvalRunner(
        project_name="talkdesk-demo",
        logstream_name="demo-scenario-run"
    )
    
    # Check if the scenario file exists
    scenario_file = "sampleEvalScenario.json"
    if not os.path.exists(scenario_file):
        print(f"❌ Scenario file '{scenario_file}' not found in current directory.")
        print("Please make sure you're running this script from the project root.")
        sys.exit(1)
    
    try:
        # Load and run the scenario
        scenario = runner.load_evaluation_scenario(scenario_file)
        
        print(f"\n📋 Scenario Details:")
        print(f"   Name: {scenario.get('name')}")
        print(f"   Test Type: {scenario.get('metadata', {}).get('test_type')}")
        print(f"   Complexity: {scenario.get('metadata', {}).get('test_complexity')}")
        print(f"   User Input: {scenario['datapoint']['message'][:100]}...")
        
        # Run the evaluation
        results = runner.run_scenario_with_logging(scenario)
        
        # Display summary
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        
        if "error" in results:
            print(f"❌ Execution failed: {results['error']}")
        else:
            evaluation = results["evaluation"]
            agent_result = results["agent_result"]
            
            print(f"🎯 Overall Success: {'✅ PASSED' if evaluation['overall_success'] else '❌ FAILED'}")
            print(f"⏱️  Execution Time: {results['duration']:.2f} seconds")
            print(f"🔄 Processing Time: {agent_result.get('metadata', {}).get('processing_time', 0):.2f} seconds")
            
            print(f"\n📋 Success Indicators:")
            for indicator, passed in evaluation["success_indicators"].items():
                status = "✅" if passed else "❌"
                print(f"   {status} {indicator.replace('_', ' ').title()}")
            
            # Show agent's response
            final_response = agent_result.get('final_response', '')
            if final_response:
                print(f"\n💬 Agent Response:")
                # Truncate long responses for display
                if len(final_response) > 200:
                    print(f"   {final_response[:200]}...")
                else:
                    print(f"   {final_response}")
            
            # Show intent classification if available
            intent_result = agent_result.get('intent_result')
            if intent_result:
                print(f"\n🧠 Intent Classification:")
                print(f"   Type: {intent_result.intent_type}")
                print(f"   Confidence: {intent_result.confidence:.2f}")
        
        # Galileo information
        if runner.galileo_logger:
            print(f"\n📈 Galileo Logging:")
            print(f"   Project: {runner.project_name}")
            print(f"   Log Stream: {runner.logstream_name}")
            print(f"   Trace ID: {results.get('trace_id', 'N/A')}")
            print(f"   💡 Check your Galileo dashboard for detailed traces!")
        else:
            print(f"\n⚠️  Galileo logging was disabled (missing API key)")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n🎉 Evaluation complete!")

if __name__ == "__main__":
    main()
