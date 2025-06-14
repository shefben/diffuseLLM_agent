from flask import Flask, render_template, request, redirect, url_for
import sys
from pathlib import Path

# Add project root to sys.path to allow imports from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import json # For json.dumps as a fallback for displaying objects

# Conditional imports for project modules, handle if not found during early setup
try:
    from src.utils.config_loader import load_app_config
    from src.digester.repository_digester import RepositoryDigester
    from src.planner.phase_planner import PhasePlanner
    # Spec is needed for type hints and potentially direct use if normalise_request returns it
    from src.planner.spec_model import Spec
except ImportError as e:
    print(f"Error importing project modules in webapp/app.py: {e}. Ensure PROJECT_ROOT is correct and src modules are accessible.")
    # Define placeholders if imports fail, to allow Flask app to run for basic HTML serving
    load_app_config = lambda: {"error": "load_app_config failed"}
    RepositoryDigester = lambda path, config: {"error": "RepositoryDigester failed"} # type: ignore
    PhasePlanner = lambda root, config, digester: {"error": "PhasePlanner failed"} # type: ignore
    Spec = dict # Placeholder type

app = Flask(__name__)

# Load application configuration
# This assumes config.yaml or defaults are accessible relative to PROJECT_ROOT
# or via load_app_config's internal logic.
app_config_global = {}
digester_global = None
phase_planner_global = None

try:
    app_config_global = load_app_config()
    # Initialize digester and planner globally if they are stateless enough
    # or if their state is managed per request. For now, simple global init.
    # This might need adjustment if they are stateful and need per-request setup.
    repo_path_str = app_config_global.get("general", {}).get("project_root", str(PROJECT_ROOT))

    if isinstance(RepositoryDigester, type): # Check if it's the actual class
        digester_global = RepositoryDigester(repo_path=Path(repo_path_str), app_config=app_config_global)
        # Optionally, run a quick digest if it's fast and needed for all requests.
        # For a webapp, might be better to have a pre-digested state or digest on demand.
        # digester_global.digest_repository() # Potentially long running

    if isinstance(PhasePlanner, type) and digester_global and not isinstance(digester_global, dict):
            phase_planner_global = PhasePlanner(
            project_root_path=Path(repo_path_str),
            app_config=app_config_global,
            digester=digester_global
        )
    elif isinstance(PhasePlanner, type) and isinstance(digester_global, dict): # Digester init failed
            print("WebApp Warning: Digester failed to initialize, PhasePlanner cannot be initialized with it.")

    print("WebApp: Global components (app_config, digester, planner) initialized.")
    if isinstance(digester_global, dict) and "error" in digester_global : print(f"  Digester status: {digester_global['error']}")
    if isinstance(phase_planner_global, dict) and "error" in phase_planner_global : print(f"  PhasePlanner status: {phase_planner_global['error']}")


except Exception as e_init:
    print(f"WebApp Error: Failed to initialize global components: {e_init}")
    # app_config_global, digester_global, phase_planner_global will retain placeholder error states or be None.

@app.route('/', methods=['GET', 'POST'])
def index():
    spec_data_str = None
    plan_data_str = None
    error_message = None
    submitted_issue = None # Initialize to ensure it's always available for render_template

    if request.method == 'POST':
        submitted_issue = request.form.get('issue_text', '')

        # Check global components
        if phase_planner_global is None or (isinstance(phase_planner_global, dict) and 'error' in phase_planner_global):
            error_message = "PhasePlanner not initialized correctly. Check webapp startup logs."
        elif not hasattr(phase_planner_global, 'spec_normalizer') or \
             phase_planner_global.spec_normalizer is None or \
             (isinstance(phase_planner_global.spec_normalizer, dict) and 'error' in phase_planner_global.spec_normalizer): # Assuming spec_normalizer could also be an error dict
            error_message = "SpecNormalizer component within PhasePlanner not initialized correctly. Check webapp startup logs."
        else:
            try:
                # Call normalise_request
                spec_object = phase_planner_global.spec_normalizer.normalise_request(submitted_issue)

                if spec_object is None:
                    error_message = "Failed to normalize request into a Spec (SpecNormalizer returned None)."
                elif isinstance(spec_object, dict) and 'error' in spec_object: # Check if spec_object itself is an error placeholder
                    error_message = f"Error during spec normalization: {spec_object['error']}"
                else:
                    # Successfully got a Spec object
                    if hasattr(spec_object, 'model_dump_json'):
                        spec_data_str = spec_object.model_dump_json(indent=2)
                    elif hasattr(spec_object, 'json'): # Pydantic v1
                        spec_data_str = spec_object.json(indent=2)
                    else:
                        spec_data_str = json.dumps(spec_object, indent=2, default=str) # Fallback

                    # Call generate_plan_from_spec
                    plan_list = phase_planner_global.generate_plan_from_spec(spec_object)

                    if not plan_list: # Handles None or empty list
                        error_message = (error_message + "\n" if error_message else "") + "Failed to generate a plan from the Spec (generate_plan_from_spec returned empty or None)."
                    else:
                        phase_data_list = []
                        for phase in plan_list:
                            if hasattr(phase, 'model_dump_json'):
                                phase_data_list.append(phase.model_dump_json(indent=2))
                            elif hasattr(phase, 'json'): # Pydantic v1
                                phase_data_list.append(phase.json(indent=2))
                            else:
                                # Basic string representation for Phase objects if no Pydantic methods
                                phase_data_list.append(f"Phase: {getattr(phase, 'operation_name', 'Unknown Op')}\nTarget: {getattr(phase, 'target_file', 'N/A')}\nParams: {getattr(phase, 'parameters', {})}\nDescription: {getattr(phase, 'description', 'N/A')}")
                        plan_data_str = "\n\n---\n\n".join(phase_data_list)

            except Exception as e_process:
                print(f"WebApp Error: Error during POST processing: {e_process}")
                error_message = f"An unexpected error occurred: {str(e_process)}"

        return render_template('index.html',
                               submitted_issue=submitted_issue,
                               spec_data=spec_data_str,
                               plan_data=plan_data_str,
                               error_message=error_message)

    # For GET request
    return render_template('index.html', submitted_issue=None, spec_data=None, plan_data=None, error_message=None)

if __name__ == '__main__':
    # Consider adding host='0.0.0.0' for accessibility if running in a container/VM
    # Debug should be False in production
    app.run(debug=True, port=5001)
