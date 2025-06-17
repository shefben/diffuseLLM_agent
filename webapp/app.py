from flask import Flask, render_template, request, redirect, url_for
import sys
from pathlib import Path
import json  # For json.dumps as a fallback for displaying objects

# Add project root to sys.path to allow imports from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Conditional imports for project modules, handle if not found during early setup
try:
    from src.utils.config_loader import load_app_config
    from src.digester.repository_digester import RepositoryDigester
    from src.planner.phase_planner import PhasePlanner
    from src.planner.phase_model import Phase
    from src.planner.spec_model import Spec
    from src.profiler.orchestrator import run_phase1_style_profiling_pipeline
    from src.utils.memory_logger import load_success_memory
    from src.learning.ft_dataset import build_finetune_dataset
except ImportError as e:
    print(
        f"Error importing project modules in webapp/app.py: {e}. Ensure PROJECT_ROOT is correct and src modules are accessible."
    )

    def load_app_config(*_a, **_k):
        return {"error": "load_app_config failed"}

    def RepositoryDigester(*_a, **_k):  # type: ignore
        return {"error": "RepositoryDigester failed"}

    def PhasePlanner(*_a, **_k):  # type: ignore
        return {"error": "PhasePlanner failed"}

    def run_phase1_style_profiling_pipeline(*_a, **_k):  # type: ignore
        return {"error": "profiling failed"}

    def build_finetune_dataset(*_a, **_k):  # type: ignore
        return False

    Spec = dict  # Placeholder type
    Phase = dict  # Placeholder type

app = Flask(__name__)

# Globals populated by initialize_components()
app_config_global = {}
digester_global = None
phase_planner_global = None
initialized = False


def initialize_components(
    project_root: Path, config_path: Path | None = None, verbose: bool = False
) -> None:
    """Initializes profiler, digester, and planner for the web UI."""
    global app_config_global, digester_global, phase_planner_global, initialized

    app_config_global = load_app_config(config_path)
    app_config_global["general"]["project_root"] = str(project_root)
    if verbose:
        app_config_global["general"]["verbose"] = True

    # Run style profiling pipeline to generate configs and naming DB
    run_phase1_style_profiling_pipeline(project_root, app_config_global)

    digester_global = RepositoryDigester(
        repo_path=project_root, app_config=app_config_global
    )
    digester_global.digest_repository()

    phase_planner_global = PhasePlanner(
        project_root_path=project_root,
        app_config=app_config_global,
        digester=digester_global,
    )

    initialized = True


@app.route("/", methods=["GET", "POST"])
def index():
    spec_data_str = None
    plan_data_str = None
    error_message = None
    submitted_issue = (
        None  # Initialize to ensure it's always available for render_template
    )

    if not initialized:
        error_message = (
            "Application not initialized. Please run initialize_components()."
        )
        return render_template(
            "index.html",
            submitted_issue=None,
            spec_data=None,
            plan_data=None,
            error_message=error_message,
        )

    if request.method == "POST":
        submitted_issue = request.form.get("issue_text", "")

        # Check global components
        if phase_planner_global is None or (
            isinstance(phase_planner_global, dict) and "error" in phase_planner_global
        ):
            error_message = (
                "PhasePlanner not initialized correctly. Check webapp startup logs."
            )
        elif (
            not hasattr(phase_planner_global, "spec_normalizer")
            or phase_planner_global.spec_normalizer is None
            or (
                isinstance(phase_planner_global.spec_normalizer, dict)
                and "error" in phase_planner_global.spec_normalizer
            )
        ):  # Assuming spec_normalizer could also be an error dict
            error_message = "SpecNormalizer component within PhasePlanner not initialized correctly. Check webapp startup logs."
        else:
            try:
                # Call normalise_request
                spec_object = phase_planner_global.spec_normalizer.normalise_request(
                    submitted_issue
                )

                if spec_object is None:
                    error_message = "Failed to normalize request into a Spec (SpecNormalizer returned None)."
                elif (
                    isinstance(spec_object, dict) and "error" in spec_object
                ):  # Check if spec_object itself is an error placeholder
                    error_message = (
                        f"Error during spec normalization: {spec_object['error']}"
                    )
                else:
                    # Successfully got a Spec object
                    if hasattr(spec_object, "model_dump_json"):
                        spec_data_str = spec_object.model_dump_json(indent=2)
                    elif hasattr(spec_object, "json"):  # Pydantic v1
                        spec_data_str = spec_object.json(indent=2)
                    else:
                        spec_data_str = json.dumps(
                            spec_object, indent=2, default=str
                        )  # Fallback

                    # Call generate_plan_from_spec
                    plan_list = phase_planner_global.generate_plan_from_spec(
                        spec_object
                    )

                    if not plan_list:  # Handles None or empty list
                        error_message = (
                            (error_message + "\n" if error_message else "")
                            + "Failed to generate a plan from the Spec (generate_plan_from_spec returned empty or None)."
                        )
                    else:
                        phase_data_list = []
                        for phase in plan_list:
                            if hasattr(phase, "model_dump_json"):
                                phase_data_list.append(phase.model_dump_json(indent=2))
                            elif hasattr(phase, "json"):  # Pydantic v1
                                phase_data_list.append(phase.json(indent=2))
                            else:
                                # Basic string representation for Phase objects if no Pydantic methods
                                phase_data_list.append(
                                    f"Phase: {getattr(phase, 'operation_name', 'Unknown Op')}\nTarget: {getattr(phase, 'target_file', 'N/A')}\nParams: {getattr(phase, 'parameters', {})}\nDescription: {getattr(phase, 'description', 'N/A')}"
                                )
                        plan_data_str = "\n\n---\n\n".join(phase_data_list)

            except Exception as e_process:
                print(f"WebApp Error: Error during POST processing: {e_process}")
                error_message = f"An unexpected error occurred: {str(e_process)}"

        return render_template(
            "index.html",
            submitted_issue=submitted_issue,
            spec_data=spec_data_str,
            plan_data=plan_data_str,
            error_message=error_message,
        )

    # For GET request
    return render_template(
        "index.html",
        submitted_issue=None,
        spec_data=None,
        plan_data=None,
        error_message=None,
    )


@app.route("/memory")
def view_memory():
    """Display recent success memory entries."""
    if not initialized:
        return "Application not initialized", 503

    data_dir = Path(app_config_global.get("general", {}).get("data_dir", ".agent_data"))
    if not data_dir.is_absolute():
        project_root = Path(
            app_config_global.get("general", {}).get("project_root", ".")
        )
        data_dir = project_root / data_dir
    entries = load_success_memory(data_dir)
    recent_entries = entries[-20:]
    return render_template("memory.html", entries=recent_entries)


@app.route("/apply_patch", methods=["POST"])
def apply_patch_route():
    """Run full patch generation for an issue and redirect to memory."""
    if not initialized:
        return "Application not initialized", 503

    issue_text = request.form.get("issue_text", "")
    if not issue_text:
        return redirect(url_for("index"))

    spec_obj = phase_planner_global.spec_normalizer.normalise_request(issue_text)
    if spec_obj is None or isinstance(spec_obj, dict) and "error" in spec_obj:
        return render_template(
            "index.html",
            submitted_issue=issue_text,
            spec_data=None,
            plan_data=None,
            error_message="Spec normalization failed",
        )

    phase_planner_global.plan_phases(spec_obj)
    return redirect(url_for("view_memory"))


@app.route("/apply_plan", methods=["POST"])
def apply_plan_route():
    """Execute a user-approved plan represented as JSON."""
    if not initialized:
        return "Application not initialized", 503

    issue_text = request.form.get("issue_text", "")
    plan_json = request.form.get("plan_json", "")
    if not issue_text or not plan_json:
        return redirect(url_for("index"))

    spec_obj = phase_planner_global.spec_normalizer.normalise_request(issue_text)
    if spec_obj is None or isinstance(spec_obj, dict) and "error" in spec_obj:
        return render_template(
            "index.html",
            submitted_issue=issue_text,
            spec_data=None,
            plan_data=None,
            error_message="Spec normalization failed",
        )

    try:
        plan_list_dicts = json.loads(plan_json)
        custom_plan = [Phase(**p) for p in plan_list_dicts]
    except Exception as e:
        return render_template(
            "index.html",
            submitted_issue=issue_text,
            spec_data=None,
            plan_data=plan_json,
            error_message=f"Invalid plan JSON: {e}",
        )

    cache_key = phase_planner_global._get_spec_cache_key(spec_obj)
    phase_planner_global.plan_cache[cache_key] = custom_plan
    phase_planner_global.plan_phases(spec_obj)
    return redirect(url_for("view_memory"))


@app.route("/feedback", methods=["POST"])
def feedback_route():
    if not initialized:
        return "Application not initialized", 503

    timestamp = request.form.get("timestamp")
    rating = request.form.get("rating")
    if not timestamp or not rating:
        return redirect(url_for("view_memory"))

    data_dir = Path(app_config_global.get("general", {}).get("data_dir", ".agent_data"))
    if not data_dir.is_absolute():
        project_root = Path(
            app_config_global.get("general", {}).get("project_root", ".")
        )
        data_dir = project_root / data_dir

    entries = load_success_memory(data_dir)
    for entry in entries:
        if entry.get("timestamp_utc") == timestamp:
            entry["user_rating"] = int(rating)

    log_path = data_dir / "success_memory.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    return redirect(url_for("view_memory"))


@app.route("/prepare_dataset")
def prepare_dataset_route():
    """Generate a fine-tuning dataset from success memory."""
    if not initialized:
        return "Application not initialized", 503

    data_dir = Path(app_config_global.get("general", {}).get("data_dir", ".agent_data"))
    if not data_dir.is_absolute():
        project_root = Path(
            app_config_global.get("general", {}).get("project_root", ".")
        )
        data_dir = project_root / data_dir

    output_path = data_dir / "finetune_dataset.jsonl"
    ok = build_finetune_dataset(data_dir, output_path, verbose=True)
    message = f"Dataset written to {output_path}" if ok else "Failed to build dataset"
    return render_template("dataset.html", message=message)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch diffuseLLM web UI")
    parser.add_argument("project_root", type=Path, help="Path to the codebase")
    parser.add_argument(
        "--config", type=Path, default=None, help="Optional config YAML"
    )
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    initialize_components(args.project_root, args.config, args.verbose)

    app.run(debug=args.verbose, host="0.0.0.0", port=args.port)
