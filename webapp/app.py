from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    session,
)
from functools import wraps
from werkzeug.security import check_password_hash, generate_password_hash
import sys
import uuid
import threading
from pathlib import Path
import json  # For json.dumps as a fallback for displaying objects
import yaml
from src.learning.easyedit_helper import apply_easyedit
from src.mcp.mcp_manager import get_mcp_prompt

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
    from src.profiler.llm_interfacer import generate_custom_workflow
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

    def generate_custom_workflow(*_a, **_k):  # type: ignore
        return {"name": "custom", "steps": ["LLMCore", "DiffusionCore", "LLMCore"]}

    Spec = dict  # Placeholder type
    Phase = dict  # Placeholder type

app = Flask(__name__)
app.secret_key = "change-me"

# Globals populated by initialize_components()
app_config_global = {}
digester_global = None
phase_planner_global = None
initialized = False
job_status: dict[str, dict] = {}

# ---------------------------
# User management
# ---------------------------
USERS_FILE = PROJECT_ROOT / "config" / "users.json"


def load_users() -> dict:
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    # create default user
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    default = {"admin": generate_password_hash("admin")}
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(default, f, indent=2)
    return default


def save_users(users: dict) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login_route", next=request.path))
        return func(*args, **kwargs)

    return wrapper


def start_job(target, *args, **kwargs) -> str:
    """Start a background thread and return a job id."""
    job_id = str(uuid.uuid4())
    job_status[job_id] = {"status": "running", "message": ""}

    def wrapper():
        try:
            target(*args, **kwargs, job_id=job_id)
            job_status[job_id]["status"] = "complete"
        except Exception as e:  # pragma: no cover - threading
            job_status[job_id]["status"] = "error"
            job_status[job_id]["message"] = str(e)

    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
    return job_id


@app.route("/login", methods=["GET", "POST"])
def login_route():
    users = load_users()
    message = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username in users and check_password_hash(users[username], password):
            session["user"] = username
            return redirect(request.args.get("next") or url_for("dashboard_route"))
        message = "Invalid credentials"
    return render_template("login.html", message=message)


@app.route("/logout")
def logout_route():
    session.pop("user", None)
    return redirect(url_for("login_route"))


# -------------------------------------------------------------
# Environment management and selection
# -------------------------------------------------------------


@app.route("/projects", methods=["GET", "POST"])
def projects_route():
    envs = load_environments()
    message = None
    if request.method == "POST":
        action = request.form.get("action")
        if action == "select":
            name = request.form.get("env_name")
            for env in envs:
                if env.get("name") == name:
                    initialize_components(Path(env["path"]))
                    return redirect(url_for("dashboard_route"))
        elif action == "create":
            name = request.form.get("new_name")
            path = request.form.get("new_path")
            desc = request.form.get("new_desc")
            if name and path:
                envs.append({"name": name, "path": path, "description": desc})
                save_environments(envs)
                initialize_components(Path(path))
                return redirect(url_for("dashboard_route"))
            message = "Name and path are required"
        elif action == "delete":
            name = request.form.get("env_name")
            envs = [e for e in envs if e.get("name") != name]
            save_environments(envs)
        elif action == "rename":
            name = request.form.get("env_name")
            new_name = request.form.get("new_name")
            for env in envs:
                if env.get("name") == name and new_name:
                    env["name"] = new_name
            save_environments(envs)
    for env in envs:
        fp = Path(env.get("path", "")) / "style_fingerprint.json"
        env["profiled"] = fp.exists()
    return render_template("projects.html", environments=envs, message=message)


# Environments config
ENV_FILE = PROJECT_ROOT / "config" / "environments.json"


def load_environments() -> list[dict]:
    if ENV_FILE.exists():
        try:
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_environments(envs: list[dict]) -> None:
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        json.dump(envs, f, indent=2)


def initialize_components(
    project_root: Path, config_path: Path | None = None, verbose: bool = False
) -> None:
    """Initializes profiler, digester, and planner for the web UI."""
    global app_config_global, digester_global, phase_planner_global, initialized

    app_config_global = load_app_config(config_path)
    app_config_global["general"]["project_root"] = str(project_root)
    app_config_global["loaded_config_path"] = str(config_path or "config.yaml")
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


@app.route("/")
@login_required
def dashboard_route():
    if not initialized:
        return redirect(url_for("projects_route"))

    data_dir = Path(app_config_global.get("general", {}).get("data_dir", ".agent_data"))
    if not data_dir.is_absolute():
        project_root = Path(
            app_config_global.get("general", {}).get("project_root", ".")
        )
        data_dir = project_root / data_dir
    entries = load_success_memory(data_dir)
    recent_entries = entries[-5:]
    dataset_path = data_dir / "finetune_dataset.jsonl"
    dataset_entries = 0
    if dataset_path.exists():
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset_entries = sum(1 for _ in f)
        except Exception:
            dataset_entries = 0
    return render_template(
        "dashboard.html",
        recent_entries=recent_entries,
        memory_count=len(entries),
        dataset_entries=dataset_entries,
    )


@app.route("/issue", methods=["GET", "POST"])
@login_required
def issue_route():
    spec_data_str = None
    plan_data_str = None
    error_message = None
    submitted_issue = (
        None  # Initialize to ensure it's always available for render_template
    )

    if not initialized:
        return redirect(url_for("projects_route"))

    if request.method == "POST":
        submitted_issue = request.form.get("issue_text", "")
        workflow_selected = request.form.get("workflow", "orchestrator-workers")

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
                # Generate plan directly from goal text
                spec_object, plan_list = phase_planner_global.generate_plan_from_goal(
                    submitted_issue,
                    workflow_selected,
                )

                if hasattr(spec_object, "model_dump_json"):
                    spec_data_str = spec_object.model_dump_json(indent=2)
                elif hasattr(spec_object, "json"):
                    spec_data_str = spec_object.json(indent=2)
                else:
                    spec_data_str = json.dumps(spec_object, indent=2, default=str)

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
                        elif hasattr(phase, "json"):
                            phase_data_list.append(phase.json(indent=2))
                        else:
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
            workflow_selected=workflow_selected,
            custom_workflows=app_config_global.get("custom_workflows", []),
        )

    # For GET request
    return render_template(
        "index.html",
        submitted_issue=None,
        spec_data=None,
        plan_data=None,
        error_message=None,
        workflow_selected="orchestrator-workers",
        custom_workflows=app_config_global.get("custom_workflows", []),
    )


@app.route("/memory")
@login_required
def view_memory():
    """Display recent success memory entries with optional filtering."""
    if not initialized:
        return "Application not initialized", 503

    data_dir = Path(app_config_global.get("general", {}).get("data_dir", ".agent_data"))
    if not data_dir.is_absolute():
        project_root = Path(app_config_global.get("general", {}).get("project_root", "."))
        data_dir = project_root / data_dir

    query = request.args.get("query", "").lower()
    rating = request.args.get("rating")
    start = request.args.get("start")
    end = request.args.get("end")

    entries = load_success_memory(data_dir)
    filtered: list[dict] = []

    for e in entries:
        keep = True
        issue_text = str(e.get("spec_issue_description", "")).lower()
        if query and query not in issue_text:
            keep = False
        if rating:
            try:
                if int(rating) != int(e.get("user_rating", -1)):
                    keep = False
            except Exception:
                pass
        ts = e.get("timestamp_utc")
        if start and ts and ts < start:
            keep = False
        if end and ts and ts > end:
            keep = False
        if keep:
            filtered.append(e)

    recent_entries = filtered[-20:]
    return render_template(
        "memory.html",
        entries=recent_entries,
        query=query,
        rating=rating,
        start=start,
        end=end,
    )


@app.route("/memory/<int:index>")
@login_required
def memory_detail_route(index: int):
    """Display full details for a single memory entry."""
    if not initialized:
        return "Application not initialized", 503

    data_dir = Path(app_config_global.get("general", {}).get("data_dir", ".agent_data"))
    if not data_dir.is_absolute():
        project_root = Path(
            app_config_global.get("general", {}).get("project_root", ".")
        )
        data_dir = project_root / data_dir
    entries = load_success_memory(data_dir)
    if index < 0 or index >= len(entries):
        return redirect(url_for("view_memory"))
    entry = entries[index]
    return render_template("patch_detail.html", entry=entry)


@app.route("/apply_patch", methods=["POST"])
@login_required
def apply_patch_route():
    """Run full patch generation for an issue and redirect to memory."""
    if not initialized:
        return "Application not initialized", 503

    issue_text = request.form.get("issue_text", "")
    if not issue_text:
        return redirect(url_for("issue_route"))

    workflow = request.form.get("workflow", "orchestrator-workers")
    spec_prompt = get_mcp_prompt(app_config_global, workflow, "SpecNormalizer")
    def job(job_id=None):
        spec_obj_local = phase_planner_global.spec_normalizer.normalise_request(
            issue_text, spec_prompt
        )
        if spec_obj_local is None or (
            isinstance(spec_obj_local, dict) and "error" in spec_obj_local
        ):
            job_status[job_id]["status"] = "error"
            job_status[job_id]["message"] = "Spec normalization failed"
            return
        phase_planner_global.plan_phases(spec_obj_local, workflow_type=workflow)
        job_status[job_id]["redirect"] = url_for("view_memory")

    jid = start_job(job)
    return render_template("progress.html", title="Applying Patch", job_id=jid)


@app.route("/apply_plan", methods=["POST"])
@login_required
def apply_plan_route():
    """Execute a user-approved plan represented as JSON."""
    if not initialized:
        return "Application not initialized", 503

    issue_text = request.form.get("issue_text", "")
    plan_json = request.form.get("plan_json", "")
    workflow = request.form.get("workflow", "orchestrator-workers")
    if not issue_text or not plan_json:
        return redirect(url_for("issue_route"))

    spec_prompt = get_mcp_prompt(app_config_global, workflow, "SpecNormalizer")
    def job(job_id=None):
        spec_obj_local = phase_planner_global.spec_normalizer.normalise_request(
            issue_text, spec_prompt
        )
        if spec_obj_local is None or (
            isinstance(spec_obj_local, dict) and "error" in spec_obj_local
        ):
            job_status[job_id]["status"] = "error"
            job_status[job_id]["message"] = "Spec normalization failed"
            return

        try:
            plan_list_dicts = json.loads(plan_json)
            custom_plan = [Phase(**p) for p in plan_list_dicts]
        except Exception as e:
            job_status[job_id]["status"] = "error"
            job_status[job_id]["message"] = f"Invalid plan JSON: {e}"
            return

        cache_key = phase_planner_global._get_spec_cache_key(spec_obj_local)
        phase_planner_global.plan_cache[cache_key] = custom_plan
        phase_planner_global.plan_phases(spec_obj_local, workflow_type=workflow)
        job_status[job_id]["redirect"] = url_for("view_memory")

    jid = start_job(job)
    return render_template("progress.html", title="Applying Plan", job_id=jid)


@app.route("/feedback", methods=["POST"])
@login_required
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
@login_required
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
    def job(job_id=None):
        ok = build_finetune_dataset(data_dir, output_path, verbose=True)
        msg = f"Dataset written to {output_path}" if ok else "Failed to build dataset"
        job_status[job_id]["message"] = msg
        job_status[job_id]["redirect"] = url_for("train_route")

    jid = start_job(job)
    return render_template("progress.html", title="Preparing Dataset", job_id=jid)


@app.route("/train", methods=["GET", "POST"])
@login_required
def train_route():
    if not initialized:
        return "Application not initialized", 503

    message = None
    data_dir = Path(app_config_global.get("general", {}).get("data_dir", ".agent_data"))
    if not data_dir.is_absolute():
        project_root = Path(
            app_config_global.get("general", {}).get("project_root", ".")
        )
        data_dir = project_root / data_dir

    if request.method == "POST":
        action = request.form.get("action")
        if action == "Prepare Dataset":
            def job(job_id=None):
                output_path = data_dir / "finetune_dataset.jsonl"
                ok = build_finetune_dataset(data_dir, output_path, verbose=True)
                job_status[job_id]["message"] = (
                    f"Dataset written to {output_path}" if ok else "Failed to build dataset"
                )
                job_status[job_id]["redirect"] = url_for("train_route")

            jid = start_job(job)
            return render_template("progress.html", title="Preparing Dataset", job_id=jid)
        elif action == "Fine-tune LoRA":
            def job(job_id=None):
                try:
                    import subprocess
                    subprocess.run([
                        "python3",
                        "scripts/finetune_lora.py",
                        "--data-dir",
                        str(data_dir),
                    ], check=False)
                    job_status[job_id]["message"] = "LoRA fine-tuning started"
                except Exception as e:
                    job_status[job_id]["status"] = "error"
                    job_status[job_id]["message"] = str(e)
                job_status[job_id]["redirect"] = url_for("train_route")

            jid = start_job(job)
            return render_template("progress.html", title="Fine-tuning", job_id=jid)
        elif action == "Train Predictor":
            def job(job_id=None):
                try:
                    import subprocess
                    subprocess.run([
                        "python3",
                        "scripts/train_core_predictor.py",
                        "--data-dir",
                        str(data_dir),
                    ], check=False)
                    job_status[job_id]["message"] = "Predictor training started"
                except Exception as e:
                    job_status[job_id]["status"] = "error"
                    job_status[job_id]["message"] = str(e)
                job_status[job_id]["redirect"] = url_for("train_route")

            jid = start_job(job)
            return render_template("progress.html", title="Training Predictor", job_id=jid)
        elif action == "Run Full Pipeline":
            def job(job_id=None):
                try:
                    import subprocess
                    output_path = data_dir / "finetune_dataset.jsonl"
                    build_finetune_dataset(data_dir, output_path, verbose=True)
                    subprocess.run([
                        "python3",
                        "scripts/finetune_lora.py",
                        "--data-dir",
                        str(data_dir),
                    ], check=False)
                    subprocess.run([
                        "python3",
                        "scripts/train_core_predictor.py",
                        "--data-dir",
                        str(data_dir),
                    ], check=False)
                    job_status[job_id]["message"] = "Full training pipeline started"
                except Exception as e:
                    job_status[job_id]["status"] = "error"
                    job_status[job_id]["message"] = str(e)
                job_status[job_id]["redirect"] = url_for("train_route")

            jid = start_job(job)
            return render_template("progress.html", title="Training", job_id=jid)

    return render_template("train.html", message=message)


@app.route("/job_status/<job_id>")
def job_status_route(job_id: str):
    return jsonify(job_status.get(job_id, {"status": "unknown"}))


@app.route("/query_graph", methods=["GET"])
@login_required
def query_graph_route():
    """Query the knowledge graph and display results."""
    if not initialized:
        return "Application not initialized", 503

    src = request.args.get("src", "")
    relation = request.args.get("relation") or None
    depth = int(request.args.get("depth", "1"))

    result = {}
    if src:
        result = digester_global.query_knowledge_paths(src, relation, depth)

    return render_template(
        "graph.html",
        query_src=src,
        query_relation=relation,
        depth=depth,
        result=result,
        result_json=json.dumps(result),
    )


@app.route("/search_code", methods=["GET", "POST"])
@login_required
def search_code_route():
    """Search symbols using the digester's retriever."""
    if not initialized:
        return "Application not initialized", 503

    query = ""
    results = []
    if request.method == "POST":
        query = request.form.get("query", "")
        if query:
            retriever = getattr(digester_global, "symbol_retriever", None)
            if retriever and hasattr(retriever, "search_relevant_symbols_with_details"):
                results = retriever.search_relevant_symbols_with_details(
                    query, top_k=10
                )

    return render_template("search.html", query=query, results=results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch diffuseLLM web UI")
    parser.add_argument(
        "project_root", type=Path, nargs="?", help="Path to the codebase"
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Optional config YAML"
    )
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.project_root:
        initialize_components(args.project_root, args.config, args.verbose)

    app.run(debug=args.verbose, host="0.0.0.0", port=args.port)


@app.route("/config", methods=["GET", "POST"])
@login_required
def config_route():
    if not initialized:
        return "Application not initialized", 503
    config_path = Path(app_config_global.get("loaded_config_path", "config.yaml"))
    basic_fields = {
        "general__data_dir": "",
        "webui__port": "",
        "active_learning__enabled": "",
    }

    if request.method == "POST":
        mode = request.form.get("mode", "text")
        if mode == "basic":
            for key in basic_fields:
                section, field = key.split("__")
                val = request.form.get(key)
                app_config_global.setdefault(section, {})[field] = yaml.safe_load(val)
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(app_config_global, f)
            message = "Configuration updated"
        else:
            new_text = request.form.get("config_text", "")
            try:
                yaml.safe_load(new_text)
                with open(config_path, "w", encoding="utf-8") as f:
                    f.write(new_text)
                app_config_global.update(yaml.safe_load(new_text) or {})
                message = "Configuration updated"
            except Exception as e:
                message = f"Failed to update config: {e}"
            return render_template(
                "config.html",
                config_text=new_text,
                message=message,
                basic_fields=basic_fields,
            )
    text = (
        config_path.read_text(encoding="utf-8")
        if config_path.exists()
        else yaml.dump(app_config_global)
    )
    # populate field values
    for key in basic_fields:
        section, field = key.split("__")
        basic_fields[key] = app_config_global.get(section, {}).get(field, "")
    return render_template(
        "config.html",
        config_text=text,
        message=None,
        basic_fields=basic_fields,
    )


@app.route("/chat", methods=["GET", "POST"])
@login_required
def chat_route():
    if not initialized:
        return "Application not initialized", 503
    global chat_history
    if request.method == "POST":
        msg = request.form.get("message", "")
        if msg:
            chat_history.append({"role": "user", "text": msg})
            reply = "No response"
            try:
                if phase_planner_global and hasattr(phase_planner_global, "llm_core"):
                    llm_core = phase_planner_global.llm_core
                    reply, _ = llm_core.generate_scaffold_patch(
                        {"phase_description": msg}
                    ) or ("", None)
                    if not reply:
                        reply = "(LLM returned empty response)"
            except Exception as e:
                reply = f"Error: {e}"
            chat_history.append({"role": "assistant", "text": reply})
    return render_template("chat.html", history=chat_history[-20:])


@app.route("/easyedit", methods=["GET", "POST"])
@login_required
def easyedit_route():
    if not initialized:
        return "Application not initialized", 503
    message = None
    if request.method == "POST":
        instr = request.form.get("instruction", "")
        new_text = request.form.get("new_text", "")
        model_dir = Path(app_config_global.get("models", {}).get("agent_llm_gguf", ""))
        ok = apply_easyedit(model_dir, instr, new_text, verbose=True)
        message = "Edit applied" if ok else "EasyEdit unavailable or failed"
    return render_template("easyedit.html", message=message)


@app.route("/mcp", methods=["GET", "POST"])
@login_required
def mcp_route():
    if not initialized:
        return "Application not initialized", 503

    mcp_cfg = app_config_global.setdefault("mcp", {})
    tools = mcp_cfg.setdefault("tools", [])
    workflow_settings = mcp_cfg.setdefault("workflow_settings", {})

    message = None
    if request.method == "POST":
        action = request.form.get("action")
        if action == "add":
            name = request.form.get("tool_name", "").strip()
            prompt = request.form.get("tool_prompt", "")
            if name and prompt:
                tools.append({"name": name, "prompt": prompt})
                message = "Tool added"
        elif action == "save_settings":
            for key, val in request.form.items():
                if "__" in key:
                    wf, agent = key.split("__", 1)
                    workflow_settings.setdefault(wf, {})[agent] = val
            message = "Settings saved"
        with open(
            app_config_global.get("loaded_config_path", "config.yaml"),
            "w",
            encoding="utf-8",
        ) as f:
            yaml.safe_dump(app_config_global, f)

    return render_template(
        "mcp.html", tools=tools, workflow_settings=workflow_settings, message=message
    )


@app.route("/custom_workflow", methods=["GET", "POST"])
@login_required
def custom_workflow_route():
    if not initialized:
        return "Application not initialized", 503

    message = None
    workflow_json = None
    workflows = app_config_global.setdefault("custom_workflows", [])

    if request.method == "POST":
        name = request.form.get("workflow_name", "custom")
        prompt = request.form.get("prompt", "")
        model_path = app_config_global.get("models", {}).get("operations_llm_gguf")
        wf = generate_custom_workflow(prompt, model_path=model_path)
        if wf:
            if "name" not in wf:
                wf["name"] = name
            workflows.append(wf)
            workflow_json = json.dumps(wf, indent=2)
            message = "Workflow added"
            with open(
                app_config_global.get("loaded_config_path", "config.yaml"),
                "w",
                encoding="utf-8",
            ) as f:
                yaml.safe_dump(app_config_global, f)

    return render_template(
        "custom_workflow.html",
        message=message,
        workflow_json=workflow_json,
        workflows=workflows,
    )
