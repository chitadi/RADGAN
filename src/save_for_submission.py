import os
import argparse
import yaml
import zipfile
import subprocess
import datetime
from pathlib import Path
import tempfile

# Resolve repository paths relative to this file so it works
# both inside Docker and when run locally.
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent

def _first_writable_dir(candidates):
    for d in candidates:
        p = Path(d)
        try:
            p.mkdir(parents=True, exist_ok=True)
            test_file = p / ".__write_test__"
            with open(test_file, "w") as f:
                f.write("ok")
            test_file.unlink(missing_ok=True)
            return str(p)
        except Exception:
            continue
    raise PermissionError("No writable directory found for logs.")

# Choose a writable base logs directory with sensible fallbacks
OUTPUT_DIR = _first_writable_dir([
    SRC_DIR / "logs",
    REPO_ROOT / "submission_logs",
    Path.home() / ".rase_submission_logs",
    Path(tempfile.gettempdir()) / "rase_submission_logs",
])

def _select_writable_output_dir(preferred_dir: Path):
    """Return a writable directory for the output zip, preferring the given dir."""
    candidates = [preferred_dir,
                  REPO_ROOT / "submission_artifacts",
                  Path.home() / ".rase_submission_artifacts",
                  Path(tempfile.gettempdir()) / "rase_submission_artifacts"]
    return Path(_first_writable_dir(candidates))

def run_cmd(cmd):
    """Run a shell command and return its output as text."""
    return subprocess.check_output(cmd, shell=True, text=True)

# Collect files/folders to zip
def save_pip_packages(timestamp, output_dir=f"{OUTPUT_DIR}/pip_freeze_logs"):
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a timestamped filename
    filename = f"requirements_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    # Run pip freeze and capture output
    with open(filepath, "w") as f:
        # Prefer pip-chill if available; fallback to pip freeze
        try:
            subprocess.run(["pip-chill"], stdout=f, text=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            subprocess.run(["pip", "freeze"], stdout=f, text=True, check=False)
    
    print(f"Pip dependents saved to {filepath}")

    return filepath

def save_list(filename, items):
    """Save a list of strings to file."""
    with open(filename, "w") as f:
        for item in sorted(items):
            f.write(item + "\n")
    print(f"Saved {filename} ({len(items)} entries)")

def get_installed_packages(timestamp, output_dir=f"{OUTPUT_DIR}/apt_mark_logs"):
    """Get dpkg package list from a Docker image."""
    cmd = f"apt-mark showmanual"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"aptmark_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    try:
        dpkg_list =  set(run_cmd(cmd).splitlines())
    except subprocess.CalledProcessError:
        dpkg_list = set()

    installed_pkgs = {line.split()[0] for line in dpkg_list if line.endswith("install")}

    save_list(filepath, installed_pkgs)

    return filepath


def save_best_model_path(best_model_path, timestamp, output_dir=f"{OUTPUT_DIR}/best_model_logs"):
    """Save the best model path into a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"best_model_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write(str(best_model_path) + "\n")
    
    print(f"Best model path saved to {filepath}")
    return filepath



def arcname_from_abs(abs_path: str) -> str:
    """
    Produce an archive name relative to the repo root so that
    paths appear as 'src/...', 'results/...', etc., regardless of
    whether the script runs in Docker or locally.
    """
    p = Path(abs_path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except Exception:
        # Fallback: just use the filename
        return p.name

def main():
    parser = argparse.ArgumentParser(description='Save for submission')

    parser.add_argument('-c', "--config", type=str, required=True, help='The config file for submission with contents to be saved in zip.')
    args = parser.parse_args()

    yaml_path = args.config
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Extract paths
    
    best_model_path = data.get("best_model_path")
    config_path = data.get("config")
    source_folder_path = str(SRC_DIR / "models")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Determine output zip path with permission-aware fallback
    preferred_dir = Path(os.path.dirname(yaml_path))
    zip_dir = _select_writable_output_dir(preferred_dir)
    if zip_dir.resolve() != preferred_dir.resolve():
        print(f"Output directory '{preferred_dir}' not writable; using fallback '{zip_dir}'.")
        output_zip = str(zip_dir / f"model_submission_{timestamp}.zip")
    else:
        output_zip = str(zip_dir / "model_submission.zip")
    pip_requirements = save_pip_packages(timestamp)
    dpkg_requirements = get_installed_packages(timestamp)
    

    
    files_to_zip = [best_model_path, config_path, source_folder_path, pip_requirements, dpkg_requirements]
    # Ensure best_model.yaml appears under 'src/' inside the zip
    files_to_zip_with_different_dir = [{"output_path": str(SRC_DIR / "best_model.yaml"), "item": yaml_path}]

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in files_to_zip:
            if not os.path.exists(item):
                print(f"Warning: {item} not found, skipping.")
                continue

            if os.path.isdir(item):
                # Add folder recursively, preserving '/...' structure (minus the leading '/')
                for root, _, files in os.walk(item):
                    for fn in files:
                        abs_path = os.path.join(root, fn)
                        zf.write(abs_path, arcname_from_abs(abs_path))
            else:
                # Single file
                zf.write(item, arcname_from_abs(os.path.abspath(item)))

        for item_dict in files_to_zip_with_different_dir:
            item = item_dict["item"]
            output_path = item_dict["output_path"]
            if not os.path.exists(item):
                print(f"Warning: {item} not found, skipping.")
                continue
            else:
                zf.write(item, arcname_from_abs(output_path)) 

    print(f"Created {output_zip}")
  


if __name__ == "__main__":

    main()
# print(config)


