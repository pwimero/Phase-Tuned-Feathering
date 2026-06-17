from pathlib import Path
import subprocess
root = Path(__file__).resolve().parents[2]
subprocess.check_call(['python', str(root / 'scripts' / 'run_radiating_covariance_validation.py')])
