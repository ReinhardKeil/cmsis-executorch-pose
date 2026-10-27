"""Load the version-matched SDS VSI implementation from ARM::SDS."""
import os
import sys
from pathlib import Path

base = Path(os.environ["CMSIS_PACK_ROOT"]) / "ARM" / "SDS" / "3.1.0" / "template" / "Board" / "Corstone-300" / "vsi" / "python"
sys.path.insert(0, str(base))
exec((base / "arm_vsi3.py").read_text(encoding="utf-8"), globals())
