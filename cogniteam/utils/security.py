import os
from typing import Optional
from urllib.parse import urlparse

from cogniteam.config.settings import settings


def is_path_safe(path: str) -> Optional[str]:
    if not path or not isinstance(path, str):
        return None
    norm_path = os.path.normpath(path)
    if ".." in norm_path.split(os.sep):
        return None
    try:
        project_root_abs = os.path.abspath(os.path.normpath(settings.project_root))
        if os.path.isabs(norm_path):
            if os.path.commonpath([norm_path, project_root_abs]) == project_root_abs:
                return norm_path
            return None
        prospective_full_path = os.path.abspath(os.path.join(project_root_abs, norm_path))
        if os.path.commonpath([prospective_full_path, project_root_abs]) == project_root_abs:
            return prospective_full_path
        return None
    except Exception:
        return None


def validate_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    try:
        result = urlparse(url)
        return all([result.scheme in ["http", "https"], result.netloc])
    except ValueError:
        return False


def is_command_blocked(command: str) -> bool:
    blocked_commands = {
        "rm", "del", "erase", "sudo", "runas", "chmod", "chown",
        "shutdown", "reboot", "halt", "format", "mkfs", "fdisk",
        "diskpart", "dd", "rundll32", "regsvr32",
        "net", "cipher",
    }
    import shlex
    try:
        command_parts = shlex.split(command)
        if not command_parts:
            return True
        first_word = command_parts[0].lower()
        if first_word in blocked_commands:
            return True
        if os.path.isabs(first_word) and not first_word.startswith(
            os.path.abspath(settings.project_root)
        ):
            return True
        return False
    except Exception:
        return True
