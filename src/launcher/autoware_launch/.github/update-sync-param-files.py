import argparse
import dataclasses
from pathlib import Path
from typing import List
from typing import Optional

import git

"""
This module updates `sync-param-files.yaml` based on the launch parameter files in `autoware.universe`.
"""

REPO_NAME = "autowarefoundation/autoware.universe"
REPO_URL = f"https://github.com/{REPO_NAME}.git"
CLONE_PATH = Path("/tmp/autoware.universe")


@dataclasses.dataclass
class FileSyncConfig:
    source: str
    dest: str
    replace: Optional[bool] = None
    delete_orphaned: Optional[bool] = None
    pre_commands: Optional[str] = None
    post_commands: Optional[str] = None


def create_tier4_launch_sync_configs(tier4_launch_package_path: Path) -> List[FileSyncConfig]:
    launch_package_name = tier4_launch_package_path.name
    launch_config_path = tier4_launch_package_path / "config"

    sync_configs = []
    for param_file_path in tier4_launch_package_path.glob("config/**/*.param.yaml"):
        relative_param_file_path = param_file_path.relative_to(launch_config_path)

        source = param_file_path.relative_to(CLONE_PATH)
        dest = Path("autoware_launch/config") / launch_package_name / relative_param_file_path

        sync_configs.append(FileSyncConfig(str(source), str(dest)))

    return sync_configs


def dump_sync_config(section_name: str, sync_configs: List[FileSyncConfig]) -> List[str]:
    indent = 4 * " "
    lines = [f"{indent}# {section_name}\n"]
    for sync_config in sync_configs:
        lines.append(f"{indent}- source: {sync_config.source}\n")
        lines.append(f"{indent}  dest: {sync_config.dest}\n")
    lines.append("\n")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sync_param_files_path", type=Path, help="path to sync-param-files.yaml")
    args = parser.parse_args()

    # Clone Autoware
    if not CLONE_PATH.exists():
        git.Repo.clone_from(REPO_URL, CLONE_PATH)

    # Create sync config for tier4_*_launch
    tier4_launch_package_paths = sorted(
        CLONE_PATH.glob("launch/tier4_*_launch"), key=lambda p: p.name
    )
    tier4_launch_sync_configs_map = {
        p.name: create_tier4_launch_sync_configs(p) for p in tier4_launch_package_paths
    }

    # Create sync-param-files.yaml
    with open(args.sync_param_files_path, "w") as f:
        f.write(f"- repository: {REPO_NAME}\n")
        f.write("  files:\n")
        for section_name, sync_config in tier4_launch_sync_configs_map.items():
            f.writelines(dump_sync_config(section_name, sync_config))


if __name__ == "__main__":
    main()
