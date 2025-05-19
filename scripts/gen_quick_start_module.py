#!/usr/bin/env python3
"""
Generates quick start module for https://pytorch.org/get-started/locally/ page
If called from update-quick-start-module.yml workflow (--autogenerate parameter set)
Will output new quick-start-module.js, and new published_version.json file
based on the current release matrix.
If called standalone will generate quick-start-module.js from existing
published_version.json file
"""

import argparse
import copy
import json
from enum import Enum
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).parent.parent


class OperatingSystem(Enum):
    LINUX: str = "linux"
    WINDOWS: str = "windows"
    WINDOWS_ARM64: str = "windows-arm64"
    MACOS: str = "macos"


PRE_CXX11_ABI = "pre-cxx11"
CXX11_ABI = "cxx11-abi"
DEBUG = "debug"
RELEASE = "release"
DEFAULT = "default"
ENABLE = "enable"
DISABLE = "disable"
MACOS = "macos"

# Mapping json to release matrix default values
acc_arch_ver_default = {
    "nightly": {
        "accnone": ("cpu", ""),
        "cuda.x": ("cuda", "11.8"),
        "cuda.y": ("cuda", "12.1"),
        "cuda.z": ("cuda", "12.4"),
        "rocm5.x": ("rocm", "6.0"),
    },
    "release": {
        "accnone": ("cpu", ""),
        "cuda.x": ("cuda", "11.8"),
        "cuda.y": ("cuda", "12.1"),
        "cuda.z": ("cuda", "12.4"),
        "rocm5.x": ("rocm", "6.0"),
    },
}

# Initialize arch version to default values
# these default values will be overwritten by
# extracted values from the release marix
acc_arch_ver_map = acc_arch_ver_default

LIBTORCH_DWNL_INSTR = {
    PRE_CXX11_ABI: "Download here (Pre-cxx11 ABI):",
    CXX11_ABI: "Download here (cxx11 ABI):",
    RELEASE: "Download here (Release version):",
    DEBUG: "Download here (Debug version):",
    MACOS: "Download arm64 libtorch here (ROCm and CUDA are not supported):",
}


def load_json_from_basedir(filename: str):
    try:
        with open(BASE_DIR / filename) as fptr:
            return json.load(fptr)
    except FileNotFoundError as exc:
        raise ImportError(f"File {filename} not found error: {exc.strerror}") from exc
    except json.JSONDecodeError as exc:
        raise ImportError(f"Invalid JSON {filename}") from exc


def read_published_versions():
    return load_json_from_basedir("published_versions.json")


def write_published_versions(versions):
    with open(BASE_DIR / "published_versions.json", "w") as outfile:
        json.dump(versions, outfile, indent=2)

# Create releases JSON for PyTorch website.
# Matrix is being used to populate config data for 
# the "Start Locally" installation options table.
def write_releases_file(matrix):
    with open(BASE_DIR / "releases.json", "w") as outfile:
        json.dump(matrix, outfile, indent=2)

def read_matrix_for_os(osys: OperatingSystem, channel: str):
    jsonfile = load_json_from_basedir(f"{osys.value}_{channel}_matrix.json")
    return jsonfile["include"]


def read_quick_start_module_template():
    with open(BASE_DIR / "_includes" / "quick-start-module.js") as fptr:
        return fptr.read()


def get_package_type(pkg_key: str, os_key: OperatingSystem) -> str:
    if pkg_key != "pip":
        return pkg_key
    return "manywheel" if os_key == OperatingSystem.LINUX.value else "wheel"


def get_gpu_info(acc_key, instr, acc_arch_map):
    gpu_arch_type, gpu_arch_version = acc_arch_map[acc_key]
    if DEFAULT in instr:
        gpu_arch_type, gpu_arch_version = acc_arch_map["accnone"]
    return (gpu_arch_type, gpu_arch_version)


# This method is used for generating new published_versions.json file
# It will modify versions json object with installation instructions
# Provided by generate install matrix Github Workflow, stored in release_matrix
# json object.
def update_versions(versions, release_matrix, release_version):
    """
    Updates the versions JSON object with installation instructions
    based on the release matrix.
    """
    version = "preview"
    template = "preview"
    acc_arch_map = acc_arch_ver_map[release_version]

    if release_version != "nightly":
        version = release_matrix[OperatingSystem.LINUX.value][0]["stable_version"]
        if version not in versions["versions"]:
            versions["versions"][version] = copy.deepcopy(
                versions["versions"][template]
            )
            versions["latest_stable"] = version

    # Perform update of the JSON file from the release matrix
    for os_key, os_vers in versions["versions"][version].items():
        for pkg_key, pkg_vers in os_vers.items():
            for acc_key, instr in pkg_vers.items():
                package_type = get_package_type(pkg_key, os_key)
                gpu_arch_type, gpu_arch_version = get_gpu_info(
                    acc_key, instr, acc_arch_map
                )

                pkg_arch_matrix = [
                    x
                    for x in release_matrix[os_key]
                    if (x["package_type"], x["gpu_arch_type"], x["gpu_arch_version"])
                    == (package_type, gpu_arch_type, gpu_arch_version)
                ]
                if pkg_arch_matrix:
                    if package_type != "libtorch":
                        instr["command"] = pkg_arch_matrix[0]["installation"]
                    else:
                        if os_key == OperatingSystem.LINUX.value:
                            rel_entry_dict = {
                                x["devtoolset"]: x["installation"]
                                for x in pkg_arch_matrix
                                if x["libtorch_variant"] == "shared-with-deps"
                            }
                            if instr["versions"] is not None:
                                for ver in [CXX11_ABI, PRE_CXX11_ABI]:
                                    # Temporarily remove setting pre-cxx11-abi. For Release 2.7 we
                                    # should remove pre-cxx11-abi completely.
                                    if ver == PRE_CXX11_ABI:
                                        continue
                                    else:
                                        instr["versions"][LIBTORCH_DWNL_INSTR[ver]] = (
                                            rel_entry_dict[ver]
                                        )

                        elif os_key == OperatingSystem.WINDOWS.value:
                            if instr["versions"] is None:
                                instr["versions"] = {}
                            found = False
                            for entry in pkg_arch_matrix:
                                found = True
                                if isinstance(entry["installation"], dict):
                                    for config in [RELEASE, DEBUG]:
                                        if config in entry["installation"]:
                                            instr["versions"][LIBTORCH_DWNL_INSTR[config]] = entry["installation"][config]
                                    instr["versions"] = flatten_libtorch_versions(instr["versions"])
                                else:
                                    for config in [RELEASE, DEBUG]:
                                        if entry["libtorch_config"] == config:
                                            instr["versions"][LIBTORCH_DWNL_INSTR[config]] = entry["installation"]
                            if not found:
                                print(f"NO MATCH for {os_key=} {pkg_key=} {acc_key=} {package_type=} {gpu_arch_type=} {gpu_arch_version=}")

                        elif os_key == OperatingSystem.MACOS.value:
                            if instr["versions"] is not None:
                                instr["versions"][LIBTORCH_DWNL_INSTR[MACOS]] = (
                                    pkg_arch_matrix[0]["installation"]
                                )


# This method is used for generating new quick-start-module.js
# from the versions json object
def gen_install_matrix(versions) -> Dict[str, str]:
    """
    Generates the installation matrix for the quick-start module.
    Handles combined Windows x64 and ARM64 JSON objects by including both versions.
    """
    result = {}
    version_map = {
        "preview": "preview",
        "stable": versions["latest_stable"],
    }

    for ver, ver_key in version_map.items():
        for os_key, os_vers in versions["versions"][ver_key].items():
            for pkg_key, pkg_vers in os_vers.items():
                for acc_key, instr in pkg_vers.items():
                    extra_key = "python" if pkg_key != "libtorch" else "cplusplus"
                    key = f"{ver},{pkg_key},{os_key},{acc_key},{extra_key}"
                    note = instr["note"]
                    lines = [note] if note is not None else []

                    if pkg_key == "libtorch":
                        ivers = instr["versions"]
                        if ivers is not None:
                            # Handle combined Windows x64 and ARM64 versions
                            if os_key == OperatingSystem.WINDOWS.value:
                                if "x64" in ivers and "arm64" in ivers:
                                    lines.append(
                                        f"x64: <a href='{ivers['x64']}'>{ivers['x64']}</a>"
                                    )
                                    lines.append(
                                        f"ARM64: <a href='{ivers['arm64']}'>{ivers['arm64']}</a>"
                                    )
                                else:
                                    # Fallback to default behavior if not combined
                                    lines += [
                                        f"{lab}: <a href='{val}'>{val}</a>"
                                        for (lab, val) in ivers.items()
                                    ]
                            else:
                                # Default behavior for other OS
                                lines += [
                                    f"{lab}: <a href='{val}'>{val}</a>"
                                    for (lab, val) in ivers.items()
                                ]
                    else:
                        command = instr["command"]
                        if command is not None:
                            lines.append(command)

                    # Ensure all items in `lines` are strings
                    lines = [str(line) for line in lines]
                    result[key] = "<br />".join(lines)

    return result


# This method is used for extracting two latest verisons of cuda and
# last verion of rocm. It will modify the acc_arch_ver_map object used
# to update getting started page.
def extract_arch_ver_map(release_matrix):
    def gen_ver_list(chan, gpu_arch_type):
        return {
            x["desired_cuda"]: x["gpu_arch_version"]
            for x in release_matrix[chan]["linux"]
            if x["gpu_arch_type"] == gpu_arch_type
        }

    for chan in ("nightly", "release"):
        cuda_ver_list = gen_ver_list(chan, "cuda")
        rocm_ver_list = gen_ver_list(chan, "rocm")
        cuda_list = sorted(cuda_ver_list.values())
        acc_arch_ver_map[chan]["rocm5.x"] = ("rocm", max(rocm_ver_list.values()))
        for cuda_ver, label in zip(cuda_list, ["cuda.x", "cuda.y", "cuda.z"]):
            acc_arch_ver_map[chan][label] = ("cuda", cuda_ver)

def merge_windows_and_arm64(windows_data, windows_arm64_data):
    """
    Merges windows and windows-arm64 JSON objects into a single object
    with combined validation runners and installation instructions.
    """
    merged_data = []

    # Create a dictionary to match objects by their unique attributes
    for win_obj in windows_data:
        # Find the corresponding arm64 object
        matching_arm64 = next(
            (arm64_obj for arm64_obj in windows_arm64_data if arm64_obj["build_name"] == win_obj["build_name"]),
            None
        )

        if matching_arm64:
            # Combine validation runners and installation links
            combined_obj = win_obj.copy()
            combined_obj["validation_runner"] = {
                "x64": win_obj["validation_runner"],
                "arm64": matching_arm64["validation_runner"]
            }
            combined_obj["installation"] = {
                "x64": win_obj["installation"],
                "arm64": matching_arm64["installation"]
            }
            merged_data.append(combined_obj)
        else:
            # If no matching arm64 object, keep the original windows object
            merged_data.append(win_obj)

    # Add any arm64 objects that don't have a matching windows object
    for arm64_obj in windows_arm64_data:
        if not any(win_obj["build_name"] == arm64_obj["build_name"] for win_obj in windows_data):
            combined_obj = arm64_obj.copy()
            combined_obj["validation_runner"] = {
                "x64": None,
                "arm64": arm64_obj["validation_runner"]
            }
            combined_obj["installation"] = {
                "x64": None,
                "arm64": arm64_obj["installation"]
            }
            merged_data.append(combined_obj)

    return merged_data

def flatten_libtorch_versions(versions_dict):
    """
    Flattens nested x64/arm64 libtorch version dicts into separate keys.
    """
    flat = {}
    for label, value in versions_dict.items():
        if isinstance(value, dict):
            for arch, url in value.items():
                if url:  # Only add if url is not None
                    # Remove trailing colon if present, then add arch
                    new_label = label.rstrip(":") + f" {arch}"
                    flat[new_label] = url
        else:
            flat[label] = value
    return flat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autogenerate", dest="autogenerate", action="store_true")
    parser.set_defaults(autogenerate=True)

    options = parser.parse_args()
    versions = read_published_versions()
    print(f"Versions: {versions}")

    if options.autogenerate:
        release_matrix = {}
        print('GOT HERE 3 ')
        for val in ("nightly", "release"):
            release_matrix[val] = {}
            for osys in OperatingSystem:
                if osys == OperatingSystem.WINDOWS_ARM64:
                    # Read windows and windows-arm64 data
                    windows_data = release_matrix[val].get(OperatingSystem.WINDOWS.value, [])
                    windows_arm64_data = read_matrix_for_os(osys, val)

                    # Merge windows and windows-arm64 data
                    release_matrix[val][OperatingSystem.WINDOWS.value] = merge_windows_and_arm64(windows_data, windows_arm64_data)
                else:
                    # Regular behavior for other operating systems
                    release_matrix[val][osys.value] = read_matrix_for_os(osys, val)

        write_releases_file(release_matrix)

        extract_arch_ver_map(release_matrix)
        print(release_matrix["nightly"])
        for val in ("nightly", "release"):
            update_versions(versions, release_matrix[val], val)

        write_published_versions(versions)

    template = read_quick_start_module_template()
    versions_str = json.dumps(gen_install_matrix(versions))
    template = template.replace("{{ installMatrix }}", versions_str)
    template = template.replace(
        "{{ VERSION }}", f"\"Stable ({versions['latest_stable']})\""
    )
    print(template.replace("{{ ACC ARCH MAP }}", json.dumps(acc_arch_ver_map)))


if __name__ == "__main__":
    main()
