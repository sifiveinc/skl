#!/usr/bin/env python3
# Copyright (c) 2026 SiFive, Inc. All rights reserved.
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# SPDX-License-Identifier: MIT

"""
Script to check for SiFive copyright and MIT license headers in source files.
This script verifies that each relevant file contains the following header:

    Copyright (c) {year_str} SiFive, Inc. All rights reserved.
    Licensed under the MIT License.
    See LICENSE file in the project root for full license information.
    SPDX-License-Identifier: MIT

Usage:
    # Check specific files
    ./check_license_headers.py file1.c file2.h

    # Show verbose output (lists all valid files)
    ./check_license_headers.py --verbose

Exit codes:
    0 - All files have proper headers
    1 - One or more files are missing headers
"""

import os
import sys
import re
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Set

# Comment styles for different file types
# The script defaults to '' for types not listed here
FILE_TYPES = {
    # C/C++ style comments
    '.c': '//',
    '.h': '//',
    '.cpp': '//',
    '.hpp': '//',
    # Python style comments
    '.py': '#',
    # CMake style comments
    '.cmake': '#',
    'CMakeLists.txt': '#',
}

# Exclude files matching these patterns
# The script looks for a match at the beginning of a file's path relative to
# the root of the repo.
EXCLUDE_PATTERNS = [
    r'.*\.md$',
    '.clang-tidy',
    '.clang-format',
    r'.*Doxyfile$',
    '.github/',
    '.gitignore',
    'LICENSE.txt',
]

def should_check_file(filepath: Path, repo_root: Path) -> bool:
    """Determine if a file should be checked."""
    rel_path = filepath.relative_to(repo_root)
    rel_path_str = str(rel_path)
    for pattern in EXCLUDE_PATTERNS:
        if re.match(pattern, rel_path_str):
            return False
    return True


def get_comment_style(filepath: Path) -> str:
    """Get the comment style for a file."""
    if filepath.name == 'CMakeLists.txt':
        return FILE_TYPES[filepath.name]
    return FILE_TYPES.get(filepath.suffix, '')


def get_copyright_year(filepath: Path) -> str:
    year_from_date = lambda date: date.split(b'-')[0].decode('utf-8')

    date_created = subprocess.check_output(['git', 'log', '--pretty=format:%cI', '--follow', filepath]).rsplit(b'\n', 1)[-1]
    date_modified = subprocess.check_output(['git', 'log', '-1', '--pretty=format:%cI', filepath])
    year_created = year_from_date(date_created)
    year_modified = year_from_date(date_modified)

    if year_created != year_modified:
        year_str = f"{year_created}-{year_modified}"
    else:
        year_str = f"{year_modified}"

    return year_str


def generate_license_pattern(filepath: Path) -> str:
    """Generate the appropriate license header pattern for a file."""
    comment_prefix = get_comment_style(filepath)
    year_str = get_copyright_year(filepath)
    header_lines = [
        rf"{comment_prefix} Copyright \(c\) {year_str} SiFive, Inc\. All rights reserved\.(\r)?\n",
        rf"{comment_prefix} Licensed under the MIT License\.(\r)?\n",
        rf"{comment_prefix} See LICENSE file in the project root for full license information\.(\r)?\n",
        rf"{comment_prefix} SPDX-License-Identifier: MIT(\r)?\n",
        ""
    ]
    return ''.join(header_lines)


def check_file_header(filepath: Path) -> Tuple[bool, List[str]]:
    """
    Check if a file has the required copyright and license header.

    Returns:
        Tuple of (has_valid_header, list_of_issues)
    """
    issues = []

    try:
        with open(filepath, 'r') as f:
            # Read first 6 lines (header should be in this range)
            header_lines = []
            for i, line in enumerate(f):
                if i >= 6:
                    break
                header_lines.append(line)
            header_text = ''.join(header_lines)

            license_pattern = generate_license_pattern(filepath)
            if not re.search(license_pattern, header_text):
                issues.append("Missing or incorrect header. Should be: " + license_pattern)

    except Exception as e:
        issues.append(f"Error reading file: {e}")

    return (len(issues) == 0, issues)


def main():
    """Main function to check given files in the repository."""
    parser = argparse.ArgumentParser(
        description="Check for SiFive copyright and MIT license headers in source files.",
    )
    parser.add_argument(
        'files',
        nargs='*',
        help='Files to check'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show verbose output including all checked files'
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.resolve()

    files_checked = 0
    files_skipped = 0
    files_with_issues = 0
    files_ok = []
    all_issues = []

    files_to_check = [Path(f).resolve() for f in args.files]
    print(f"Checking {len(files_to_check)} specified file(s)")
    print("=" * 80)

    for filepath in files_to_check:
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue

        if not filepath.is_file():
            print(f"Warning: Not a file: {filepath}")
            continue

        try:
            rel_path = filepath.relative_to(repo_root)
        except ValueError:
            print(f"Warning: File outside repository: {filepath}")
            continue

        if not should_check_file(filepath, repo_root):
            files_skipped += 1
            print(f"Skipping {filepath}")
            continue

        files_checked += 1

        has_valid_header, issues = check_file_header(filepath)

        if has_valid_header:
            files_ok.append(rel_path)
        else:
            files_with_issues += 1
            all_issues.append((rel_path, issues))

    # Print results
    print(f"\nFiles checked: {files_checked}")
    print(f"Files skipped: {files_skipped}")
    print(f"Files OK: {len(files_ok)}")
    print(f"Files with issues: {files_with_issues}")

    # Show verbose output if requested
    if args.verbose and files_ok:
        print("\n" + "=" * 80)
        print("FILES WITH VALID HEADERS:")
        print("=" * 80)
        for filepath in sorted(files_ok):
            print(f"  ✓ {filepath}")

    if all_issues:
        print("\n" + "=" * 80)
        print("FILES WITH ISSUES:")
        print("=" * 80)

        for filepath, issues in sorted(all_issues):
            print(f"\n{filepath}")
            for issue in issues:
                print(f"  ✗ {issue}")

    print("\n" + "=" * 80)

    if files_with_issues == 0:
        print("✓ All files have proper copyright and license headers!")
        return 0
    else:
        print(f"✗ Found {files_with_issues} file(s) with missing or incorrect headers.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
