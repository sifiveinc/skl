#!/usr/bin/env python3
# Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# SPDX-License-Identifier: MIT

"""
Script to check for SiFive copyright and MIT license headers in source files.

This script verifies that each relevant file contains:
1. A copyright notice with "SiFive, Inc." and years 2025-Present or 2026-Present
2. MIT License reference
3. Proper SPDX-License-Identifier

Usage:
    # Check all files in the repository
    ./check_license_headers.py

    # Check specific files
    ./check_license_headers.py file1.c file2.h

    # Show file type summary
    ./check_license_headers.py --summary

    # Show verbose output (lists all valid files)
    ./check_license_headers.py --verbose

    # Fix files with missing headers
    ./check_license_headers.py --fix

    # Fix specific files only
    ./check_license_headers.py --fix file1.c file2.h

Exit codes:
    0 - All files have proper headers
    1 - One or more files are missing headers
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Set

# Expected copyright patterns (allowing for 2025 or 2026)
COPYRIGHT_PATTERNS = [
    r"Copyright\s+\(c\)\s+202[56](-2026)?\s+SiFive,?\s+Inc\.\s+All\s+rights\s+reserved\.",
]

# Expected license reference
LICENSE_REFERENCE = [
    r"Licensed under the MIT License\.",
    r"See LICENSE file in the project root for full license information\."
]

# Expected SPDX identifier
SPDX_IDENTIFIER = [
    r"SPDX-License-Identifier:\s*MIT"
]

# File extensions to check and their comment styles
FILE_TYPES = {
    # C/C++ style comments
    '.c': ('//', '//'),
    '.h': ('//', '//'),
    '.cpp': ('//', '//'),
    '.hpp': ('//', '//'),
    # Python style comments
    '.py': ('#', '#'),
    # CMake style comments
    '.cmake': ('#', '#'),
    'CMakeLists.txt': ('#', '#'),
}

# Directories to exclude
EXCLUDE_DIRS = {
    '.git',
    'build',
    'doc/html',
    '__pycache__',
    '.pytest_cache',
}

# Files to exclude (relative paths from repo root)
EXCLUDE_FILES = {
    '.clang-tidy',
    '.gitignore',
    'LICENSE.txt',
    'CHANGELOG.md',
    'README.md',
    'CONTRIBUTING.md',
    'CONTRIBUTORS.md',
    '.github/CODEOWNERS',
}

# Exclude files matching these patterns
EXCLUDE_PATTERNS = [
    r'.*README\.md$',
    r'.*\.md$',  # Exclude all markdown files
]

def should_check_file(filepath: Path, repo_root: Path) -> bool:
    """Determine if a file should be checked for license headers."""
    # Check if in excluded directory
    rel_path = filepath.relative_to(repo_root)
    for part in rel_path.parts:
        if part in EXCLUDE_DIRS:
            return False

    # Check if in excluded files
    if str(rel_path) in EXCLUDE_FILES:
        return False

    # Check against exclude patterns
    rel_path_str = str(rel_path)
    for pattern in EXCLUDE_PATTERNS:
        if re.match(pattern, rel_path_str):
            return False

    # Check file extension or name
    if filepath.name == 'CMakeLists.txt':
        return True

    return filepath.suffix in FILE_TYPES


def get_comment_style(filepath: Path) -> Tuple[str, str]:
    """Get the comment style for a file."""
    if filepath.name == 'CMakeLists.txt':
        return FILE_TYPES['CMakeLists.txt']
    return FILE_TYPES.get(filepath.suffix, (None, None))


def generate_header(filepath: Path, year: str = "2025") -> str:
    """
    Generate the appropriate copyright header for a file.

    Args:
        filepath: Path to the file
        year: Starting copyright year (2025 or 2026)

    Returns:
        String containing the copyright header
    """
    comment_prefix, _ = get_comment_style(filepath)

    if comment_prefix is None:
        return ""

    header_lines = [
        f"{comment_prefix} Copyright (c) {year} SiFive, Inc. All rights reserved.",
        f"{comment_prefix} Licensed under the MIT License.",
        f"{comment_prefix} See LICENSE file in the project root for full license information.",
        f"{comment_prefix} SPDX-License-Identifier: MIT",
        ""
    ]

    return '\n'.join(header_lines)


def fix_file_header(filepath: Path, year: str = "2025") -> bool:
    """
    Add the copyright header to a file that's missing it.

    Args:
        filepath: Path to the file to fix
        year: Starting copyright year (2025 or 2026)

    Returns:
        True if file was modified, False otherwise
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            original_content = f.read()

        header = generate_header(filepath, year)
        if not header:
            return False

        # Check if file starts with shebang
        new_content = ""
        if original_content.startswith('#!'):
            # Preserve shebang line
            lines = original_content.split('\n', 1)
            shebang = lines[0] + '\n'
            rest = lines[1] if len(lines) > 1 else ""
            new_content = shebang + header + '\n' + rest
        else:
            new_content = header + '\n' + original_content

        # Write the modified content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True

    except Exception as e:
        print(f"Error fixing file {filepath}: {e}")
        return False


def check_file_header(filepath: Path) -> Tuple[bool, List[str]]:
    """
    Check if a file has the required copyright and license header.
    
    Returns:
        Tuple of (has_valid_header, list_of_issues)
    """
    issues = []
    
    try:
        with open(filepath, 'r') as f:
            # Read first 20 lines (header should be in this range)
            header_lines = []
            for i, line in enumerate(f):
                if i >= 20:
                    break
                header_lines.append(line)
            
            header_text = ''.join(header_lines)
            
            copyright_license_lines = COPYRIGHT_PATTERNS + LICENSE_REFERENCE + SPDX_IDENTIFIER
            # copyright_license_lines = LICENSE_REFERENCE
            comment_char, _ = get_comment_style(filepath)
            print(f"comment char is {comment_char}\n")
            lines = []
            for line in copyright_license_lines:
                line = comment_char + r"\s+" + line
                lines.append(line)
            copyright_license_pattern = r"(\r)?\n".join(lines)

            copyright_found = False
            if re.search(copyright_license_pattern, header_text):
                copyright_found = True

            # Check for copyright notice
            # copyright_found = False
            # for pattern in COPYRIGHT_PATTERNS:
            #     if re.search(pattern, header_text, re.IGNORECASE):
            #         copyright_found = True
            #         break
            
            if not copyright_found:
                issues.append("Missing or incorrect SiFive copyright notice (should be '2025-Present' or '2026-Present')")
            
            # Check for MIT license reference
            # if not re.search(LICENSE_REFERENCE, header_text, re.IGNORECASE):
            #     issues.append("Missing MIT License reference")
            
            # Check for SPDX identifier
            # if not re.search(SPDX_IDENTIFIER, header_text):
            #     issues.append("Missing SPDX-License-Identifier: MIT")
            
    except Exception as e:
        issues.append(f"Error reading file: {e}")
    
    return (len(issues) == 0, issues)

def main():
    """Main function to check all files in the repository."""
    parser = argparse.ArgumentParser(
        description="Check for SiFive copyright and MIT license headers in source files.",
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show verbose output including all checked files'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show summary of file types checked'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Automatically add missing headers to files'
    )
    parser.add_argument(
        '--year',
        choices=['2025', '2026'],
        default='2025',
        help='Copyright year to use when fixing files (default: 2025)'
    )
    parser.add_argument(
        'files',
        nargs='*',
        help='Specific files to check (if not provided, checks entire repository)'
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.resolve()

    files_checked = 0
    files_with_issues = 0
    all_issues = []
    file_type_counts = {}
    files_ok = []
    files_fixed = []

    # If specific files are provided, check only those
    if args.files:
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

            files_checked += 1

            # Track file types
            if filepath.name == 'CMakeLists.txt':
                ext = 'CMakeLists.txt'
            else:
                ext = filepath.suffix if filepath.suffix else filepath.name
            file_type_counts[ext] = file_type_counts.get(ext, 0) + 1

            has_valid_header, issues = check_file_header(filepath)

            if not has_valid_header:
                if args.fix:
                    # Try to fix the file
                    if fix_file_header(filepath, args.year):
                        files_fixed.append(rel_path)
                        # Re-check to verify the fix worked
                        has_valid_header, issues = check_file_header(filepath)
                        if has_valid_header:
                            files_ok.append(rel_path)
                        else:
                            files_with_issues += 1
                            all_issues.append((rel_path, issues))
                    else:
                        files_with_issues += 1
                        all_issues.append((rel_path, issues))
                else:
                    files_with_issues += 1
                    all_issues.append((rel_path, issues))
            else:
                files_ok.append(rel_path)
    else:
        # Check all files in repository
        print(f"Checking license headers in: {repo_root}")
        print("=" * 80)

        # Walk through all files
        for root, dirs, files in os.walk(repo_root):
            root_path = Path(root)

            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for filename in files:
                filepath = root_path / filename

                if not should_check_file(filepath, repo_root):
                    continue

                files_checked += 1
                rel_path = filepath.relative_to(repo_root)

                # Track file types (special case for CMakeLists.txt)
                if filepath.name == 'CMakeLists.txt':
                    ext = 'CMakeLists.txt'
                else:
                    ext = filepath.suffix if filepath.suffix else filepath.name
                file_type_counts[ext] = file_type_counts.get(ext, 0) + 1

                has_valid_header, issues = check_file_header(filepath)

                if not has_valid_header:
                    if args.fix:
                        # Try to fix the file
                        if fix_file_header(filepath, args.year):
                            files_fixed.append(rel_path)
                            # Re-check to verify the fix worked
                            has_valid_header, issues = check_file_header(filepath)
                            if has_valid_header:
                                files_ok.append(rel_path)
                            else:
                                files_with_issues += 1
                                all_issues.append((rel_path, issues))
                        else:
                            files_with_issues += 1
                            all_issues.append((rel_path, issues))
                    else:
                        files_with_issues += 1
                        all_issues.append((rel_path, issues))
                else:
                    files_ok.append(rel_path)

    # Print results
    print(f"\nFiles checked: {files_checked}")
    if args.fix and files_fixed:
        print(f"Files fixed: {len(files_fixed)}")
    print(f"Files with issues: {files_with_issues}")
    print(f"Files OK: {files_checked - files_with_issues}")

    # Show file type summary if requested
    if args.summary:
        print("\n" + "=" * 80)
        print("FILE TYPE SUMMARY:")
        print("=" * 80)
        for ext, count in sorted(file_type_counts.items()):
            print(f"  {ext:20s}: {count:4d} files")

    # Show fixed files
    if files_fixed:
        print("\n" + "=" * 80)
        print("FILES FIXED:")
        print("=" * 80)
        for filepath in sorted(files_fixed):
            print(f"  ✓ {filepath}")

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
