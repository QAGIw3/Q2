#!/usr/bin/env python3
"""
Documentation validation script for Q2 Platform.

This script validates documentation quality and consistency across the repository.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

def find_markdown_files(root_dir: str) -> List[Path]:
    """Find all markdown files in the repository."""
    root_path = Path(root_dir)
    markdown_files = []
    
    # Skip certain directories
    skip_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'build', 'dist'}
    
    for md_file in root_path.rglob('*.md'):
        # Skip if any parent directory is in skip_dirs
        if any(part in skip_dirs for part in md_file.parts):
            continue
        markdown_files.append(md_file)
    
    return sorted(markdown_files)

def check_service_readme_structure(file_path: Path) -> List[str]:
    """Check if service README follows the expected structure."""
    issues = []
    
    if not file_path.name == 'README.md':
        return issues
    
    # Skip if not in a service directory (heuristic: has app/ subdirectory)
    service_dir = file_path.parent
    if not (service_dir / 'app').exists() and not any(p.suffix == '.py' for p in service_dir.rglob('*.py')):
        return issues
    
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        issues.append(f"Could not read file: {e}")
        return issues
    
    # Check for required sections
    required_sections = [
        r'# .+ Service',  # Service title
        r'## Overview',   # Overview section
        r'## Getting Started|## Quick Start',  # Getting started
    ]
    
    for section_pattern in required_sections:
        if not re.search(section_pattern, content, re.MULTILINE):
            issues.append(f"Missing required section matching pattern: {section_pattern}")
    
    # Check for service port documentation
    if not re.search(r'\*\*Port:\*\*|\*\*Service Port:\*\*|Port: \d+', content):
        issues.append("Missing service port documentation")
    
    return issues

def check_broken_links(file_path: Path, root_dir: Path) -> List[str]:
    """Check for broken internal links in markdown files."""
    issues = []
    
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        issues.append(f"Could not read file: {e}")
        return issues
    
    # Find markdown links
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    links = re.findall(link_pattern, content)
    
    for link_text, link_path in links:
        # Skip external links (http/https)
        if link_path.startswith(('http://', 'https://', 'mailto:')):
            continue
        
        # Skip anchors within the same document
        if link_path.startswith('#'):
            continue
        
        # Handle relative links
        if not os.path.isabs(link_path):
            # Resolve relative to the current file
            full_path = (file_path.parent / link_path).resolve()
        else:
            full_path = Path(link_path)
        
        # Check if the target exists
        if not full_path.exists():
            issues.append(f"Broken link: [{link_text}]({link_path})")
    
    return issues

def check_documentation_consistency(files: List[Path]) -> List[str]:
    """Check for consistency issues across documentation."""
    issues = []
    
    # Check for consistent terminology
    terminology_checks = [
        (r'Q Platform', r'Q2 Platform', "Use 'Q2 Platform' instead of 'Q Platform'"),
        (r'QuantumPulse ecosystem', r'Q2 Platform', "Use 'Q2 Platform' instead of 'QuantumPulse ecosystem'"),
    ]
    
    for file_path in files:
        try:
            content = file_path.read_text(encoding='utf-8')
        except:
            continue
        
        for old_term, new_term, message in terminology_checks:
            if re.search(old_term, content):
                issues.append(f"{file_path}: {message}")
    
    return issues

def validate_api_docs(root_dir: Path) -> List[str]:
    """Validate API documentation structure."""
    issues = []
    
    api_docs_dir = root_dir / 'docs' / 'api'
    if not api_docs_dir.exists():
        issues.append("Missing docs/api directory")
        return issues
    
    # Check for index file
    if not (api_docs_dir / 'index.md').exists():
        issues.append("Missing docs/api/index.md")
    
    # Expected service API docs based on service directories
    expected_services = []
    for service_dir in root_dir.iterdir():
        if service_dir.is_dir() and (service_dir / 'app').exists():
            service_name = service_dir.name.lower()
            expected_services.append(f"{service_name}.md")
    
    # Check if API docs exist for services
    for expected_doc in expected_services:
        api_doc_path = api_docs_dir / expected_doc
        if not api_doc_path.exists():
            issues.append(f"Missing API documentation: docs/api/{expected_doc}")
    
    return issues

def main():
    """Main validation function."""
    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    else:
        root_dir = Path.cwd()
    
    if not root_dir.exists():
        print(f"Error: Directory {root_dir} does not exist")
        sys.exit(1)
    
    print(f"Validating documentation in: {root_dir}")
    print("=" * 60)
    
    # Find all markdown files
    markdown_files = find_markdown_files(str(root_dir))
    print(f"Found {len(markdown_files)} markdown files")
    
    all_issues = []
    
    # Check each file
    for file_path in markdown_files:
        file_issues = []
        
        # Check service README structure
        service_issues = check_service_readme_structure(file_path)
        file_issues.extend(service_issues)
        
        # Check for broken links
        link_issues = check_broken_links(file_path, root_dir)
        file_issues.extend(link_issues)
        
        if file_issues:
            all_issues.append((file_path, file_issues))
    
    # Check documentation consistency
    consistency_issues = check_documentation_consistency(markdown_files)
    if consistency_issues:
        all_issues.append(("Consistency", consistency_issues))
    
    # Validate API documentation
    api_issues = validate_api_docs(root_dir)
    if api_issues:
        all_issues.append(("API Documentation", api_issues))
    
    # Report results
    if all_issues:
        print("\nDocumentation Issues Found:")
        print("=" * 60)
        
        for source, issues in all_issues:
            print(f"\n{source}:")
            for issue in issues:
                print(f"  - {issue}")
        
        print(f"\nTotal: {sum(len(issues) for _, issues in all_issues)} issues found")
        sys.exit(1)
    else:
        print("\n✅ No documentation issues found!")
        sys.exit(0)

if __name__ == "__main__":
    main()