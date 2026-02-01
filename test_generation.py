#!/usr/bin/env python3
"""Test script for repository generation with substantial code."""

import sys
import os
# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from auto_repo_generator import generate_repo

# Test parameters
github_token = input('Enter GitHub token: ').strip()
github_username = input('Enter GitHub username: ').strip()

print('\n[*] Starting repository generation with substantial code...\n')

try:
    repo_url, eval_results = generate_repo(
        github_token=github_token,
        github_username=github_username,
        ollama_model='qwen3:8b',
        run_evaluation=True
    )
    
    print(f'\n[OK] Generation complete: {repo_url}')
    
    if eval_results:
        score = eval_results.get('overall_score', 0)
        print(f'\n[EVAL] Evaluation Results:')
        print(f'   Score: {score:.1f}/100')
        status = "PASSED" if eval_results.get("passed") else "FAILED"
        print(f'   Status: [{status}]')
        print(f'   Test files: {eval_results.get("test_files", 0)}')
        print(f'   Test file ratio: {eval_results.get("test_file_ratio", 0):.1f}%')
        print(f'   CI/CD: {"YES" if eval_results.get("has_ci_cd") else "NO"}')
        print(f'   Commits: {eval_results.get("recent_commits_6mo", 0)}')
    else:
        print('\n[INFO] Evaluation still running in parallel...')
        
except Exception as e:
    print(f'\n[ERROR] {e}')
    import traceback
    traceback.print_exc()
