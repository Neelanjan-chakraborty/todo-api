#!/usr/bin/env python3
"""
Test script to evaluate locally generated repositories
"""
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class RepoStats:
    name: str
    path: str
    total_files: int
    py_files: int
    test_files: int
    test_file_ratio: float
    total_lines_of_code: int
    source_lines_of_code: int
    test_lines_of_code: int
    commits: int
    has_github_workflow: bool
    has_pytest: bool
    

def count_lines(file_path):
    """Count non-empty lines in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for line in f if line.strip())
    except:
        return 0


def analyze_repo(repo_path):
    """Analyze a local repository"""
    repo_path = Path(repo_path)
    
    if not repo_path.exists():
        print(f"Error: {repo_path} does not exist")
        return None
    
    # Count files
    py_files = list(repo_path.rglob('*.py'))
    test_files = [f for f in py_files if 'test' in f.name or 'test' in str(f).lower()]
    
    # Count lines of code
    source_files = [f for f in py_files if f not in test_files]
    
    total_loc = sum(count_lines(f) for f in py_files)
    source_loc = sum(count_lines(f) for f in source_files)
    test_loc = sum(count_lines(f) for f in test_files)
    
    # Get commit count
    try:
        result = subprocess.run(
            ['git', 'rev-list', '--count', 'HEAD'],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=5
        )
        commits = int(result.stdout.strip()) if result.returncode == 0 else 0
    except:
        commits = 0
    
    # Check for GitHub workflow
    has_workflow = (repo_path / '.github' / 'workflows').exists()
    
    # Check for pytest
    has_pytest_config = any((repo_path / name).exists() for name in ['pytest.ini', 'pyproject.toml', 'setup.cfg'])
    
    stats = RepoStats(
        name=repo_path.name,
        path=str(repo_path),
        total_files=len(py_files),
        py_files=len(py_files),
        test_files=len(test_files),
        test_file_ratio=len(test_files) / len(py_files) if py_files else 0,
        total_lines_of_code=total_loc,
        source_lines_of_code=source_loc,
        test_lines_of_code=test_loc,
        commits=commits,
        has_github_workflow=has_workflow,
        has_pytest=has_pytest_config
    )
    
    return stats


def main():
    """Analyze all generated repos"""
    generated_dir = Path('generated_repos')
    
    if not generated_dir.exists():
        print(f"Error: {generated_dir} does not exist")
        return
    
    repos = sorted([d for d in generated_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    print(f"Found {len(repos)} repositories\n")
    
    all_stats = []
    for repo_path in repos:
        stats = analyze_repo(repo_path)
        if stats:
            all_stats.append(stats)
            print(f"[{repo_path.name}]")
            print(f"  Python files: {stats.py_files}")
            print(f"  Test files: {stats.test_files} ({stats.test_file_ratio*100:.1f}%)")
            print(f"  Total LoC: {stats.total_lines_of_code}")
            print(f"  Source LoC: {stats.source_lines_of_code}")
            print(f"  Test LoC: {stats.test_lines_of_code}")
            print(f"  Commits: {stats.commits}")
            print(f"  Has CI/CD: {stats.has_github_workflow}")
            print(f"  Has pytest: {stats.has_pytest}")
            print()
    
    # Print summary
    if all_stats:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        avg_test_ratio = sum(s.test_file_ratio for s in all_stats) / len(all_stats)
        avg_commits = sum(s.commits for s in all_stats) / len(all_stats)
        avg_loc = sum(s.total_lines_of_code for s in all_stats) / len(all_stats)
        
        print(f"Total repos: {len(all_stats)}")
        print(f"Avg test file ratio: {avg_test_ratio*100:.1f}%")
        print(f"Avg commits: {avg_commits:.1f}")
        print(f"Avg total LoC: {avg_loc:.0f}")
        print(f"Repos with CI/CD: {sum(1 for s in all_stats if s.has_github_workflow)}/{len(all_stats)}")
        
        # Export to JSON
        with open('repo_stats.json', 'w') as f:
            json.dump([asdict(s) for s in all_stats], f, indent=2)
        print("\nStats exported to repo_stats.json")


if __name__ == '__main__':
    main()
