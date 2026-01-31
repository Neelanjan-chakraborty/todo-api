#!/usr/bin/env python3
"""
Test script to evaluate generated repositories using the repo_evaluator.py

This script clones a generated repository and runs the evaluation metrics on it
to verify that it meets the SWE-Bench criteria.
"""

import os
import sys
import subprocess
import argparse
import tempfile
from pathlib import Path
from repo_evaluator import RepoEvaluator

def evaluate_generated_repo(repo_url: str, github_token: str = None):
    """Evaluate a generated repository using the repo evaluator."""

    print(f"Evaluating repository: {repo_url}")

    # Extract owner and repo name from URL
    # Format: https://github.com/owner/repo
    parts = repo_url.rstrip('/').split('/')
    if len(parts) < 2:
        raise ValueError(f"Invalid repo URL format: {repo_url}")

    owner = parts[-2]
    repo_name = parts[-1]

    # Create evaluator
    evaluator = RepoEvaluator(
        owner=owner,
        repo_name=repo_name,
        token=github_token,
        platform="github"
    )

    # Run evaluation
    try:
        report = evaluator.evaluate()

        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Repository: {report.repo_full_name}")
        print(f"Overall Score: {report.overall_score}/100")
        print(f"Recommendation: {report.recommendation}")
        print()

        print("Repository Metrics:")
        print(f"  Total files: {report.repo_metrics.total_files}")
        print(f"  Test files: {report.repo_metrics.test_files}")
        print(f"  Test coverage ratio: {report.repo_metrics.test_file_ratio*100:.1f}%")
        print(f"  CI/CD: {'Yes' if report.repo_metrics.has_ci_cd else 'No'}")
        print(f"  Test frameworks: {', '.join(report.repo_metrics.test_frameworks) if report.repo_metrics.test_frameworks else 'None'}")
        print(f"  Total commits: {report.repo_metrics.total_commits or 0}")
        print(f"  Recent commits (6mo): {report.repo_metrics.recent_commits_6mo or 0}")

        print("\nPR Analysis:")
        print(f"  Total PRs analyzed: {report.pr_analysis.total_prs}")
        print(f"  Accepted: {report.pr_analysis.accepted}")
        print(f"  Rejected: {report.pr_analysis.rejected}")
        print(f"  Acceptance rate: {report.pr_analysis.acceptance_rate*100:.1f}%")

        return report

    except Exception as e:
        print(f"Error evaluating repository: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate a generated repository")
    parser.add_argument("repo_url", help="URL of the repository to evaluate")
    parser.add_argument("--github-token", help="GitHub token for private repos or higher rate limits")

    args = parser.parse_args()

    try:
        report = evaluate_generated_repo(args.repo_url, args.github_token)

        if report and report.overall_score >= 70:
            print("
✅ Repository meets SWE-Bench criteria!"            sys.exit(0)
        elif report:
            print(f"\n⚠️  Repository score: {report.overall_score}/100 - May need improvements")
            sys.exit(1)
        else:
            print("\n❌ Evaluation failed")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">D:\Softwaes\personal\GHARI\test_generated_repo.py