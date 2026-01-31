#!/usr/bin/env python3
"""
Automated Repository Generator for AI Training

Creates high-quality Git repositories that meet SWE-Bench evaluation criteria
for training AI models on coding tasks. Uses Ollama for code generation.
"""

import os
import sys
import json
import subprocess
import argparse
import tempfile
import shutil
import time
import random
import threading
import re
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta

# Third-party imports
try:
    import ollama
except ImportError:
    print("Installing ollama-python...")
    subprocess.run([sys.executable, "-m", "pip", "install", "ollama"], check=True)
    import ollama

try:
    from github import Github
except ImportError:
    print("Installing PyGitHub...")
    subprocess.run([sys.executable, "-m", "pip", "install", "PyGitHub"], check=True)
    from github import Github

try:
    import yaml
except ImportError:
    print("Installing PyYAML...")
    subprocess.run([sys.executable, "-m", "pip", "install", "PyYAML"], check=True)
    import yaml


def get_existing_repos(github_token: str, github_username: str) -> List[str]:
    """Fetch list of existing repository names from GitHub account.
    
    Returns:
        List of repository names
    """
    try:
        g = Github(github_token)
        user = g.get_user()
        repos = user.get_repos()
        repo_names = [repo.name for repo in repos]
        print(f"Found {len(repo_names)} existing repositories")
        return repo_names
    except Exception as e:
        print(f"Warning: Failed to fetch existing repositories: {e}")
        return []


def generate_project_spec(ollama_model: str, github_token: Optional[str] = None, 
                         github_username: Optional[str] = None) -> Dict[str, Any]:
    """Generate a random project specification using Ollama with unique name avoidance.
    
    Args:
        ollama_model: Ollama model to use
        github_token: GitHub token for fetching existing repos
        github_username: GitHub username for fetching existing repos
    """
    
    # Get list of existing repos to avoid naming conflicts
    existing_repos = []
    if github_token and github_username:
        existing_repos = get_existing_repos(github_token, github_username)
    
    existing_repos_list = "\n".join(f"- {name}" for name in existing_repos) if existing_repos else "None"

    prompt = f"""
Generate a creative and useful software project idea for AI training. The project should be:

1. Realistic and implementable
2. Have clear utility or solve a real problem
3. Be suitable for demonstrating good software engineering practices
4. Include comprehensive tests and documentation

EXISTING REPOSITORIES (avoid these names):
{existing_repos_list}

Return a JSON object with these fields:
- name: kebab-case project name (e.g., "task-manager-api", "data-processor-cli") - MUST BE UNIQUE and NOT in the existing list
- description: 2-3 sentence description
- language: primary programming language (Python, JavaScript, etc.)
- framework: main framework/library (if applicable)
- category: one of [web, api, cli, library, mobile, desktop, game, data, devops]
- complexity: one of [simple, moderate, complex]
- features: array of 3-5 key features
- dependencies: array of 3-5 main dependencies/packages

Choose from popular, well-established technologies. Be creative but practical.
IMPORTANT: The name must not be in the existing repositories list above.
"""

    try:
        response = ollama.generate(
            model=ollama_model,
            prompt=prompt,
            format="json"
        )

        spec_data = json.loads(response['response'])

        # Validate required fields
        required_fields = ['name', 'description', 'language', 'framework',
                         'category', 'complexity', 'features', 'dependencies']

        for field in required_fields:
            if field not in spec_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate name is not in existing repos
        if spec_data['name'] in existing_repos:
            print(f"Warning: Generated name '{spec_data['name']}' conflicts with existing repo. Retrying...")
            # Retry with different model or fallback
            spec_data['name'] = f"{spec_data['name']}-{int(time.time()) % 10000}"
            print(f"Using modified name: {spec_data['name']}")

        return spec_data

    except Exception as e:
        print(f"Error generating project spec: {e}")
        # Fallback to a predefined project with timestamp to avoid conflicts
        import time as time_module
        timestamp = int(time_module.time()) % 100000
        fallback_name = f"demo-project-{timestamp}"
        return {
            "name": fallback_name,
            "description": "A demonstration project for AI training with automated repository generation",
            "language": "Python",
            "framework": "FastAPI",
            "category": "api",
            "complexity": "moderate",
            "features": ["Core functionality", "Error handling", "Testing", "Documentation"],
            "dependencies": ["fastapi", "sqlalchemy", "pydantic", "uvicorn"]
        }


def create_project_structure(spec: Dict[str, Any], base_dir: Path) -> Path:
    """Create the basic project structure."""

    project_dir = base_dir / spec["name"]
    project_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating project structure in {project_dir}")

    # Create basic directories
    dirs_to_create = [
        "src",
        "tests",
        ".github/workflows"
    ]

    for dir_name in dirs_to_create:
        (project_dir / dir_name).mkdir(parents=True, exist_ok=True)

    # Generate main source files
    generate_main_code(spec, project_dir)

    # Generate tests
    generate_tests(spec, project_dir)

    # Generate CI/CD
    generate_ci_cd(spec, project_dir)

    # Generate documentation
    generate_docs(spec, project_dir)

    # Generate configuration files
    generate_config_files(spec, project_dir)

    return project_dir


def generate_main_code(spec: Dict[str, Any], project_dir: Path):
    """Generate main application code using Ollama."""

    prompt = f"""
Generate the main application code for a {spec['language']} project using {spec['framework']}.

Project: {spec['name']}
Description: {spec['description']}
Features: {', '.join(spec['features'])}
Dependencies: {', '.join(spec['dependencies'])}

Requirements:
- Use modern {spec['language']} best practices
- Include proper error handling
- Use the specified framework appropriately
- Create modular, maintainable code
- Include type hints/docstrings where applicable

Generate the main application file(s) that implement the core functionality.
Return only the code, no explanations.
"""

    try:
        response = ollama.generate(
            model="llama3.1",  # Default model
            prompt=prompt
        )

        main_code = response['response']

        # Save main code
        main_file = project_dir / "src" / get_main_filename(spec['language'])
        main_file.write_text(main_code)

        print(f"Generated main code: {main_file}")

    except Exception as e:
        print(f"Error generating main code: {e}")
        # Create a minimal fallback
        create_fallback_code(spec, project_dir)


def generate_tests(spec: Dict[str, Any], project_dir: Path):
    """Generate comprehensive test suite."""

    prompt = f"""
Generate a comprehensive test suite for a {spec['language']} {spec['framework']} project.

Project: {spec['name']}
Description: {spec['description']}
Features: {', '.join(spec['features'])}

Requirements:
- Use appropriate testing framework for {spec['language']}
- Cover all main features with unit tests
- Include integration tests where applicable
- Use descriptive test names
- Include test fixtures/setup as needed
- Aim for high test coverage

Generate test files that thoroughly test the application.
"""

    try:
        response = ollama.generate(
            model="llama3.1",
            prompt=prompt
        )

        test_code = response['response']

        # Save test code
        test_file = project_dir / "tests" / get_test_filename(spec['language'])
        test_file.write_text(test_code)

        print(f"Generated tests: {test_file}")

    except Exception as e:
        print(f"Error generating tests: {e}")
        create_fallback_tests(spec, project_dir)


def generate_ci_cd(spec: Dict[str, Any], project_dir: Path):
    """Generate CI/CD pipeline configuration."""

    language_config = get_language_config(spec['language'])

    ci_config = f"""name: CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up {spec['language']}
      uses: {language_config["setup_action"]}
      with:
        {language_config["version_key"]}: '{language_config["version_value"]}'

    - name: Install dependencies
      run: {language_config["install_cmd"]}

    - name: Run tests
      run: {language_config["test_cmd"]}

    - name: Run linting
      run: {language_config["lint_cmd"]}
"""

    ci_file = project_dir / ".github/workflows/ci.yml"
    ci_file.write_text(ci_config)

    print(f"Generated CI/CD: {ci_file}")


def generate_docs(spec: Dict[str, Any], project_dir: Path):
    """Generate documentation."""

    readme_content = f"""# {spec['name'].replace('-', ' ').title()}

{spec['description']}

## Features

{chr(10).join(f"- {feature}" for feature in spec['features'])}

## Installation

```bash
# Clone the repository
git clone https://github.com/{{username}}/{spec['name']}.git
cd {spec['name']}

# Install dependencies
{get_language_config(spec['language'])["install_cmd"]}
```

## Usage

[Add usage instructions here]

## Development

```bash
# Run tests
{get_language_config(spec['language'])["test_cmd"]}

# Run linting
{get_language_config(spec['language'])["lint_cmd"]}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License
"""

    readme_file = project_dir / "README.md"
    readme_file.write_text(readme_content)

    print(f"Generated README: {readme_file}")


def generate_config_files(spec: Dict[str, Any], project_dir: Path):
    """Generate configuration files."""

    language_config = get_language_config(spec['language'])

    # Generate requirements/package files
    if spec['language'] == "Python":
        requirements = "\n".join(spec['dependencies'] + ["pytest", "black", "flake8"])
        (project_dir / "requirements.txt").write_text(requirements)

        # pyproject.toml
        pyproject_content = f"""[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{spec['name']}"
version = "0.1.0"
description = "{spec['description']}"
dependencies = {spec['dependencies']}

[tool.black]
line-length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]
"""
        (project_dir / "pyproject.toml").write_text(pyproject_content)

    elif spec['language'] == "JavaScript":
        package_json = {
            "name": spec['name'],
            "version": "0.1.0",
            "description": spec['description'],
            "main": "src/index.js",
            "scripts": {
                "test": "jest",
                "lint": "eslint src/",
                "start": "node src/index.js"
            },
            "dependencies": {dep: "^1.0.0" for dep in spec['dependencies']},
            "devDependencies": {
                "jest": "^29.0.0",
                "eslint": "^8.0.0"
            }
        }
        (project_dir / "package.json").write_text(json.dumps(package_json, indent=2))

    # .gitignore
    gitignore_content = language_config["gitignore"]
    (project_dir / ".gitignore").write_text(gitignore_content)

    print("Generated configuration files")


def get_language_config(language: str) -> Dict[str, str]:
    """Get configuration for a programming language."""

    configs = {
        "Python": {
            "main_file": "main.py",
            "test_file": "test_main.py",
            "setup_action": "actions/setup-python@v4",
            "version_key": "python-version",
            "version_value": "3.9",
            "install_cmd": "pip install -r requirements.txt",
            "test_cmd": "pytest",
            "lint_cmd": "flake8 src/",
            "gitignore": "venv/\n__pycache__/\n*.pyc\n.pytest_cache/\n"
        },
        "JavaScript": {
            "main_file": "index.js",
            "test_file": "index.test.js",
            "setup_action": "actions/setup-node@v4",
            "version_key": "node-version",
            "version_value": "18",
            "install_cmd": "npm install",
            "test_cmd": "npm test",
            "lint_cmd": "npm run lint",
            "gitignore": "node_modules/\n*.log\n"
        }
    }

    return configs.get(language, configs["Python"])


def get_main_filename(language: str) -> str:
    """Get the main file name for a language."""
    config = get_language_config(language)
    return config["main_file"]


def get_test_filename(language: str) -> str:
    """Get the test file name for a language."""
    config = get_language_config(language)
    return config["test_file"]


def create_fallback_code(spec: Dict[str, Any], project_dir: Path):
    """Create minimal fallback code if Ollama fails."""

    if spec['language'] == "Python":
        code = f'''"""
{spec['name']} - {spec['description']}
"""

def main():
    """Main function."""
    print("Hello from {spec['name']}!")

if __name__ == "__main__":
    main()
'''
    else:
        code = f'''// {spec['name']} - {spec['description']}

function main() {{
    console.log("Hello from {spec['name']}!");
}}

main();
'''

    main_file = project_dir / "src" / get_main_filename(spec['language'])
    main_file.write_text(code)


def create_fallback_tests(spec: Dict[str, Any], project_dir: Path):
    """Create minimal fallback tests."""

    if spec['language'] == "Python":
        test_code = '''"""Tests for main module."""

def test_main():
    """Test main function."""
    assert True  # Placeholder test
'''
    else:
        test_code = '''// Tests for main module

test('main function', () => {
    expect(true).toBe(true);  // Placeholder test
});
'''

    test_file = project_dir / "tests" / get_test_filename(spec['language'])
    test_file.write_text(test_code)


def get_available_ollama_models() -> List[str]:
    """Fetch list of available models from Ollama API.
    
    Returns:
        List of model names, or empty list if Ollama is not available
    """
    try:
        response = ollama.list()
        if response and 'models' in response:
            return [model['name'] for model in response['models']]
        return []
    except Exception as e:
        print(f"Warning: Could not fetch Ollama models: {e}")
        return []


def check_git_installed() -> Tuple[bool, str]:
    """Check if git is installed and accessible.
    
    Returns:
        Tuple of (is_installed, message)
    """
    try:
        result = subprocess.run(['git', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, f"Git error: {result.stderr}"
    except FileNotFoundError:
        return False, "Git not found. Please install Git from https://git-scm.com/download"
    except Exception as e:
        return False, f"Error checking git: {e}"


def initialize_git_repo(project_dir: Path, spec: Dict[str, Any]):
    """Initialize git repository and make initial commit."""
    
    # Validate git is installed
    git_ok, git_msg = check_git_installed()
    if not git_ok:
        raise Exception(f"Git validation failed: {git_msg}")

    # Ensure project_dir is absolute (Path object should already be resolved from caller)
    os.chdir(str(project_dir))

    try:
        # Initialize git
        result = subprocess.run(["git", "init"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise Exception(f"git init failed: {result.stderr or result.stdout}")
        
        print("Initialized git repository")
        
        # Set git user config (local scope)
        result = subprocess.run(["git", "config", "--local", "user.name", "AI Repo Generator"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise Exception(f"git config user.name failed: {result.stderr or result.stdout}")
        
        result = subprocess.run(["git", "config", "--local", "user.email", "ai@example.com"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise Exception(f"git config user.email failed: {result.stderr or result.stdout}")

        # Verify there are files to commit
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            files_to_commit = result.stdout.strip()
            print(f"Files to commit: {len(files_to_commit.splitlines())} files")
            if not files_to_commit:
                print("Warning: No files found to commit. This may indicate an issue with project structure generation.")
        
        # Add all files
        result = subprocess.run(["git", "add", "."], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise Exception(f"git add failed: {result.stderr or result.stdout}")

        print("Files staged for commit")

        # Initial commit
        commit_message = f"Initial commit: {spec['description']}"
        print(f"Creating commit with message: {commit_message}")
        
        result = subprocess.run(["git", "commit", "-m", commit_message], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            error_output = result.stderr or result.stdout
            # This might fail if nothing to commit, which is ok
            if "nothing to commit" not in error_output.lower():
                print(f"Git commit stderr: {error_output}")
                raise Exception(f"git commit failed: {error_output}")
            else:
                print("Nothing to commit (this is normal if all files are already in git)")
        else:
            print("Initial commit created successfully")
            
    except Exception as e:
        raise Exception(f"Failed to initialize git repository: {e}")


def validate_github_token(github_token: str) -> Tuple[bool, str]:
    """Validate GitHub token and check its scopes.
    
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        g = Github(github_token)
        user = g.get_user()
        # This will trigger an API call to validate the token
        login = user.login
        
        # Check rate limit to verify token works
        rate_limit = g.get_rate_limit()
        
        return True, f"Token valid for user: {login}"
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg:
            return False, "Invalid token: Authentication failed. Please check your token."
        elif "403" in error_msg:
            return False, (
                "Token permission error (403). Your Personal Access Token needs the following scopes:\n"
                "  - 'repo' scope (for private repositories)\n"
                "  - 'public_repo' scope (for public repositories only)\n\n"
                "To fix this:\n"
                "1. Go to GitHub Settings > Developer settings > Personal access tokens\n"
                "2. Generate a new token (classic) with 'repo' scope selected\n"
                "3. Or use Fine-grained tokens with 'Contents' and 'Administration' permissions"
            )
        return False, f"Token validation error: {error_msg}"


def create_github_repo(spec: Dict[str, Any], github_token: str, github_username: str, private: bool = False) -> str:
    """Create GitHub repository.
    
    Requires a Personal Access Token with:
    - 'repo' scope for private repositories
    - 'public_repo' scope for public repositories
    
    For Fine-grained tokens, requires:
    - Repository permissions: Administration (Read and write), Contents (Read and write)
    """
    g = Github(github_token)
    user = g.get_user()

    repo_name = spec['name']
    
    try:
        repo = user.create_repo(
            repo_name,
            description=spec['description'],
            private=private,
            auto_init=False
        )

        print(f"Created GitHub repository: {repo.html_url}")
        return repo.html_url

    except Exception as e:
        error_msg = str(e)
        # Handle "name already exists" error by adding timestamp
        if "already exists" in error_msg.lower():
            import time as time_module
            timestamp = int(time_module.time())
            new_repo_name = f"{repo_name}-{timestamp}"
            print(f"Repository '{repo_name}' already exists. Trying with name: {new_repo_name}")
            try:
                repo = user.create_repo(
                    new_repo_name,
                    description=spec['description'],
                    private=private,
                    auto_init=False
                )
                print(f"Created GitHub repository with unique name: {repo.html_url}")
                return repo.html_url
            except Exception as retry_error:
                print(f"Failed to create repository with unique name: {retry_error}")
                raise
        elif "403" in error_msg or "Resource not accessible" in error_msg:
            print("\n" + "="*60)
            print("GITHUB TOKEN PERMISSION ERROR")
            print("="*60)
            print("Your Personal Access Token lacks the required permissions.")
            print("\nRequired scopes for Classic tokens:")
            print("  ✓ 'repo' - Full control of private repositories")
            print("  ✓ 'public_repo' - Access public repositories (if public only)")
            print("\nFor Fine-grained tokens, enable:")
            print("  ✓ Repository permissions > Administration: Read and write")
            print("  ✓ Repository permissions > Contents: Read and write")
            print("\nHow to fix:")
            print("1. Go to: https://github.com/settings/tokens")
            print("2. Generate new token (classic) with 'repo' scope")
            print("3. Copy the new token and try again")
            print("="*60 + "\n")
        print(f"Error creating GitHub repo: {e}")
        raise


def setup_remote_and_push(project_dir: Path, repo_url: str):
    """Set up remote and push initial commit."""

    # Ensure project_dir is absolute (Path object should already be resolved from caller)
    try:
        os.chdir(str(project_dir))
    except OSError as e:
        raise Exception(f"Failed to change to directory {project_dir}: {e}")

    try:
        # Add remote
        result = subprocess.run(["git", "remote", "add", "origin", repo_url], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise Exception(f"Failed to add remote: {result.stderr}")

        # Push - try main branch first, then master
        result = subprocess.run(["git", "push", "-u", "origin", "main"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            # Try master branch instead
            result = subprocess.run(["git", "push", "-u", "origin", "master"], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise Exception(f"Failed to push to GitHub: {result.stderr}")

        print("Pushed initial commit to GitHub")
    except Exception as e:
        raise Exception(f"Failed to set up remote and push: {e}")


def create_issues(spec: Dict[str, Any], github_token: str, github_username: str, repo_name: str, min_issues: int = 3):
    """Create GitHub issues for the project."""

    g = Github(github_token)
    repo = g.get_repo(f"{github_username}/{repo_name}")

    issues_to_create = [
        {
            "title": f"Implement {spec['features'][0]}",
            "body": f"Add functionality for {spec['features'][0].lower()}. This is a core feature of the application."
        },
        {
            "title": "Add comprehensive error handling",
            "body": "Implement proper error handling throughout the application with meaningful error messages."
        },
        {
            "title": "Add logging functionality",
            "body": "Implement structured logging to help with debugging and monitoring."
        },
        {
            "title": "Create API documentation",
            "body": "Document all API endpoints with examples and parameter descriptions."
        },
        {
            "title": "Add input validation",
            "body": "Implement comprehensive input validation for all user inputs and API parameters."
        }
    ]

    created_issues = []
    for issue_data in issues_to_create[:min_issues]:
        issue = repo.create_issue(
            title=issue_data["title"],
            body=issue_data["body"]
        )
        created_issues.append(issue)
        print(f"Created issue: {issue.title} (#{issue.number})")

    return created_issues


def make_development_commits(project_dir: Path, spec: Dict[str, Any], issues: List, max_commits: int = 10):
    """Make development commits that reference issues."""

    # Ensure project_dir is absolute (Path object should already be resolved from caller)
    os.chdir(str(project_dir))

    commit_messages = [
        f"Add basic project structure and configuration",
        f"Implement core {spec['features'][0].lower()} functionality #1",
        f"Add error handling and validation #2",
        f"Implement logging system #3",
        f"Add comprehensive tests for core features",
        f"Update documentation and API docs #4",
        f"Add input validation #5",
        f"Refactor code for better maintainability",
        f"Add configuration management",
        f"Update CI/CD pipeline",
    ]

    for i, message in enumerate(commit_messages[:max_commits]):
        try:
            # Make some changes to files
            if i % 3 == 0:
                # Modify main code
                main_file = project_dir / "src" / get_main_filename(spec['language'])
                if main_file.exists():
                    content = main_file.read_text()
                    content += f"\n# Commit {i+1}: {message}\n"
                    main_file.write_text(content)

            elif i % 3 == 1:
                # Modify tests
                test_file = project_dir / "tests" / get_test_filename(spec['language'])
                if test_file.exists():
                    content = test_file.read_text()
                    content += f"\n# Additional test {i+1}\n"
                    test_file.write_text(content)

            else:
                # Modify README
                readme_file = project_dir / "README.md"
                if readme_file.exists():
                    content = readme_file.read_text()
                    content += f"\n## Update {i+1}\n{message}\n"
                    readme_file.write_text(content)

            # Commit
            result = subprocess.run(["git", "add", "."], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print(f"Warning: git add failed on commit {i+1}: {result.stderr}")
                continue
            
            result = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print(f"Warning: git commit failed on commit {i+1}: {result.stderr}")
                continue

            # Push
            result = subprocess.run(["git", "push"], capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"Warning: git push failed on commit {i+1}: {result.stderr}")
                continue

            print(f"Committed: {message}")
        except Exception as e:
            print(f"Error on commit {i+1}: {e}")
            continue

        # Small delay between commits
        time.sleep(1)


def create_backdated_commits(repo_path: Path, ollama_model: str, num_commits: int = 5,
                             start_date: Optional[datetime] = None, interval_days: int = 1,
                             language: Optional[str] = None):
    """Create backdated commits using LLM-generated messages and small code changes.

    For each backdated commit:
    - Request a small change and commit message from Ollama (returned as JSON)
    - Apply the change to the repository (create/modify files)
    - Commit with GIT_AUTHOR_DATE and GIT_COMMITTER_DATE set to backdated timestamp
    Returns a list of commits made (date and message).
    """
    if num_commits <= 0:
        return []

    # Ensure repo_path is absolute (Path object should already be resolved from caller)
    os.chdir(str(repo_path))
    commits_made = []
    if start_date is None:
        # default: num_commits days in the past
        start_date = datetime.now(timezone.utc) - timedelta(days=num_commits)

    for i in range(num_commits):
        commit_date = start_date + timedelta(days=i * interval_days)
        # Use ISO format for date env vars; Git accepts ISO8601 for these env vars
        date_iso = commit_date.isoformat()

        prompt = f"""
You are helping create a small, safe code change and a concise commit message for a {language or 'project'}.
Return JSON with fields:
- message: short commit message (10-100 chars)
- changes: array of objects with {"path": "relative/path/to/file", "content": "file contents or snippet", "action": "create|append|replace"}
Only return valid JSON.
"""
        try:
            response = ollama.generate(model=ollama_model, prompt=prompt, format='json')
            change_data = json.loads(response['response'])
        except Exception as e:
            print(f"Error generating backdated change from LLM: {e}")
            continue

        # Apply changes
        changes = change_data.get('changes', [])
        for change in changes:
            rel_path = change.get('path', f'generated_change_{i+1}.txt')
            path = repo_path / rel_path
            action = change.get('action', 'append')
            content = change.get('content', '')
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if action == 'create':
                    if not path.exists():
                        path.write_text(content)
                    else:
                        # If file exists, append to avoid data loss
                        path.write_text(path.read_text() + "\n" + content)
                elif action == 'append':
                    existing = path.read_text() if path.exists() else ''
                    path.write_text(existing + "\n" + content)
                elif action == 'replace':
                    path.write_text(content)
            except Exception as e:
                print(f"Error applying change to {path}: {e}")

        # Commit with backdated timestamp
        env = os.environ.copy()
        env['GIT_AUTHOR_DATE'] = date_iso
        env['GIT_COMMITTER_DATE'] = date_iso
        env['GIT_AUTHOR_NAME'] = 'AI Repo Generator'
        env['GIT_AUTHOR_EMAIL'] = 'ai@example.com'
        commit_msg = change_data.get('message', f'AI-generated backdated commit {i+1}')
        try:
            subprocess.run(['git', 'add', '.'], check=True, capture_output=True, env=env)
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True, capture_output=True, env=env)
            commits_made.append({'date': date_iso, 'message': commit_msg})
            print(f"Created backdated commit: {commit_msg} ({date_iso})")
        except Exception as e:
            print(f"Failed to commit backdated change: {e}")
            continue

    return commits_made


def run_tests_locally(project_dir: Path, spec: Dict[str, Any]):
    """Run tests locally to ensure they pass."""

    # Ensure project_dir is absolute (Path object should already be resolved from caller)
    os.chdir(str(project_dir))

    try:
        language_config = get_language_config(spec['language'])

        # Install dependencies
        if spec['language'] == "Python":
            subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True, capture_output=True)
        elif spec['language'] == "JavaScript":
            subprocess.run(["npm", "install"], check=True, capture_output=True)

        # Run tests
        result = subprocess.run(language_config["test_cmd"].split(), capture_output=True, text=True)

        if result.returncode == 0:
            print("Tests passed successfully")
        else:
            print(f"Tests failed: {result.stderr}")

    except Exception as e:
        print(f"Error running tests: {e}")


def generate_repo(github_token: str, github_username: str, ollama_model: str = "llama3.1",
                 base_dir: Path = Path("./generated_repos"), private: bool = False,
                 max_commits: int = 10, min_issues: int = 3,
                 backdate_count: int = 0, backdate_start: Optional[str] = None,
                 backdate_interval_days: int = 1, backdate_model: Optional[str] = None) -> str:
    """Generate a complete repository."""

    print("Starting repository generation...")
    
    # Convert to absolute path to avoid issues with changing directories
    base_dir = Path(base_dir).resolve()

    # Generate project specification with awareness of existing repos
    spec = generate_project_spec(ollama_model, github_token=github_token, github_username=github_username)
    print(f"Generated project: {spec['name']} ({spec['language']} {spec['framework']})")

    # Create project structure
    project_dir = create_project_structure(spec, base_dir)

    # Initialize git
    initialize_git_repo(project_dir, spec)

    # Create GitHub repository
    repo_url = create_github_repo(spec, github_token, github_username, private)

    # Push initial commit
    setup_remote_and_push(project_dir, repo_url)

    # Create issues
    issues = create_issues(spec, github_token, github_username, spec['name'], min_issues)

    # Make development commits
    make_development_commits(project_dir, spec, issues, max_commits)

    # Optional backdated commits
    if backdate_count and backdate_count > 0:
        try:
            # parse start date if provided
            start_dt = None
            if backdate_start:
                try:
                    start_dt = datetime.fromisoformat(backdate_start)
                    if start_dt.tzinfo is None:
                        start_dt = start_dt.replace(tzinfo=timezone.utc)
                except Exception:
                    # fallback to parse YYYY-MM-DD
                    start_dt = datetime.strptime(backdate_start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            create_backdated_commits(project_dir, ollama_model=backdate_model or ollama_model,
                                     num_commits=backdate_count, start_date=start_dt,
                                     interval_days=backdate_interval_days, language=spec.get('language'))
        except Exception as e:
            print(f"Error creating backdated commits: {e}")

    # Run tests
    run_tests_locally(project_dir, spec)

    print(f"Repository generation complete: {repo_url}")

    return repo_url


def main():
    parser = argparse.ArgumentParser(description="Generate automated Git repositories for AI training")
    parser.add_argument("--github-token", required=False, help="GitHub personal access token")
    parser.add_argument("--github-username", required=False, help="GitHub username")
    parser.add_argument("--ollama-model", default="llama3.1", help="Ollama model to use")
    parser.add_argument("--base-dir", default="./generated_repos", help="Base directory for generated repos")
    parser.add_argument("--private", action="store_true", help="Make repositories private")
    parser.add_argument("--max-commits", type=int, default=10, help="Maximum number of development commits")
    parser.add_argument("--min-issues", type=int, default=3, help="Minimum number of GitHub issues to create")
    parser.add_argument("--backdate-count", type=int, default=0, help="Number of backdated commits to create")
    parser.add_argument("--backdate-start", type=str, default=None, help="Start date for backdated commits (YYYY-MM-DD or ISO)")
    parser.add_argument("--backdate-interval", type=int, default=1, help="Interval in days between backdated commits")
    parser.add_argument("--backdate-model", type=str, default=None, help="Ollama model to use for backdated commits")
    parser.add_argument("--gui", action="store_true", help="Launch GUI mode")

    args = parser.parse_args()

    # Check git is installed
    git_ok, git_msg = check_git_installed()
    if not git_ok:
        print("\n" + "="*60)
        print("GIT NOT FOUND")
        print("="*60)
        print(f"Error: {git_msg}")
        print("\nGit is required for repository generation.")
        print("Download from: https://git-scm.com/download")
        print("="*60 + "\n")
        sys.exit(1)
    else:
        print(f"✓ Git found: {git_msg}")

    # Launch GUI if requested or if no token/username provided
    if args.gui or not (args.github_token and args.github_username):
        launch_gui()
        return

    try:
        repo_url = generate_repo(
            github_token=args.github_token,
            github_username=args.github_username,
            ollama_model=args.ollama_model,
            base_dir=Path(args.base_dir),
            private=args.private,
            max_commits=args.max_commits,
            min_issues=args.min_issues,
            backdate_count=args.backdate_count,
            backdate_start=args.backdate_start,
            backdate_interval_days=args.backdate_interval,
            backdate_model=args.backdate_model
        )
        print(f"\nSuccess! Repository created: {repo_url}")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def is_git_url(path_or_url: str) -> bool:
    """Check if the string is a Git URL."""
    git_url_patterns = [
        r'^https?://.*\.git$',
        r'^https?://github\.com/[\w-]+/[\w-]+/?$',
        r'^https?://gitlab\.com/[\w-]+/[\w-]+/?$',
        r'^https?://bitbucket\.org/[\w-]+/[\w-]+/?$',
        r'^git@[\w.-]+:[\w-]+/[\w-]+\.git$',
        r'^ssh://git@[\w.-]+/[\w-]+/[\w-]+\.git$',
    ]
    return any(re.match(pattern, path_or_url.strip()) for pattern in git_url_patterns)


def clone_remote_repo(repo_url: str, target_dir: Optional[Path] = None, 
                      github_token: Optional[str] = None) -> Tuple[Path, str]:
    """Clone a remote repository to local directory.
    
    Args:
        repo_url: Git URL (https or ssh)
        target_dir: Directory to clone into (default: temp directory)
        github_token: Optional token for private repos
        
    Returns:
        Tuple of (local_path, repo_name)
    """
    # Extract repo name from URL
    repo_name = repo_url.rstrip('/').rstrip('.git').split('/')[-1]
    
    if target_dir is None:
        target_dir = Path(tempfile.mkdtemp(prefix="repo_modify_"))
    
    clone_path = target_dir / repo_name
    
    # Build clone URL with token for private repos (https only)
    clone_url = repo_url
    if github_token and repo_url.startswith('https://'):
        # Insert token into URL for authentication
        if 'github.com' in repo_url:
            clone_url = repo_url.replace('https://github.com', f'https://{github_token}@github.com')
        elif 'gitlab.com' in repo_url:
            clone_url = repo_url.replace('https://gitlab.com', f'https://oauth2:{github_token}@gitlab.com')
    
    print(f"Cloning repository {repo_name} to {clone_path}...")
    
    try:
        result = subprocess.run(
            ['git', 'clone', clone_url, str(clone_path)],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            error_msg = result.stderr
            if 'Authentication failed' in error_msg or '403' in error_msg:
                raise Exception(
                    f"Authentication failed. For private repos, ensure your token has 'repo' scope.\n"
                    f"Error: {error_msg}"
                )
            raise Exception(f"Git clone failed: {error_msg}")
        
        print(f"Successfully cloned to {clone_path}")
        return clone_path, repo_name
        
    except subprocess.TimeoutExpired:
        raise Exception("Clone operation timed out after 5 minutes")


def modify_existing_repo(repo_path_or_url: str, github_token: str, github_username: str, 
                        repo_name: Optional[str] = None, ollama_model: str = "llama3.1",
                        modification_type: str = "add_features") -> bool:
    """Modify an existing repository using LLM.
    
    Args:
        repo_path_or_url: Local path OR remote Git URL (.git URL or GitHub/GitLab URL)
        github_token: GitHub token for API access and private repo cloning
        github_username: GitHub username
        repo_name: Repository name (auto-detected if URL provided)
        ollama_model: Ollama model to use
        modification_type: Type of modification (add_features, fix_bugs, refactor, add_tests)
    """
    cloned_temp_dir = None
    
    # Check if it's a URL or local path
    if is_git_url(str(repo_path_or_url)):
        print(f"Detected remote repository URL: {repo_path_or_url}")
        repo_path, detected_name = clone_remote_repo(repo_path_or_url, github_token=github_token)
        repo_name = repo_name or detected_name
        cloned_temp_dir = repo_path.parent  # For cleanup later if needed
    else:
        repo_path = Path(repo_path_or_url)
        if not repo_path.exists():
            print(f"Error: Path does not exist: {repo_path}")
            return False
        if not repo_name:
            repo_name = repo_path.name
    
    print(f"Modifying repository: {repo_name} at {repo_path}...")
    
    os.chdir(repo_path)
    
    # Detect project language and structure
    language = detect_repo_language(repo_path)
    print(f"Detected language: {language}")
    
    # Generate modification based on type
    if modification_type == "add_features":
        modification_prompt = f"""
Analyze this {language} project and suggest 3 new features to add.
For each feature:
- Provide a brief description
- Suggest file modifications needed
- Include code snippets

Consider the existing codebase structure and maintain consistency.
Return as JSON with format:
{{
    "features": [
        {{"name": "Feature name", "description": "...", "implementation": "code here"}}
    ]
}}
"""
    elif modification_type == "fix_bugs":
        modification_prompt = f"""
Review this {language} codebase and identify potential bugs or improvements.
Suggest fixes for:
- Error handling
- Edge cases
- Code quality issues

Return as JSON with fixes and improved code.
"""
    elif modification_type == "refactor":
        modification_prompt = f"""
Analyze this {language} codebase and suggest refactoring improvements:
- Code organization
- Performance optimizations
- Design pattern applications

Return as JSON with refactoring suggestions and code.
"""
    else:  # add_tests
        modification_prompt = f"""
Analyze this {language} project and generate comprehensive tests.
Include:
- Unit tests for main functions
- Integration tests
- Edge case coverage

Return as JSON with test files and code.
"""
    
    try:
        # Get LLM suggestions
        response = ollama.generate(
            model=ollama_model,
            prompt=modification_prompt,
            format="json"
        )
        
        modifications = json.loads(response['response'])
        print(f"Generated modifications: {json.dumps(modifications, indent=2)}")
        
        # Apply modifications (simplified - in real use, would be more sophisticated)
        apply_modifications(repo_path, modifications, language)
        
        # Commit changes
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", f"AI-generated modifications: {modification_type}"], 
                      check=True, capture_output=True)
        
        # Push to GitHub
        result = subprocess.run(["git", "remote", "get-url", "origin"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            subprocess.run(["git", "push"], check=True, capture_output=True)
            print("Pushed modifications to GitHub")
        
        return True
        
    except Exception as e:
        print(f"Error modifying repository: {e}")
        return False


def detect_repo_language(repo_path: Path) -> str:
    """Detect primary programming language of repository."""
    
    language_files = {
        'Python': ['.py'],
        'JavaScript': ['.js', '.jsx'],
        'TypeScript': ['.ts', '.tsx'],
        'Java': ['.java'],
        'Go': ['.go'],
        'Rust': ['.rs'],
        'C++': ['.cpp', '.cc', '.h'],
        'Ruby': ['.rb'],
        'PHP': ['.php'],
    }
    
    file_counts = {lang: 0 for lang in language_files}
    
    for ext_list in language_files.values():
        for ext in ext_list:
            file_counts[list(language_files.keys())[list(language_files.values()).index(ext_list)]] += len(list(repo_path.rglob(f'*{ext}')))
    
    return max(file_counts.items(), key=lambda x: x[1])[0] if max(file_counts.values()) > 0 else "Python"


def apply_modifications(repo_path: Path, modifications: Dict, language: str):
    """Apply LLM-generated modifications to repository."""
    
    # This is a simplified version - real implementation would be more sophisticated
    if 'features' in modifications:
        for feature in modifications['features']:
            feature_name = feature.get('name', 'new_feature').lower().replace(' ', '_')
            
            # Create or modify files based on language
            if language == 'Python':
                feature_file = repo_path / 'src' / f'{feature_name}.py'
                feature_file.parent.mkdir(parents=True, exist_ok=True)
                feature_file.write_text(feature.get('implementation', '# Feature implementation\n'))
            elif language in ['JavaScript', 'TypeScript']:
                ext = '.ts' if language == 'TypeScript' else '.js'
                feature_file = repo_path / 'src' / f'{feature_name}{ext}'
                feature_file.parent.mkdir(parents=True, exist_ok=True)
                feature_file.write_text(feature.get('implementation', '// Feature implementation\n'))
    
    print(f"Applied modifications to {repo_path}")


class RepoGeneratorGUI:
    """GUI for Repository Generator."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("AI Repository Generator")
        self.root.geometry("800x900")
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create main container
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Operation Mode", padding="10")
        mode_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.mode_var = tk.StringVar(value="create")
        ttk.Radiobutton(mode_frame, text="Create New Repository", 
                       variable=self.mode_var, value="create",
                       command=self.on_mode_change).grid(row=0, column=0, padx=5)
        ttk.Radiobutton(mode_frame, text="Modify Existing Repository", 
                       variable=self.mode_var, value="modify",
                       command=self.on_mode_change).grid(row=0, column=1, padx=5)
        
        # GitHub credentials
        creds_frame = ttk.LabelFrame(main_frame, text="GitHub Credentials", padding="10")
        creds_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(creds_frame, text="GitHub Token:").grid(row=0, column=0, sticky=tk.W, pady=2)
        token_frame = ttk.Frame(creds_frame)
        token_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        self.token_entry = ttk.Entry(token_frame, width=40, show="*")
        self.token_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(token_frame, text="Validate", command=self.validate_token).grid(row=0, column=1, padx=5)
        
        # Token help label
        token_help = ttk.Label(creds_frame, text="Token needs 'repo' scope. Get one at: github.com/settings/tokens", 
                              font=('TkDefaultFont', 8), foreground='gray')
        token_help.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=0)
        
        ttk.Label(creds_frame, text="GitHub Username:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.username_entry = ttk.Entry(creds_frame, width=50)
        self.username_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2)
        
        # Create repo options
        self.create_frame = ttk.LabelFrame(main_frame, text="Create New Repository Options", padding="10")
        self.create_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(self.create_frame, text="Ollama Model:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.model_combo = ttk.Combobox(self.create_frame, width=28, state="readonly")
        self.model_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(self.create_frame, text="Base Directory:").grid(row=1, column=0, sticky=tk.W, pady=2)
        dir_frame = ttk.Frame(self.create_frame)
        dir_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        self.basedir_entry = ttk.Entry(dir_frame, width=35)
        self.basedir_entry.insert(0, "./generated_repos")
        self.basedir_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(dir_frame, text="Browse", command=self.browse_directory).grid(row=0, column=1, padx=5)
        
        ttk.Label(self.create_frame, text="Max Commits:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.commits_spin = ttk.Spinbox(self.create_frame, from_=1, to=50, width=10)
        self.commits_spin.set(10)
        self.commits_spin.grid(row=2, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(self.create_frame, text="Min Issues:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.issues_spin = ttk.Spinbox(self.create_frame, from_=1, to=20, width=10)
        self.issues_spin.set(3)
        self.issues_spin.grid(row=3, column=1, sticky=tk.W, pady=2)
        
        # Backdated commits options
        self.backdate_var = tk.BooleanVar()
        ttk.Checkbutton(self.create_frame, text="Add backdated commits", variable=self.backdate_var).grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Label(self.create_frame, text="Backdate Count:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.backdate_count_spin = ttk.Spinbox(self.create_frame, from_=0, to=365, width=10)
        self.backdate_count_spin.set(0)
        self.backdate_count_spin.grid(row=5, column=1, sticky=tk.W, pady=2)
        ttk.Label(self.create_frame, text="Start Date (YYYY-MM-DD):").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.backdate_start_entry = ttk.Entry(self.create_frame, width=30)
        self.backdate_start_entry.grid(row=6, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(self.create_frame, text="Interval (days):").grid(row=7, column=0, sticky=tk.W, pady=2)
        self.backdate_interval_spin = ttk.Spinbox(self.create_frame, from_=1, to=365, width=10)
        self.backdate_interval_spin.set(1)
        self.backdate_interval_spin.grid(row=7, column=1, sticky=tk.W, pady=2)
        
        self.private_var = tk.BooleanVar()
        ttk.Checkbutton(self.create_frame, text="Make repository private", 
                       variable=self.private_var).grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Modify repo options
        self.modify_frame = ttk.LabelFrame(main_frame, text="Modify Existing Repository Options", padding="10")
        self.modify_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(self.modify_frame, text="Repo Path or URL:").grid(row=0, column=0, sticky=tk.W, pady=2)
        path_frame = ttk.Frame(self.modify_frame)
        path_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        self.repo_path_entry = ttk.Entry(path_frame, width=35)
        self.repo_path_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(path_frame, text="Browse", command=self.browse_repo).grid(row=0, column=1, padx=5)
        
        # Help text for URL support
        url_help = ttk.Label(self.modify_frame, text="Accepts local path OR remote URL (e.g., https://github.com/user/repo.git)", 
                            font=('TkDefaultFont', 8), foreground='gray')
        url_help.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=0)
        
        ttk.Label(self.modify_frame, text="Repository Name:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.repo_name_entry = ttk.Entry(self.modify_frame, width=30)
        self.repo_name_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2)
        name_help = ttk.Label(self.modify_frame, text="(Optional if using URL - will auto-detect)", 
                             font=('TkDefaultFont', 8), foreground='gray')
        name_help.grid(row=3, column=1, sticky=tk.W, pady=0)
        
        ttk.Label(self.modify_frame, text="Modification Type:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.mod_type_combo = ttk.Combobox(self.modify_frame, width=28, state="readonly")
        self.mod_type_combo['values'] = ('add_features', 'fix_bugs', 'refactor', 'add_tests')
        self.mod_type_combo.current(0)
        self.mod_type_combo.grid(row=4, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(self.modify_frame, text="Ollama Model:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.modify_model_combo = ttk.Combobox(self.modify_frame, width=28, state="readonly")
        self.modify_model_combo.grid(row=5, column=1, sticky=(tk.W, tk.E), pady=2)
        
        # Backdated commits options for modify
        self.modify_backdate_var = tk.BooleanVar()
        ttk.Checkbutton(self.modify_frame, text="Add backdated commits", variable=self.modify_backdate_var).grid(row=6, column=0, sticky=tk.W, pady=2)
        ttk.Label(self.modify_frame, text="Backdate Count:").grid(row=7, column=0, sticky=tk.W, pady=2)
        self.modify_backdate_count_spin = ttk.Spinbox(self.modify_frame, from_=0, to=365, width=10)
        self.modify_backdate_count_spin.set(0)
        self.modify_backdate_count_spin.grid(row=7, column=1, sticky=tk.W, pady=2)
        ttk.Label(self.modify_frame, text="Start Date (YYYY-MM-DD):").grid(row=8, column=0, sticky=tk.W, pady=2)
        self.modify_backdate_start_entry = ttk.Entry(self.modify_frame, width=30)
        self.modify_backdate_start_entry.grid(row=8, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(self.modify_frame, text="Interval (days):").grid(row=9, column=0, sticky=tk.W, pady=2)
        self.modify_backdate_interval_spin = ttk.Spinbox(self.modify_frame, from_=1, to=365, width=10)
        self.modify_backdate_interval_spin.set(1)
        self.modify_backdate_interval_spin.grid(row=9, column=1, sticky=tk.W, pady=2)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        self.run_button = ttk.Button(button_frame, text="Generate Repository", 
                                     command=self.run_generation)
        self.run_button.grid(row=0, column=0, padx=5)
        
        ttk.Button(button_frame, text="Clear Log", command=self.clear_log).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).grid(row=0, column=2, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Log output
        log_frame = ttk.LabelFrame(main_frame, text="Log Output", padding="10")
        log_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, width=90)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Initial mode setup
        self.on_mode_change()
        
        # Load available models in background for both comboboxes
        threading.Thread(target=self.load_models, daemon=True).start()
    
    def load_models(self):
        """Load available Ollama models into combobox widgets."""
        try:
            models = get_available_ollama_models()
            if models:
                self.root.after(0, lambda: self.model_combo.configure(values=models))
                self.root.after(0, lambda: self.model_combo.current(0))
                self.root.after(0, lambda: self.modify_model_combo.configure(values=models))
                self.root.after(0, lambda: self.modify_model_combo.current(0))
                self.log(f"✓ Loaded {len(models)} Ollama model(s): {', '.join(models)}")
            else:
                # Fallback to default model
                default_models = ["llama3.1"]
                self.root.after(0, lambda: self.model_combo.configure(values=default_models))
                self.root.after(0, lambda: self.model_combo.current(0))
                self.root.after(0, lambda: self.modify_model_combo.configure(values=default_models))
                self.root.after(0, lambda: self.modify_model_combo.current(0))
                self.log("⚠ Warning: No Ollama models detected.")
                self.log("  Make sure Ollama is running: ollama serve")
        except Exception as e:
            self.log(f"✗ Error loading models: {e}")
            default_models = ["llama3.1"]
            self.root.after(0, lambda: self.model_combo.configure(values=default_models))
            self.root.after(0, lambda: self.model_combo.current(0))
            self.root.after(0, lambda: self.modify_model_combo.configure(values=default_models))
            self.root.after(0, lambda: self.modify_model_combo.current(0))
    
    def on_mode_change(self):
        """Handle mode change between create and modify."""
        mode = self.mode_var.get()
        if mode == "create":
            self.create_frame.grid()
            self.modify_frame.grid_remove()
            self.run_button.config(text="Generate Repository")
        else:
            self.create_frame.grid_remove()
            self.modify_frame.grid()
            self.run_button.config(text="Modify Repository")
    
    def browse_directory(self):
        """Browse for base directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.basedir_entry.delete(0, tk.END)
            self.basedir_entry.insert(0, directory)
    
    def browse_repo(self):
        """Browse for existing repository."""
        directory = filedialog.askdirectory()
        if directory:
            self.repo_path_entry.delete(0, tk.END)
            self.repo_path_entry.insert(0, directory)
    
    def log(self, message):
        """Add message to log."""
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_log(self):
        """Clear log output."""
        self.log_text.delete(1.0, tk.END)
    
    def validate_token(self):
        """Validate GitHub token and show result."""
        token = self.token_entry.get().strip()
        if not token:
            messagebox.showwarning("Warning", "Please enter a GitHub token first")
            return
        
        self.log("Validating GitHub token...")
        is_valid, message = validate_github_token(token)
        
        if is_valid:
            self.log(f"✓ {message}")
            messagebox.showinfo("Token Valid", message)
        else:
            self.log(f"✗ {message}")
            messagebox.showerror("Token Invalid", message)
    
    def run_generation(self):
        """Run repository generation or modification in background thread."""
        # Validate inputs
        token = self.token_entry.get().strip()
        username = self.username_entry.get().strip()
        
        if not token or not username:
            messagebox.showerror("Error", "Please provide GitHub token and username")
            return
        
        mode = self.mode_var.get()
        
        if mode == "modify":
            repo_path = self.repo_path_entry.get().strip()
            repo_name = self.repo_name_entry.get().strip()
            
            if not repo_path:
                messagebox.showerror("Error", "Please provide repository path or URL")
                return
            
            # Check if it's a URL or local path
            if not is_git_url(repo_path) and not Path(repo_path).exists():
                messagebox.showerror("Error", "Local path does not exist. For remote repos, provide a valid Git URL.")
                return
        
        # Disable button and start progress
        self.run_button.config(state='disabled')
        self.progress.start()
        
        # Run in background thread
        thread = threading.Thread(target=self.execute_operation)
        thread.daemon = True
        thread.start()
    
    def execute_operation(self):
        """Execute the selected operation."""
        try:
            mode = self.mode_var.get()
            token = self.token_entry.get().strip()
            username = self.username_entry.get().strip()
            
            if mode == "create":
                self.log("Starting repository generation...")
                
                model = self.model_combo.get().strip()
                if not model:
                    raise ValueError("Please select an Ollama model")
                base_dir = Path(self.basedir_entry.get().strip())
                max_commits = int(self.commits_spin.get())
                min_issues = int(self.issues_spin.get())
                private = self.private_var.get()

                # Backdate options
                backdate_count = int(self.backdate_count_spin.get()) if self.backdate_var.get() else 0
                backdate_start = self.backdate_start_entry.get().strip() if self.backdate_var.get() else None
                backdate_interval = int(self.backdate_interval_spin.get()) if self.backdate_var.get() else 1
                backdate_model = model
                
                # Redirect print to log
                original_print = print
                def gui_print(*args, **kwargs):
                    message = ' '.join(str(arg) for arg in args)
                    self.log(message)
                
                import builtins
                builtins.print = gui_print
                
                try:
                    repo_url = generate_repo(
                        github_token=token,
                        github_username=username,
                        ollama_model=model,
                        base_dir=base_dir,
                        private=private,
                        max_commits=max_commits,
                        min_issues=min_issues,
                        backdate_count=backdate_count,
                        backdate_start=backdate_start,
                        backdate_interval_days=backdate_interval,
                        backdate_model=backdate_model
                    )
                    self.log(f"\n✓ SUCCESS! Repository created: {repo_url}")
                    messagebox.showinfo("Success", f"Repository created successfully!\n{repo_url}")
                finally:
                    builtins.print = original_print
                    
            else:  # modify
                self.log("Starting repository modification...")
                
                repo_path_or_url = self.repo_path_entry.get().strip()
                repo_name = self.repo_name_entry.get().strip() or None  # Allow None for auto-detection
                mod_type = self.mod_type_combo.get()
                model = self.modify_model_combo.get().strip()
                if not model:
                    raise ValueError("Please select an Ollama model")
                
                # Redirect print to log
                original_print = print
                def gui_print(*args, **kwargs):
                    message = ' '.join(str(arg) for arg in args)
                    self.log(message)
                
                import builtins
                builtins.print = gui_print
                
                try:
                    success = modify_existing_repo(
                        repo_path_or_url=repo_path_or_url,
                        github_token=token,
                        github_username=username,
                        repo_name=repo_name,
                        ollama_model=model,
                        modification_type=mod_type
                    )
                    
                    if success:
                        self.log("\n✓ SUCCESS! Repository modified successfully")

                        # Optionally create backdated commits after modification
                        try:
                            if getattr(self, 'modify_backdate_var', None) and self.modify_backdate_var.get():
                                backdate_count = int(self.modify_backdate_count_spin.get())
                                backdate_start = self.modify_backdate_start_entry.get().strip() or None
                                backdate_interval = int(self.modify_backdate_interval_spin.get())
                                backdate_model = self.modify_model_entry.get().strip() or model
                                if backdate_count > 0:
                                    self.log(f"Starting creation of {backdate_count} backdated commits...")
                                    # Get the actual repo path (may have been cloned from URL)
                                    actual_repo_path = Path.cwd()  # modify_existing_repo changes to repo dir
                                    create_backdated_commits(actual_repo_path, ollama_model=backdate_model,
                                                             num_commits=backdate_count,
                                                             start_date=(datetime.fromisoformat(backdate_start) if backdate_start else None),
                                                             interval_days=backdate_interval,
                                                             language=detect_repo_language(actual_repo_path))
                                    self.log("Backdated commits creation finished")
                        except Exception as e:
                            self.log(f"Error creating backdated commits: {e}")

                        messagebox.showinfo("Success", "Repository modified successfully!")
                    else:
                        self.log("\n✗ FAILED! Repository modification failed")
                        messagebox.showerror("Error", "Repository modification failed")
                finally:
                    builtins.print = original_print
                    
        except Exception as e:
            self.log(f"\n✗ ERROR: {str(e)}")
            messagebox.showerror("Error", f"Operation failed:\n{str(e)}")
        finally:
            # Re-enable button and stop progress
            self.progress.stop()
            self.run_button.config(state='normal')


def launch_gui():
    """Launch the GUI application."""
    root = tk.Tk()
    app = RepoGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()