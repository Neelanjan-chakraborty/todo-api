# Automated Repository Generator

This script creates high-quality Git repositories that meet SWE-Bench evaluation criteria for training AI models on coding tasks. It uses Ollama to generate meaningful code and projects.

## Features

- **AI-Generated Projects**: Uses Ollama to create diverse, realistic software projects
- **Complete Repository Structure**: Generates proper project structure with source code, tests, documentation, and configuration
- **CI/CD Integration**: Automatically sets up GitHub Actions workflows
- **Test Coverage**: Creates comprehensive test suites
- **Git History**: Makes multiple commits with meaningful messages referencing GitHub issues
- **Issue Tracking**: Creates GitHub issues for project development
- **Evaluation Compliance**: Designed to score highly on the provided repo evaluation metrics

## Prerequisites

1. **Python 3.8+**
2. **Git** installed and configured
3. **Ollama** running with a code-capable model (e.g., llama3.1)
4. **GitHub Account** with a personal access token

## Installation

1. Clone or download the `auto_repo_generator.py` script
2. Install dependencies (automatically handled on first run)

## Usage

```bash
python auto_repo_generator.py \
  --github-token YOUR_GITHUB_TOKEN \
  --github-username YOUR_USERNAME \
  --ollama-model llama3.1 \
  --base-dir ./generated_repos \
  --max-commits 15 \
  --min-issues 5
```

### Parameters

- `--github-token`: **Required**. GitHub personal access token with repo creation permissions
- `--github-username`: **Required**. Your GitHub username
- `--ollama-model`: Ollama model to use (default: llama3.1)
- `--base-dir`: Directory to store generated repositories (default: ./generated_repos)
- `--private`: Make repositories private (default: public)
- `--max-commits`: Maximum number of development commits (default: 10)
- `--min-issues`: Minimum number of GitHub issues to create (default: 3)

## How It Works

1. **Project Generation**: Uses Ollama to generate a creative project idea with specifications
2. **Code Creation**: Generates main application code, tests, documentation, and configuration
3. **Repository Setup**: Initializes Git, creates GitHub repository, and pushes initial commit
4. **Issue Creation**: Creates GitHub issues for development tasks
5. **Development Simulation**: Makes multiple commits that reference issues, simulating real development
6. **CI/CD Setup**: Configures GitHub Actions for automated testing

## Evaluation Metrics Compliance

The generated repositories are designed to score highly on these SWE-Bench evaluation criteria:

- **Test Coverage** (40 points): High test-to-code ratios with comprehensive test suites
- **CI/CD** (15 points): GitHub Actions workflows for automated testing
- **Test Frameworks** (15 points): Proper testing frameworks (pytest, Jest, etc.)
- **Git Activity** (15 points): Multiple commits over time with recent activity
- **Issue Tracking** (15 points): Commits that reference GitHub issues

## Generated Project Types

The script can generate various types of projects:
- Web APIs (Python FastAPI, Node.js Express)
- CLI tools
- Data processing utilities
- Library packages
- Configuration management tools

## Ethical Use

This tool is intended for:
- **Academic Research**: Training AI models for code generation and understanding
- **Educational Purposes**: Learning software engineering practices
- **AI Model Development**: Creating diverse training datasets

**Important**: Only use for ethical purposes. Do not create repositories that could be confused with real projects or violate platform terms of service.

## Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running: `ollama serve`
- Verify the model is available: `ollama list`
- Try a different model if generation fails

### GitHub API Issues
- Check your token has `repo` scope
- Verify rate limits haven't been exceeded
- Ensure the repository name doesn't conflict with existing repos

### Permission Errors
- Run with appropriate file system permissions
- Ensure Git is properly configured

## License

MIT License - See LICENSE file for details.</content>
<parameter name="filePath">D:\Softwaes\personal\GHARI\README_AUTO_REPO.md