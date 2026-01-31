# Repository Evaluator

A tool that helps you figure out if a repository is a good fit for creating SWE-Bench+ samples. It looks at both the repository structure and pull request history to give you insights into code quality, test coverage, and how the project is maintained.

## What It Does

This tool analyzes repositories from GitHub or Bitbucket and tells you:

- **What's in the repo**: File structure, lines of code, programming languages used
- **Testing setup**: Whether tests exist, what test frameworks are used, and coverage if available
- **CI/CD**: Whether there's automated testing and deployment
- **PR quality**: How many pull requests meet certain quality criteria, and why some don't
- **Project activity**: Recent commits and how issues are tracked

## Features

- Works with both GitHub and Bitbucket repositories
- Automatically clones repos if you don't have them locally
- Handles network issues with automatic retries
- Outputs results in human-readable format or JSON
- Supports filtering PRs by date and limiting how many to analyze

## What You Need

- Python 3.6 or newer
- The `requests` library (install with `pip install requests`)
- Git installed on your system
- An API token from GitHub or Bitbucket (helps avoid rate limits and access private repos)

## Getting Started

### Installation

Just install the required dependency:

```bash
pip install requests
```

### Basic Usage

The simplest way to use it:

```bash
# GitHub repository
python repo_evaluator.py owner/repo-name --token $GITHUB_TOKEN

# Bitbucket repository
python repo_evaluator.py bitbucket:owner/repo-name --token $BITBUCKET_TOKEN --platform bitbucket

# Or just use the full URL and it'll figure out the platform
python repo_evaluator.py https://bitbucket.org/owner/repo --token $BITBUCKET_TOKEN
```

### Examples

Here are some real examples:

```bash
# Check out a GitHub repo
python repo_evaluator.py microsoft/vscode --token $GITHUB_TOKEN

# Bitbucket repo with explicit platform
python repo_evaluator.py bitbucket:owner/repo --token $BITBUCKET_TOKEN --platform bitbucket

# Use a local copy instead of cloning
python repo_evaluator.py owner/repo --repo-path /path/to/local/repo --token $GITHUB_TOKEN

# Get JSON output for scripting
python repo_evaluator.py owner/repo --token $GITHUB_TOKEN --json

# Only look at the first 50 PRs
python repo_evaluator.py owner/repo --token $GITHUB_TOKEN --max-prs 50

# Only analyze PRs merged after a certain date
python repo_evaluator.py owner/repo --token $GITHUB_TOKEN --start-date 2024-01-01
```

## Command Options

**Required:**
- `repo` - The repository in `owner/repo-name` format or a full URL

**Optional:**
- `--token` - Your API token (GitHub or Bitbucket)
- `--platform` - Force a platform: `auto`, `github`, or `bitbucket` (defaults to auto-detecting)
- `--repo-path` - Use a local repository instead of cloning
- `--min-test-files` - Minimum test files each PR should have (default: 1)
- `--max-non-test-files` - Maximum non-test files per PR (default: 100)
- `--min-code-changes` - Minimum code changes per PR (default: 10)
- `--max-prs` - Limit how many PRs to analyze (default: all of them)
- `--start-date` - Only look at PRs merged after this date (YYYY-MM-DD format)
- `--json` - Output results as JSON instead of human-readable text
- `--output` - Save the output to a file

## Repository Formats

You can specify repositories in a few different ways:

- Simple: `owner/repo-name` (assumes GitHub)
- With prefix: `bitbucket:owner/repo-name` or `github:owner/repo-name`
- Full URL: `https://github.com/owner/repo` or `https://bitbucket.org/owner/repo`

## Understanding the Output

### Human-Readable Output

When you run it normally, you'll see:
- Repository name and primary programming language
- File counts and lines of code breakdown
- Test coverage information (if available)
- CI/CD status
- Test frameworks detected
- Git activity stats
- PR analysis showing how many passed/failed the quality checks
- Breakdown of why PRs were rejected (if any)

### JSON Output

Add `--json` to get machine-readable output that's easy to parse or process:

```bash
python repo_evaluator.py owner/repo --token $TOKEN --json
```

## Why PRs Get Rejected

The tool checks PRs against several criteria. Here's what it looks for and why PRs might not make the cut:

- `fewer_than_min_test_files` - Not enough test files in the PR
- `more_than_max_non-test-files` - Too many non-test files changed
- `code_changes_not_sufficient` - Not enough actual code changes
- `issue_is_a_pr` - The linked "issue" is actually another PR
- `issue_is_not_closed` - The linked issue isn't closed
- `issue_word_count` - Issue description is too short or too long
- `content_not_in_english` - PR title/description doesn't appear to be in English
- `rust_embedded_tests` - Rust source files contain embedded tests (not allowed)
- `merge_date` - PR was merged before your specified start date
- `full_patch_retrieval` - Couldn't download the full diff

## Getting API Tokens

### GitHub

1. Go to your GitHub Settings → Developer settings → Personal access tokens
2. Click "Generate new token"
3. Give it the `repo` scope
4. Copy the token and use it with `--token`

### Bitbucket

1. Go to your repository's Settings → Access tokens
2. Create a new access token
3. Make sure it has repository read permissions
4. Use the token with `--token`

**Why you need tokens:** They let you access private repos and give you higher rate limits. Without one, you might hit API limits pretty quickly.

## Common Issues

### Network Problems

If you see DNS or connection errors, the tool will automatically retry a few times. If it keeps failing:
- Check your internet connection
- Make sure the repository URL is correct
- Wait a bit and try again

### Rate Limits

If you hit rate limits:
- Make sure you're using `--token` with a valid API token
- Wait a few minutes before trying again
- Try using `--max-prs` to analyze fewer PRs at once

### No PRs Found

If the tool says it analyzed 0 PRs:
- The repo might not have any merged PRs
- Your token might not have the right permissions
- Check if you used `--start-date` and it's filtering everything out
- Look at the error messages for clues

## Limitations

- Needs internet access to fetch data from GitHub/Bitbucket APIs
- Can take a while for repos with lots of PRs
- Some detection depends on common project structures and naming conventions
- Bitbucket support is good but might have some differences compared to GitHub

## Contributing

Found a bug or want to add a feature? Contributions are welcome! Just make sure your code matches the existing style and that everything still works.

## Need Help?

If you run into problems or have questions, feel free to open an issue or reach out to the maintainers.
