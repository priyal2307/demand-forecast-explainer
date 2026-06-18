# Contributing to Demand Forecast Explainer

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions. We're committed to providing a welcoming and inspiring community for all.

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check the [issues](https://github.com/priyal2307/demand-forecast-explainer/issues) list to avoid duplicates.

**Bug Report Template:**
```
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
A clear description of what you expected to happen.

**Environment**
- OS: [e.g., macOS, Ubuntu, Windows]
- Python version: [e.g., 3.9, 3.10]
- Streamlit version: [e.g., 1.30]
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, provide:
- Clear use case and motivation
- Examples of how the feature works in other projects
- Implementation approach (if you have thoughts)

### Pull Requests

We actively welcome your pull requests!

**Development Setup:**
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/demand-forecast-explainer.git
cd demand-forecast-explainer

# Create a feature branch
git checkout -b feature/your-feature-name

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies + dev tools
pip install -r requirements.txt
pip install black pytest pylint
```

**Before Submitting:**

1. **Code Style**
   - Format with `black`: `black src/ app/`
   - Follow PEP 8 conventions
   - Add type hints where possible

2. **Testing**
   - Test your changes locally: `streamlit run app/copilot.py`
   - For ML pipeline changes, verify `src/train.py` completes without errors
   - Add docstrings to new functions

3. **Commit Messages**
   ```
   Use clear, descriptive commit messages
   - Add feature: Better SHAP visualization
   - Fix: Handle edge case in lag calculation
   - Docs: Update README with API examples
   ```

4. **PR Description**
   Include:
   - What problem does this solve?
   - How does it work?
   - Any breaking changes?
   - Related issues: Closes #123

**PR Review Process:**
- All PRs require review before merge
- Address feedback promptly
- Code should pass all checks

## Project Structure

```
demand-forecast-explainer/
├── app/
│   └── copilot.py          # Streamlit dashboard
├── src/
│   ├── clean.py            # Data preprocessing
│   ├── features.py         # Feature engineering
│   ├── train.py            # Model training & SHAP
│   └── inference.py        # Prediction pipeline
├── data/
│   ├── raw/                # Original data
│   └── processed/          # Cleaned data
├── artifacts/              # Trained models & metrics
├── requirements.txt        # Python dependencies
└── README.md
```

## Development Guidelines

### Adding New Features

1. **Data Features** → Modify `src/features.py`
2. **Dashboard Pages** → Add to `app/copilot.py`
3. **Model Changes** → Update `src/train.py`
4. **Inference Logic** → Modify `src/inference.py`

### Testing Your Changes

```bash
# Test data pipeline
python3 src/clean.py
python3 src/features.py  # If standalone testing possible

# Test Streamlit app
streamlit run app/copilot.py

# Test inference
python3 -c "from src.inference import Forecaster; f = Forecaster(); print(f'Model loaded successfully')"
```

### Documentation

- Update docstrings for new/modified functions
- Update README.md if adding user-facing features
- Add comments for complex logic
- Keep CHANGELOG.md updated

## Licensing

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to:
- Open an issue for discussion
- Reach out in PR comments
- Check existing issues for similar topics

## Recognition

Contributors will be recognized in:
- Pull request credit
- CHANGELOG.md
- Project README (if substantial contribution)

Thank you for making this project better! 🎉
