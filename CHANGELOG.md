# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Dark theme configuration for Streamlit dashboard
- GitHub Actions CI/CD pipeline for automated testing
- Project documentation (CONTRIBUTING.md, LICENSE)
- Type hints and code quality improvements

### Changed
- Enhanced README with badges and better structure
- Improved project organization and documentation

### Fixed
- Fixed configuration for dark theme default

## [1.0.0] - 2024-06-18

### Added
- Initial release of Demand Forecast Explainer
- LightGBM-based sales forecasting model with quantile regression
- SHAP-powered feature explainability dashboard
- Streamlit web interface with three main pages:
  - Store Forecast: Historical sales, predictions, and SHAP drivers
  - Model Performance: Metrics, feature importance, residual analysis
  - Batch Upload: CSV-based batch scoring
- Preprocessing pipeline for Rossmann Store Sales dataset
- Feature engineering with lag and rolling statistics
- Confidence intervals using Q10/Q90 quantile regression

### Features
- Time-series forecasting with 9.4% test MAPE
- Interactive visualizations with Plotly
- What-if analysis for promotional scenarios
- Batch inference capability
- Pre-trained model artifacts

---

## Version History

### Pre-release Development
- Data exploration and EDA
- Feature engineering and selection
- Model selection and hyperparameter tuning
- SHAP explainability integration
- Dashboard prototype and refinement

---

## How to Upgrade

### From 1.0.0 to Unreleased
```bash
git pull origin main
pip install -r requirements.txt
```

---

## Reporting Issues

Found a bug or have a feature request? Please open an [issue](https://github.com/priyal2307/demand-forecast-explainer/issues).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.
