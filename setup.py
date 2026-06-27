"""
Setup configuration for NBA Churn Retention Engine package.
"""
from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="nba-churn-retention-engine",
    version="1.0.0",
    author="Portfolio Project",
    description="Next-Best-Action Churn Retention Engine for SaaS Analytics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourname/nba-churn-retention-engine",
    packages=find_packages(where=".", include=["src*", "api*"]),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.1.0",
        "catboost>=1.2.0",
        "shap>=0.44.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.4.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "mlflow>=2.8.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0.1",
        "scipy>=1.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.26.0",
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nba-pipeline=src.pipeline:run_full_pipeline",
            "nba-score=src.score:score_customers",
            "nba-report=src.utils.report_generator:run_report_generation",
            "nba-monitor=src.monitoring.drift_monitor:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Data Scientists",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="churn prediction, next-best-action, SHAP, explainable AI, customer retention, SaaS analytics",
)
