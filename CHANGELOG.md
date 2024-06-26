# Changelog

This repo is based on [Benchmark Evaluation of Counterfactual Algorithms for XAI: From a White box to a Black box](https://github.com/LeonChou5311/Counterfactual-benchmark). All notable changes to this project will be documented in this file. 

## [0.2.0] - 2022-12-20

### Added

- Add a notebook for model training `1_AE_model_training.ipynb`
- Add new attack methods (DeepFool, Carlini, LowProFool, Boundary, Ho) 
- Add `requirements.txt` to projects
- Add three new models (Linear SVC, Logistic Regression and Neural network 2) to `utils/models.py`
- Add AE trained models to `./saved_models/`
- Add original and generated datapoints to folder `./datapoints/` 
- Add information about Adversarial Robustness Toolbox in `art.md`
- Add simplified model for test
 

### Changed

- Update `README` and `CHANGELOG`
- Move output processing function to `save.py`

### Deprecated

- Remove irrelevant materials from CF

## [0.1.0] - 2022-11-30

### Added

- Fork Counterfactual Benchmark as the basis for further development
- Add `CHANGELOG.md` to record evolving changes in the development

### Changed

- Add more patterns in `.gitignore`

### Deprecated

- Remove cache folders (`.vscode/`, `.ipynb_checkpoints/`, `__pycache__/`) out of git track lists



[0.1.0]: https://github.com/ZhipengHe/Imperceptibility-in-Adversarial-attack/commits/v0.1.0
[0.2.0]: https://github.com/ZhipengHe/Imperceptibility-in-Adversarial-attack/compare/v0.1.0...v0.2.0