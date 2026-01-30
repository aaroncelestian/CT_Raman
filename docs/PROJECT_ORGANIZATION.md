# Project Organization Summary

## Overview
This document describes the organization of the CT_Raman project after restructuring on January 29, 2026.

## Changes Made

### 1. Removed Duplicate Files
- Deleted 18 duplicate files with " copy" suffix (15 Python scripts, 3 requirements files)
- All duplicates were exact copies of the original files

### 2. Created Directory Structure
```
CT_Raman/
├── src/           # All source code
├── data/          # Data files (CSV, pickle)
├── notebooks/     # Jupyter notebooks
├── reports/       # Analysis reports
├── tests/         # Test scripts
├── docs/          # Documentation
└── images/        # Image outputs
```

### 3. File Organization

#### Source Code (`src/`)
- **`src/analysis/`** - Core analysis modules
  - `ct_enhancement.py` - CT image enhancement
  - `ct_raman_correlation.py` - CT-Raman correlation analysis
  - `dog_stone_isolation.py` - Stone isolation algorithms
  - `enhanced_stone_analysis.py` - Enhanced analysis methods
  - `stone_layer_analysis.py` - Layer-by-layer analysis

- **`src/interactive/`** - Interactive tools and applications
  - `interactive_annotation.py` - Interactive annotation tool
  - `interactive_stone_tuning.py` - Parameter tuning interface
  - `stone_analysis_app.py` - Web application
  - `stone_analysis_standalone.py` - Main standalone application (80KB)
  - `stone_analysis_widget.py` - Widget-based interface

- **`src/utils/`** - Utilities and diagnostics
  - `compare_isolation_methods.py` - Method comparison tools
  - `threshold_diagnostic.py` - Threshold diagnostics

#### Data Files (`data/`)
- Line scan CSV files (4 files)
- Metadata pickle files (2 files)
- Optimized settings pickle files (2 files)

#### Reports (`reports/`)
- `annotation_based_report.txt`
- `ct_raman_correlation_report.txt`
- `enhanced_stone_report.txt`
- `stone_layer_report.txt`

#### Tests (`tests/`)
- `run_stone_analysis.py`
- `simple_test.py`

#### Documentation (`docs/`)
- `README_widget.md` - Widget documentation
- `PROJECT_ORGANIZATION.md` - This file

## Migration Notes

### Import Path Changes
Old imports need to be updated to reflect the new structure:

**Before:**
```python
from ct_enhancement import CTImageAnalyzer
```

**After:**
```python
from src.analysis.ct_enhancement import CTImageAnalyzer
```

### Running Scripts
Scripts should now be run from the project root:

```bash
# Analysis scripts
python src/analysis/ct_enhancement.py

# Interactive tools
python src/interactive/stone_analysis_standalone.py

# Tests
python tests/run_stone_analysis.py
```

## Benefits of New Structure

1. **Clarity** - Clear separation between analysis, interactive tools, and utilities
2. **Maintainability** - Easier to locate and modify specific components
3. **Scalability** - Simple to add new modules in appropriate directories
4. **No Duplicates** - Eliminated confusion from duplicate files
5. **Professional** - Standard Python project structure

## Next Steps

If you need to:
- **Add new analysis methods** → Place in `src/analysis/`
- **Create new interactive tools** → Place in `src/interactive/`
- **Add utilities** → Place in `src/utils/`
- **Store data** → Place in `data/`
- **Generate reports** → Will be saved to `reports/`
- **Create notebooks** → Place in `notebooks/`
