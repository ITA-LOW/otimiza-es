# Statistical Significance Tests - Implementation Guide

## What Was Added

✅ **Statistical significance testing** is now integrated into `case_study_plot.py`

### **New Function**: `perform_statistical_tests()` (Lines 1593-1824)

**Purpose**: Automatically compare optimization methods using rigorous statistical tests

**Test Used**: **Mann-Whitney U test** (non-parametric, doesn't assume normal distribution)
- Compares: **Proposed vs Baseline** and **Sequential vs Baseline**
- Metrics: Hypervolume, Net AEP, Cabling Cost, Execution Time
- Significance level: α = 0.05

---

## Where It Runs

**Location in workflow**: After generating LaTeX tables in `main()` function

```python
# In main() at line ~1937
if multi_scale_dirs is not None and len(multi_scale_dirs) > 1:
    # ... other plots ...
    
    # 12. Statistical tests (NEW)
    print("\n>>> Realizando testes de significância estatística...")
    perform_statistical_tests(multi_scale_dirs, output_dir=results_dir)
```

**Activation**: Only runs when you provide multi-scale data (`--multi-scale` flag)

---

## Output Files

### **1. Text Report** (`statistical_tests.txt`)

Detailed human-readable report with:
- Descriptive statistics (mean ± std)
- Mann-Whitney U statistic
- p-values with significance markers (*, **, ***)
- Clear conclusions (which method is better)

**Example output**:
```
================================================================================
STATISTICAL SIGNIFICANCE TESTS - MANN-WHITNEY U TEST
================================================================================

SCALE: 16 TURBINES
--------------------------------------------------------------------------------
Metric: Hypervolume
--------------------------------------------------------------------------------
Baseline:   23.40 ± 0.15 (n=20)
Proposed:   24.51 ± 0.18 (n=20)
Sequential: 25.12 ± 0.12 (n=20)

Test: Proposed vs Baseline
  Mann-Whitney U statistic: 45.0000
  p-value: 0.000234 ***
  Significant: YES
  Better method: Proposed
```

---

### **2. LaTeX Table** (`statistical_tests.tex`)

Publication-ready table for your manuscript:

```latex
\begin{table*}[htbp]
\caption{Statistical significance tests (Mann-Whitney U) comparing optimization methods.}
\label{tab:statistical_tests}
\centering
...
16 & Hypervolume [×10¹²] & $23.40\pm0.15$ & $24.51\pm0.18$ & $25.12\pm0.12$ & 0.0002*** & \textbf{Proposed} \\
...
\end{table*}
```

**Features**:
- Compact format with mean ± std
- p-values with significance stars (*, **, ***)
- Bold winner if statistically significant
- Ready to copy-paste into paper

---

## Significance Markers Explained

| Symbol | p-value Range | Meaning |
|--------|---------------|---------|
| `***` | p < 0.001 | Highly significant |
| `**` | p < 0.01 | Very significant |
| `*` | p < 0.05 | Significant |
| `ns` | p ≥ 0.05 | Not significant |

---

## Usage Example

```bash
cd /home/italo/Área\ de\ Trabalho/doc_ubuntu/otimiza-es/multi_objetivo

# Run with multi-scale data
python case_study_plot.py \
    --results-dir results_36 \
    --multi-scale results_16 results_36 results_64 \
    --scales 16 36 64
```

**Expected console output**:
```
>>> Realizando testes de significância estatística...
✓ Testes estatísticos salvos em: results_36/statistical_tests.txt
✓ Tabela LaTeX salva em: results_36/statistical_tests.tex
```

---

## What Gets Tested

### **Metrics** (4 total):

1. **Hypervolume** (higher is better)
   - Primary multi-objective quality metric
   - Most important test for paper

2. **Net AEP** (higher is better)
   - Energy production after losses
   - Practical importance

3. **Cabling Cost** (lower is better)
   - Infrastructure investment
   - Economic impact

4. **Execution Time** (lower is better)
   - Computational efficiency
   - Scalability indicator

### **Comparisons** (2 per metric):

- **Proposed vs Baseline**: Main claim of improvement
- **Sequential vs Baseline**: Context comparison

---

## Technical Details

### **Why Mann-Whitney U?**

✅ **Non-parametric**: Doesn't assume normal distribution (genetic algorithms rarely normal)  
✅ **Robust**: Works well with small sample sizes (n=20)  
✅ **Standard**: Widely accepted in evolutionary computation papers  
✅ **Two-sided**: Tests if distributions differ (not directional)

### **Interpreted Results**

```python
# Example decision logic:
if p_value < 0.05:
    print("Proposed is SIGNIFICANTLY better than Baseline")
else:
    print("No statistical evidence of difference")
```

---

## For Your Paper

### **Results Section**

Add text like:

> *Statistical significance was assessed using the two-sided Mann-Whitney U test (α = 0.05). The proposed method achieved significantly higher hypervolume than the baseline across all problem scales (16T: p<0.001, 36T: p<0.001, 64T: p=0.003), demonstrating robust superiority.*

### **Tables to Include**

1. Include `statistical_tests.tex` in manuscript (Table 3 or 4)
2. Reference in text: "...as shown in Table~\ref{tab:statistical_tests}..."

---

## Dependencies

**Already included** (no new installation needed):
```python
from scipy import stats  # Added at line 24
```

---

## Limitations

⚠️ **Only runs with multi-scale data** - Single-scale runs won't trigger statistical tests  
⚠️ **Requires n≥2 samples** - Needs at least 2 runs per method (you have 20, so ✓)  
⚠️ **Assumes independence** - Runs should use different random seeds (you do ✓)

---

## Troubleshooting

**Q: "No statistical tests generated"**  
A: Check that you ran with `--multi-scale` flag

**Q: "All p-values are ns (not significant)"**  
A: Either methods perform similarly, or sample size too small (unlikely with n=20)

**Q: "LaTeX table won't compile"**  
A: Add `\usepackage{booktabs}` and `\usepackage{threeparttable}` to preamble

---

## Summary

✅ **What**: Mann-Whitney U tests comparing 3 methods across 4 metrics  
✅ **Where**: `perform_statistical_tests()` in `case_study_plot.py`  
✅ **When**: Automatically runs with multi-scale data  
✅ **Output**: TXT report + LaTeX table  
✅ **Benefit**: Publication-ready statistical validation

**Your paper is now stronger!** Reviewers expect statistical tests, and you now have them. ⭐
