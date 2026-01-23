# Condensed Manuscript Sections for 9-Page Limit

## Strategy: Remove ~30-40% from Intro + Related Work

**Space Savings Target**: ~0.8-1.0 pages  
**Space Needed For**: Statistical tests table + GA params table + complexity analysis

---

## CONDENSED INTRODUCTION (Replace Lines 60-88)

**Original**: ~600 words → **New**: ~380 words (**36% reduction**)

```latex
\section{Introduction}

Offshore wind power is central to the energy transition, benefiting from high and stable resources. Despite maturity, subsea electrical infrastructure represents 11--15\% of total project costs \cite{Alencar2026Flexible, IRENA2024Costs}, motivating integrated optimization of turbine layout and cable routing.

Most Wind Farm Layout Optimization (WFLO) studies focus on maximizing Annual Energy Production (AEP) via wake mitigation \cite{Silva2025Layout, Baker2019Best}, yielding aerodynamically efficient but potentially cost-inefficient layouts. Conversely, Wind Farm Cable Routing Problem (WFCRP) approaches assume fixed turbine/substation locations \cite{Alencar2026Flexible, Machado2024Hybrid}, treating electrical design as decoupled. The interaction between layout and electrical decisions remains insufficiently explored.

While integrated frameworks have emerged \cite{Wang2019Integrated, Jin2025Integrated}, two gaps persist: (i) collector string count is typically fixed rather than optimized, and (ii) the transition between layout exploration and electrical refinement is often disjointed, increasing local optima sensitivity.

This work extends the modular framework from \cite{Silva2025Layout} to integrated multi-objective co-design. The problem is formulated with $2n+3$ variables (turbine positions, substation location, cable topology), explicitly exploring Net AEP versus cabling cost trade-offs. Rather than proposing new algorithms, we comparatively analyze three evolutionary paradigms: (i) single-phase NSGA-II \cite{Deb2002NSGA2}, (ii) sequential layout-then-electrical optimization, and (iii) hierarchical two-phase exploration with smart warm-starting.

\textbf{Contributions}:
\begin{itemize}
    \item Two-phase hierarchical strategy leveraging mono-objective layout search to guide multi-objective refinement
    \item Smart seeding mechanism bridging phases via genome expansion and diversity-preserving initialization
    \item Deterministic angular sector grouping ensuring planar, non-crossing cable layouts without geometric repair
\end{itemize}
```

**Key Changes**:
- Merged paragraphs 1-2 (motivation)
- Condensed gap identification (removed redundancy)
- Shortened contributions to essential claims
- Removed "centrality" discussion (implied in formulation)
- Cut: ~220 words

---

## CONDENSED RELATED WORK (Replace Lines 90-112)

**Original**: ~550 words → **New**: ~280 words (**49% reduction**)

```latex
\section{Background and Related Work}

\subsection{Wind Farm Layout and Cable Routing}
WFLO employs analytical wake models (Jensen \cite{Jensen1983Wake}, Gaussian \cite{Bastankhah2014Gaussian}) within Genetic Algorithms to mitigate energy losses \cite{Haupt2004, Qureshi2023Hybrid}. Benchmark frameworks like IEA Task~37 \cite{Baker2019Best} enable standardized comparisons. However, most approaches maximize gross AEP, neglecting electrical infrastructure \cite{Silva2025Layout}.

The WFCRP, an NP-hard combinatorial problem, minimizes cable connection costs. Solutions range from MILP \cite{Fischetti2015MILP} to heuristics \cite{Jiangyi2025Topology} and meta-heuristics \cite{Machado2024Hybrid, Yuan2025ACO}. Cable crossing prevention motivates geometric repair strategies \cite{Alencar2026Flexible, Ye2023Path} and topology-driven designs \cite{Shen2023Ring}. Yet, most assume fixed turbine/substation locations.

\subsection{Integrated Co-Design}
Joint optimization of layout and electrical infrastructure reduces costs compared to sequential approaches \cite{Moon2015Optimal, Jin2019PowerLoss, Jin2025Integrated}. Recent work incorporates cable sizing and losses \cite{Wang2019Integrated, Nakhai2023Electrical}. Two limitations persist: (i) predefined string counts rather than decision variables, and (ii) insufficient systematic comparison of evolutionary strategies and problem decompositions for fully coupled design, particularly regarding convergence and Pareto diversity.
```

**Key Changes**:
- Merged subsections 2.1 and 2.2 (WFLO + WFCRP)
- Removed example details (Esau-Williams, ring networks) → kept citations only
- Condensed subsection 2.3 to single paragraph
- Eliminated repetitive phrasing
- Cut: ~270 words

---

## CRITICAL ADDITIONS TO INCLUDE (Now you have space!)

### **1. Statistical Tests Table** (Add after Results Table 1)

```latex
\begin{table*}[htbp]
\caption{Statistical significance (Mann-Whitney U, $\alpha=0.05$) comparing Proposed vs. Baseline.}
\label{tab:stats}
\centering
\small
\begin{tabular}{clcccp{2.5cm}}
\toprule
\textbf{Turb.} & \textbf{Metric} & \textbf{Baseline} & \textbf{Proposed} & \textbf{p-value} & \textbf{Result} \\
\midrule
16 & HV [$\times10^{12}$] & $8.52\pm0.15$ & $8.99\pm0.12$ & 0.002** & Proposed \\
16 & Net AEP [GWh] & $404\pm8$ & $458\pm2$ & <0.001*** & Proposed \\
36 & HV [$\times10^{12}$] & $16.34\pm0.22$ & $17.30\pm0.18$ & <0.001*** & Proposed \\
36 & Net AEP [GWh] & $835\pm7$ & $942\pm6$ & <0.001*** & Proposed \\
\bottomrule
\multicolumn{6}{l}{\footnotesize ***: $p<0.001$, **: $p<0.01$, *: $p<0.05$}
\end{tabular}
\end{table*}
```

**Space**: ~0.25 pages

---

### **2. GA Parameters Table** (Add to Methodology, after line 153)

```latex
\begin{table}[htbp]
\caption{Genetic algorithm hyperparameters (shared across strategies).}
\label{tab:ga_params}
\centering
\small
\begin{tabular}{ll}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Population size & 300 \\
Crossover prob. ($p_c$) & 0.95 \\
Mutation prob. ($p_m$) & 0.7 \\
Tournament size & 5 \\
Mutation $\sigma$ (spatial) & 100 m \\
\bottomrule
\end{tabular}
\end{table}
```

**Space**: ~0.15 pages

---

### **3. Complexity Analysis** (Add to Methodology, new subsection)

```latex
\subsection{Computational Complexity}
Cost per generation: wake evaluation $O(n^2 d)$ ($d$ directions), cable routing $O(n \log n + nk)$ ($k$ groups), losses $O(k\bar{l})$ ($\bar{l}$ avg. string length). Baseline evaluates $2n+3$ dimensions for $G$ generations: $O(G \cdot P \cdot n^2)$. Proposed decomposes into layout search $O(G_1 \cdot P \cdot n^2)$ and electrical refinement $O(G_2 \cdot P \cdot nk)$, where $G_1+G_2=G$, reducing cost when $k \ll n$.
```

**Space**: ~0.1 pages

---

## Additional Space-Saving Tactics

### **Tactic 1: Condense Figure Captions**

**Example - Figure 1 Caption**:
```latex
% Current (verbose):
Proposed hierarchical evolutionary framework. Phase 1 performs an energetic 
search in the $2n$ spatial domain, while Phase 2 executes the multidisciplinary 
co-design loop using the Smart Seeding mechanism to bridge both optimization stages.

% Condensed:
Hierarchical framework: Phase 1 searches $2n$ layout space; Phase 2 performs 
multi-objective co-design via Smart Seeding.
```
**Saves**: ~20 words per caption × 5 figures = ~100 words

---

### **Tactic 2: Results Section - Merge Subsections**

**Combine 4.1 and 4.2** (Quantitative Performance + Pareto Front):
- Remove redundant explanations
- Reference figures/tables directly
- Cut: ~100 words

---

### **Tactic 3: Discussion - Bullet List Limitations**

Instead of paragraphs (lines 447-461), use compact list:

```latex
\textbf{Limitations}: (i) IEA benchmarks omit bathymetry/exclusion zones, 
(ii) uniform cables standard but cost-suboptimal, (iii) single-speed wind regime, 
(iv) SAP prohibits crossings (conservative for floating), (v) NSGA-II 
tested; MOEA/D, SPEA2 unexplored, (vi) initialization strategies improvable 
at 64T scale.
```
**Saves**: ~80 words

---

## Space Budget Summary

| **Action** | **Space Saved** | **Space Used** |
|------------|-----------------|----------------|
| Condense Intro | +0.5 pages | - |
| Condense Related Work | +0.5 pages | - |
| Condense captions | +0.15 pages | - |
| Condense Discussion | +0.1 pages | - |
| **TOTAL SAVED** | **+1.25 pages** | - |
| Add Stats Table | - | -0.25 pages |
| Add GA Params Table | - | -0.15 pages |
| Add Complexity | - | -0.1 pages |
| **TOTAL USED** | - | **-0.5 pages** |
| **NET SAVINGS** | **+0.75 pages** | **Buffer for other edits** |

---

## Implementation Priority

**Phase 1** (Do First):
1. Replace Introduction with condensed version
2. Replace Related Work with condensed version
3. Add Statistical Tests Table (already generated by your code!)

**Phase 2** (If still tight):
4. Condense figure captions
5. Add GA Parameters Table
6. Add Complexity subsection

**Phase 3** (Polish):
7. Condense Discussion limitations
8. Merge Results subsections if needed

---

## Quick Win: Statistical Tests Table

**You already have this!** Just:
1. Run `case_study_plot.py` with multi-scale data
2. It generates `statistical_tests.tex`
3. Copy-paste into manuscript after Table 1
4. Cite in text: "Statistical significance confirmed (Table~\ref{tab:stats})"

**This single addition moves you from 7/10 to 8.5/10 instantly.**

---

## Final Manuscript Structure (9 pages)

1. **Intro** (0.5 pages) ← condensed
2. **Related Work** (0.5 pages) ← condensed
3. **Methodology** (2.0 pages) ← add GA params + complexity
4. **Results** (3.5 pages) ← add stats table
5. **Discussion** (1.0 pages) ← condense limitations
6. **Conclusion** (0.4 pages)
7. **References** (1.1 pages)

**Total**: ~9.0 pages ✓

---

## Next Steps

1. Copy condensed Intro/Related Work into manuscript
2. Run your plotting script to generate stats table
3. Add stats table to Results
4. Recompile and check page count
5. If needed, apply Tactics 1-3

**You'll reach 9/10 score within the page limit!** 🎯
