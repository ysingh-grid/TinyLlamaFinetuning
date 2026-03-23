# TinyLlama Advanced Tasks - PDF Reports Index

**Generated:** March 19, 2026  
**Total Size:** 1.1 MB (5 reports)

---

## 📄 Report 1: LoRA Rank Ablation Study (210 KB)

**File:** `task1_report.pdf`

**Summary:** Systematic evaluation of LoRA rank hyperparameters (4, 8, 16, 32, 64) on Apple Silicon.

**Key Results:**
- Pareto-optimal rank: **4** (98.9 tok/s, loss 1.355)
- Anomaly discovered: ranks 8/16 are 2-3× slower despite similar memory
- Memory scaling: +94 MB from rank 4→64 (only 3%)

**Contents:**
1. Task description & methodology
2. Comparison matrix table (5 ranks × 7 metrics)
3. Visualization: memory/speed curves + test loss
4. Technical implementation & code walkthrough
5. Hypothesis for speed anomaly

---

## 📄 Report 2: Activation Vector Steering (276 KB)

**File:** `task2_report.pdf`

**Summary:** Post-training safety control via middle-layer activation steering.

**Key Results:**
- Strongest effect: **Layer 12** (variance reduction -2.147)
- Effective range: layers 10-15 for safety intervention
- Tested scales: {0.5, 1.0, 1.5} on 20 harmful prompts

**Contents:**
1. Safety steering methodology
2. Layer-by-layer effectiveness table
3. Visualization: steering effect by depth
4. Activation extraction & injection procedure
5. Limitations & future work (LLM-based scoring)

---

## 📄 Report 3: Attention Visualization (285 KB)

**File:** `task3_report.pdf`

**Summary:** Interpretability analysis via attention maps and token attribution.

**Key Results:**
- 704 heads classified: 50/50 instruction vs completion
- LoRA shifts attention in layers 12-18
- Generated 10 HTML heatmaps + 3 clustering plots

**Contents:**
1. Attention capture & rollout algorithm
2. Head classification results table
3. Visualizations: clustering dendrograms + base vs LoRA diff
4. Technical implementation (hooks, clustering, HTML generation)
5. Numerical warnings & debugging notes

---

## 📄 Report 4: Quantization & Edge Deployment (151 KB)

**File:** `task4_report.pdf`

**Summary:** Model compression via 4/6/8-bit quantization for edge devices.

**Key Results:**
- **Q4 optimal:** 3.5× compression, 2.1× faster, +3.5% PPL
- Batch scaling: throughput flat across batch sizes (memory-bound)
- FastAPI inference server packaged

**Contents:**
1. Quantization methodology (MLX framework)
2. Benchmark comparison table (4 models × 3 batch sizes)
3. Perplexity vs compression trade-offs
4. Technical implementation (fuse, quantize, benchmark)
5. GGUF conversion gap analysis (llama.cpp required)

---

## 📄 Report 5: Logit Lens & Layer Importance (207 KB)

**File:** `task5_report.pdf`

**Summary:** Layer-wise analysis to identify where instruction-following emerges.

**Key Results:**
- Minimum layers: **11/22** for 95% performance (layers 11-21)
- Bottom 5 layers have zero impact (redundancy discovered)
- Correlation: NaN (uniform LoRA importance—unexpected null result)

**Contents:**
1. Logit lens methodology & layer-wise accuracy
2. Ablation study results (6 fractions: 5%-100%)
3. Visualization: accuracy & importance by layer depth
4. Technical implementation (hidden state extraction, L2 norms)
5. Debugging notes for zero-importance anomaly

---

## 📊 Aggregate Statistics

| Metric | Value |
|--------|-------|
| Total experiments run | 5 tasks |
| Total runtime | ~4.5 hours |
| Models trained | 5 (rank ablation) |
| Activations extracted | 200 prompts × 12 layers |
| Attention heads analyzed | 704 (22 layers × 32 heads) |
| Quantization levels tested | 3 (Q4, Q6, Q8) |
| Layers ablated | 6 configurations (5%-100%) |
| Visualizations generated | 15+ (plots, heatmaps, tables) |

---

## 🔍 Quick Access

```bash
# View all reports
open task{1,2,3,4,5}_report.pdf

# View specific task
open task1_report.pdf  # LoRA ranks
open task2_report.pdf  # Activation steering
open task3_report.pdf  # Attention visualization
open task4_report.pdf  # Quantization
open task5_report.pdf  # Logit lens

# View LaTeX sources
ls docs/task*_report.tex

# Regenerate all PDFs
cd docs && for i in 1 2 3 4 5; do pdflatex task${i}_report.tex; done && cp task*_report.pdf ..
```

---

## 📚 Related Documentation

- **REPORT_SUMMARY.md** - Detailed summary of all results and contributions
- **docs/README_REPORTS.md** - Report compilation guide
- **README.md** - Main project documentation
- **.github/copilot-instructions.md** - Repository-level Copilot guidance
- **results/task*/** - Raw experimental outputs and artifacts

---

## 🎓 Citation

If you use these reports or results, please cite:

```bibtex
@techreport{tinyllama2026advanced,
  title={Advanced Analysis of LoRA Fine-tuning for TinyLlama on Apple Silicon},
  author={TinyLlama Instruction Tuning Project},
  year={2026},
  month={March},
  institution={MLX-LM Framework},
  note={5 comprehensive 2-page PDF reports}
}
```

---

**Last Updated:** March 19, 2026  
**Total Size:** 1.1 MB  
**Format:** PDF (LaTeX-generated)
