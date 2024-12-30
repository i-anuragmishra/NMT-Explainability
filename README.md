# Advancing Explainability in Neural Machine Translation

[![arXiv](https://img.shields.io/badge/arXiv-2412.XXXXX-b31b1b.svg)](http://arxiv.org/abs/2412.18669)

This repository contains the implementation code and experimental analysis for the paper "Advancing Explainability in Neural Machine Translation: Analytical Metrics for Attention and Alignment Consistency".

## Abstract

This research introduces a systematic framework to quantitatively evaluate the explainability of Neural Machine Translation (NMT) models' attention patterns by comparing them against statistical alignments and correlating them with standard machine translation quality metrics. We present novel metrics—attention entropy and alignment agreement—and validate them on an English-German test subset from WMT14 using a pre-trained mT5 model.

Key findings:
- Models with larger capacity produce more focused attention patterns
- Lower attention entropy correlates with higher alignment agreement
- Translation quality metrics (BLEU, METEOR) show positive correlation with interpretability measures
- T5-Large achieves BLEU: 19.6, METEOR: 0.346 with lowest average entropy of 0.273

## Implementation Details

### Model and Dataset Access

- **Pre-trained Model Download:**  
  ```bash
  pip install gdown
  gdown "https://drive.google.com/uc?id=19gXkvi4IP68M_d8jTFgEW8fHLsK_jr2w" -O model.zip
  unzip model.zip
  ```

### Project Structure
```
├── src/
│   ├── train.py           # Training script (30+ days runtime)
│   ├── test.py           # Full evaluation script
│   ├── demo.py           # Quick demonstration script
│   ├── attention/        # Attention analysis tools
│   │   ├── entropy.py   
│   │   └── alignment.py 
│   └── evaluation/      # Evaluation metrics
├── fastalign/           # FastAlign tool for alignments
├── experiments/        # Experimental notebooks
└── results/           # Analysis results and figures
```

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the pre-trained model using the command provided above.

3. Ensure the `fastalign` folder is present in the project directory.

## Execution Options

1. **Demo Script (Recommended):**
```bash
python demo.py
```
- Provides quick visualization of attention patterns
- Calculates BLEU and METEOR scores
- Generates attention heatmaps

2. **Full Evaluation:**
```bash
python test.py
```
- Complete evaluation on large test set
- Comprehensive metric calculation

3. **Training (Not Recommended):**
```bash
python train.py
```
- Full model training (~30 days runtime)
- Requires significant computational resources

## Research Components

1. **Attention Analysis Tools**
   - Attention entropy calculation
   - Alignment agreement metrics
   - Statistical alignment comparison

2. **Evaluation Framework**
   - Multiple T5 model variants comparison
   - Translation quality assessment
   - Attention pattern analysis

3. **Visualization Suite**
   - Attention heatmaps
   - Correlation plots
   - Normalized attention matrices

## Important Notes

- This project requires Research Computing (RC) services
- Not compatible with Narnia due to resource limitations
- Model size exceeds 2GB (download separately)
- Datasets are downloaded automatically during execution

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{mishra2024advancing,
  title={Advancing Explainability in Neural Machine Translation: Analytical Metrics for Attention and Alignment Consistency},
  author={Mishra, Anurag},
  journal={arXiv preprint arXiv:2412.XXXXX},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Anurag Mishra  
School of Information  
Rochester Institute of Technology  
am2552@rit.edu
