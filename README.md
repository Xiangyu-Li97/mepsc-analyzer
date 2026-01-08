# mEPSC/mIPSC Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2020a+-orange.svg)](https://www.mathworks.com/products/matlab.html)

A powerful, automated tool for detecting and analyzing miniature excitatory/inhibitory postsynaptic currents (mEPSC/mIPSC) from electrophysiology recordings. This tool provides a free, open-source alternative to commercial software like Mini Analysis.

[中文文档](docs/README_CN.md) | [English Documentation](docs/QUICKSTART.md)

## ✨ Features

- 🔍 **Automatic Event Detection**: Intelligent threshold-based detection of synaptic events
- 📊 **Comprehensive Feature Extraction**: Amplitude, rise time, decay tau, area, and more
- 📈 **Rich Visualizations**: Detailed analysis reports with multiple plots
- 🚀 **Batch Processing**: Analyze hundreds of files automatically
- 🐍 **Python + MATLAB**: Choose your preferred platform
- 📝 **CSV Export**: Results in universal format for further analysis
- ⚙️ **Highly Customizable**: Adjust parameters to fit your data

## 🚀 Quick Start

### Python Version

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python mepsc_analyzer.py your_data.abf
```

### MATLAB Version

```matlab
% Make sure you have abfload installed
results = mepsc_analyzer('your_data.abf');
```

That's it! The tool will automatically detect events, extract features, and generate a comprehensive analysis report.

## 📊 Example Output

From a 50-second recording, the tool detected:
- **172 events** (frequency: 3.44 Hz)
- **Average amplitude**: 29.9 ± 13.1 pA
- **Average rise time**: 0.82 ms
- **Average decay τ**: 1.8 ms

## 📖 Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get started in 5 minutes
- **[Detailed Documentation (CN)](docs/README_CN.md)** - Complete user manual in Chinese
- **[Algorithm Explanation](docs/analysis_approach_summary.md)** - How the tool works

## 🎯 Use Cases

### Standard Analysis
```bash
python mepsc_analyzer.py cell01.abf
```

### Batch Processing
See [examples/batch_analysis.py](examples/batch_analysis.py) for Python or [examples/batch_analysis.m](examples/batch_analysis.m) for MATLAB.

### Custom Parameters
```bash
python mepsc_analyzer.py data.abf \
    --threshold 4.0 \
    --min-interval 5 \
    --min-amp 10 \
    --max-amp 100
```

## 🔧 Parameter Tuning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 3.5 | Detection threshold (× noise std) |
| `min-interval` | 5 ms | Minimum event interval |
| `min-amp` | None | Minimum amplitude filter (pA) |
| `max-amp` | None | Maximum amplitude filter (pA) |

**Tip**: If you're missing small events, lower the threshold to 2.5-3.0. If you're detecting too much noise, increase it to 4.0-5.0.

## 📦 Installation

### Python Requirements
```bash
pip install -r requirements.txt
```

Required packages:
- pyabf >= 2.3.0
- numpy >= 1.20.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- scipy >= 1.7.0

### MATLAB Requirements
- MATLAB R2020a or later
- Curve Fitting Toolbox
- Statistics and Machine Learning Toolbox
- [abfload function](https://www.mathworks.com/matlabcentral/fileexchange/6190-abfload)

## 🆚 Comparison with Mini Analysis

| Feature | Mini Analysis | mEPSC Analyzer |
|---------|---------------|----------------|
| Price | Commercial | Free & Open Source |
| Automation | Manual adjustment | Fully automatic |
| Batch Processing | ❌ | ✅ |
| Customization | Limited | Fully customizable |
| Output Format | Proprietary | CSV (universal) |
| Platform | Windows only | Cross-platform |

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Xiangyu Li**
- Email: xiangyuli997@gmail.com
- GitHub: https://github.com/yourusername/mepsc-analyzer

## 🙏 Acknowledgments

This tool was inspired by:
- Mini Analysis (Synaptosoft)
- Clampfit (Molecular Devices)
- NeuroMatic (Jason Rothman)

## 📚 Citation

If you use this tool in your research, please cite:

```bibtex
@software{mepsc_analyzer,
  author = {Li, Xiangyu},
  title = {mEPSC Analyzer: Automated Analysis Tool for Miniature Postsynaptic Currents},
  year = {2026},
  url = {https://github.com/yourusername/mepsc-analyzer}
}
```

## 🌟 Star History

If you find this tool useful, please consider giving it a star! ⭐

---

**Happy analyzing!** 🔬
