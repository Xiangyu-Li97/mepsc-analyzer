# Contributing to mEPSC Analyzer

Thank you for your interest in contributing to mEPSC Analyzer! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear description of the problem
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your system information (OS, Python/MATLAB version)
- Sample data if possible (or a description of the data characteristics)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue describing:
- The enhancement you'd like to see
- Why it would be useful
- How it might work

### Code Contributions

1. **Fork the repository** and create a new branch for your feature
2. **Write clear, commented code** following the existing style
3. **Test your changes** with various data files
4. **Update documentation** if you're adding new features
5. **Submit a pull request** with a clear description of your changes

## Development Setup

### Python Development

```bash
# Clone the repository
git clone https://github.com/yourusername/mepsc-analyzer.git
cd mepsc-analyzer

# Install dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/
```

### MATLAB Development

1. Add the repository folder to your MATLAB path
2. Ensure you have the required toolboxes installed
3. Test your changes with sample data

## Code Style

### Python
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

### MATLAB
- Use clear, descriptive variable names
- Add comments for complex logic
- Include help text for all functions
- Follow MATLAB best practices

## Testing

Before submitting a pull request:
- Test your code with multiple ABF files
- Verify that existing functionality still works
- Check that output files are generated correctly
- Ensure visualizations display properly

## Documentation

When adding new features:
- Update the README.md with usage examples
- Add entries to QUICKSTART.md if relevant
- Document all parameters and return values
- Include examples in docstrings

## Questions?

If you have questions about contributing, feel free to:
- Open an issue for discussion
- Contact the maintainer: xiangyuli997@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
