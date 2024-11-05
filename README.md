# Hampel Filter

This project provides an efficient implementation of the Hampel filter for outlier detection, leveraging Numba’s JIT compilation to improve performance. It supports both serial and parallel computation modes for median and median absolute deviation (MAD) calculations, making it ideal for processing large datasets.

## Installation

Make sure to have `Numba`, `NumPy`, and `Pandas` installed:

```bash
pip install numba numpy pandas
```

and then you can install the package

```bash
pip install hampel_filter
```

## Usage

This package contains functions for outlier detection in time-series or sequence data using the Hampel filter. The main function `hampel()` identifies outliers by calculating the median and median absolute deviation (MAD) within a specified window size and compares it to a threshold to detect outliers.

### Function: `hampel`

```python
def hampel(arr, window_size=5, n=3, parallel=False, return_indices=True)
```

#### Parameters:
- `arr` (`np.ndarray`, `pd.Series`, or `pd.DataFrame`): The input data array for outlier detection.
- `window_size` (`int`, default=5): The half-size of the moving window for median calculation.
- `n` (`int`, default=3): The threshold factor; outliers are values beyond `n` times the MAD.
- `parallel` (`bool`, default=False): Whether to use parallel computation. When `True`, it leverages multi-core processing.
- `return_indices` (`bool`, default=True): If `True`, returns the indices of outliers; otherwise, returns a boolean array indicating outliers.

#### Returns:
- If `return_indices` is `True`, returns a tuple of arrays with outlier indices.
- If `return_indices` is `False`, returns a boolean array where `True` indicates outliers.

### Example

```python
import numpy as np
from hampel_filter import hampel  # Assuming the script is named hampel_filter.py

# Sample data with outliers
data = np.array([1, 1, 2, 2, 100, 2, 2, 1, 1])

# Detect outliers
outlier_indices = hampel(data, window_size=2, n=3, parallel=True)

print("Outlier indices:", outlier_indices)
```

### Detailed Functions

The implementation includes the following helper functions:

- `calc_medians(window_size, arr, medians)`: Calculates the moving median over a specified window.
- `calc_medians_std(window_size, arr, medians, medians_diff)`: Computes the median absolute deviation (MAD) for outlier detection.
- `calc_medians_parallel` and `calc_medians_std_parallel`: Parallel versions of the above functions using `prange` for improved performance.

## Performance Notes

By setting `parallel=True`, the Hampel filter calculation leverages Numba's parallel processing capabilities, significantly speeding up the computations on large arrays. However, performance gains depend on your system's CPU cores and load.

## License

This project is licensed under the MIT License.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
