# Data

The repository intentionally does not redistribute dermatology images.

For Fitzpatrick17k, run:

```bash
python scripts/prepare_fitzpatrick.py
```

The script downloads the official metadata and attempts to retrieve images from the source URLs listed by the dataset. Some original URLs are known to be unavailable; failed files are recorded in `failed_downloads.txt` and excluded from the standardized metadata.

The original dataset is subject to its own terms/licensing. Review those terms before use or redistribution.
