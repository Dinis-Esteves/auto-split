# Invoice Reader Tests

This directory contains comprehensive unit tests for the invoice reader functionality.

## Test Files

- `test_invoice_reader.py` - Comprehensive unit tests with mocking
- `test_parsing.py` - Focused tests for text parsing functionality  
- `test_data.py` - Sample test data and expected results
- `simple_test.py` - Simple test runner that can be executed directly

## Running Tests

### Method 1: Using the simple test runner
```bash
cd /home/denis/.Documents/Desktop/auto_split
python tests/simple_test.py
```

### Method 2: Using unittest
```bash
cd /home/denis/.Documents/Desktop/auto_split
python -m unittest discover tests -v
```

### Method 3: Using pytest (after installing)
```bash
cd /home/denis/.Documents/Desktop/auto_split
pip install pytest
pytest tests/ -v
```

### Method 4: Using the main test runner
```bash
cd /home/denis/.Documents/Desktop/auto_split
python run_tests.py
```

## Test Coverage

The tests cover:

1. **Text Parsing Tests**
   - Basic invoice format parsing
   - IVA tax code removal
   - Unit preservation (ml, L, kg, etc.)
   - Grocery receipt format handling
   - OCR correction functionality
   - Edge cases and error handling

2. **Image Processing Tests**
   - Simple preprocessing function
   - Advanced preprocessing with OpenCV
   - Grayscale conversion
   - Error handling for missing dependencies

3. **Integration Tests**
   - Complete pipeline from image to parsed items
   - OCR mocking for consistent testing
   - File handling and cleanup

4. **Utility Tests**
   - Regex pattern validation
   - Decimal format handling
   - Quantity extraction patterns

## Sample Test Data

The tests use realistic sample data:
- Restaurant receipts with IVA codes
- Grocery store receipts with price-per-kg formats
- Bar/café receipts with units (ml, L)

## Expected Behavior

Tests verify that:
- IVA codes (I3B, 23X, etc.) are removed from item names
- Price-per-unit lines (EUR/kg) are not treated as separate items
- Units in measurements are preserved (5L, 330ml, etc.)
- Quantities and prices are correctly extracted
- Irrelevant lines (totals, dates, etc.) are ignored

## Troubleshooting

If tests fail due to import errors:
1. Ensure you're running from the project root directory
2. Check that the `src/ocr-reader/` directory contains `read_invoice.py`
3. Try using the `simple_test.py` which has better path handling

If OpenCV/scipy tests fail:
- Tests will automatically fall back to PIL-only processing
- Install missing dependencies: `pip install opencv-python scipy`
