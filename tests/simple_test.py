#!/usr/bin/env python3
##################
# SIMPLE TEST RUNNER #
##################

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add paths for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src', 'ocr-reader'))

def test_parse_function():
    """Test the parse_invoice_text function directly."""
    
    try:
        from read_invoice import parse_invoice_text
        print("✓ Successfully imported parse_invoice_text")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False
    
    # Test 1: Basic parsing
    test_text = """
    1 Cerveja 2.50
    2 Bifana 8.00
    1 Agua 1.50
    Total: 12.00
    """
    
    items = parse_invoice_text(test_text)
    print(f"Test 1 - Basic parsing: Found {len(items)} items")
    
    expected_items = 3
    if len(items) != expected_items:
        print(f"✗ Expected {expected_items} items, got {len(items)}")
        return False
    else:
        print("✓ Correct number of items found")
    
    # Test 2: IVA code removal
    test_text_with_iva = """
    1 T-Bone I3B 13.00
    2 Imperial 23X 2.20
    1 Água I9X 1.30
    """
    
    items = parse_invoice_text(test_text_with_iva)
    print(f"Test 2 - IVA removal: Found {len(items)} items")
    
    # Check that IVA codes are removed
    for item in items:
        if any(code in item['item'] for code in ['I3B', '23X', 'I9X', 'SL']):
            print(f"✗ IVA code found in item: {item['item']}")
            return False
    
    print("✓ IVA codes properly removed")
    
    # Test 3: Grocery format (should skip EUR/kg lines)
    grocery_text = """
    TOMATE DE CACHO , B
    540 x 1.49 EUR/kg 0.80
    BATATA CONS. ROXA BX , B
    37 2.37
    """
    
    items = parse_invoice_text(grocery_text)
    print(f"Test 3 - Grocery format: Found {len(items)} items")
    
    # Should not include EUR/kg lines as items
    for item in items:
        if 'EUR/kg' in item['item']:
            print(f"✗ Found EUR/kg in item: {item['item']}")
            return False
    
    print("✓ EUR/kg lines properly filtered")
    
    # Print sample results
    print("\nSample parsed items:")
    for i, item in enumerate(items[:3], 1):
        print(f"  {i}. {item['item']} (Qty: {item['quantidade']}, Price: €{item['preco_total']})")
    
    return True

def test_image_processing():
    """Test image processing functions."""
    
    try:
        from read_invoice import preprocess_image_simple
        from PIL import Image
        print("✓ Successfully imported image processing functions")
    except ImportError as e:
        print(f"✗ Failed to import image functions: {e}")
        return False
    
    # Create a test image
    test_image = Image.new('RGB', (100, 100), color='white')
    
    # Test preprocessing
    processed = preprocess_image_simple(test_image)
    
    if processed.mode != 'L':
        print(f"✗ Expected grayscale image, got {processed.mode}")
        return False
    
    print("✓ Image preprocessing works correctly")
    return True

def run_all_tests():
    """Run all tests."""
    print("Running Invoice Reader Tests")
    print("=" * 40)
    
    tests = [
        ("Parse Function Tests", test_parse_function),
        ("Image Processing Tests", test_image_processing),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        
        try:
            result = test_func()
            results.append(result)
            status = "PASSED" if result else "FAILED"
            print(f"Status: {status}")
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
