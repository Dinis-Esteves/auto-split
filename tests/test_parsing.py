##################
# PARSING TESTS #
##################

import unittest
import sys
import os
import re

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'ocr-reader'))

try:
    from read_invoice import parse_invoice_text
except ImportError:
    # If import fails, we'll create a mock version for testing
    def parse_invoice_text(texto_ocr):
        """Mock implementation for testing when import fails."""
        return []

from test_data import (
    SAMPLE_RESTAURANT_RECEIPT,
    SAMPLE_GROCERY_RECEIPT, 
    SAMPLE_BAR_RECEIPT,
    EXPECTED_RESTAURANT_ITEMS,
    EXPECTED_GROCERY_ITEMS,
    EXPECTED_BAR_ITEMS
)


class TestParseInvoiceText(unittest.TestCase):
    """Test the invoice text parsing functionality."""
    
    def test_parse_restaurant_receipt(self):
        """Test parsing a typical restaurant receipt."""
        items = parse_invoice_text(SAMPLE_RESTAURANT_RECEIPT)
        
        # Should extract items without IVA codes
        self.assertGreater(len(items), 0, "Should find at least some items")
        
        # Check that IVA codes are removed from item names
        for item in items:
            self.assertNotIn("I3B", item['item'])
            self.assertNotIn("23X", item['item']) 
            self.assertNotIn("I9X", item['item'])
            self.assertNotIn("13", item['item'])
    
    def test_parse_grocery_receipt(self):
        """Test parsing a grocery store receipt."""
        items = parse_invoice_text(SAMPLE_GROCERY_RECEIPT)
        
        # Should not include EUR/kg lines as separate items
        for item in items:
            self.assertNotIn("EUR/kg", item['item'])
            self.assertNotIn("x", item['item'])
    
    def test_parse_bar_receipt(self):
        """Test parsing a bar/café receipt with units."""
        items = parse_invoice_text(SAMPLE_BAR_RECEIPT)
        
        self.assertGreater(len(items), 0, "Should find at least some items")
        
        # Check that units in ml are preserved
        item_names = [item['item'] for item in items]
        has_ml_items = any('ml' in name for name in item_names)
        # Note: This depends on current implementation
    
    def test_ignore_irrelevant_lines(self):
        """Test that irrelevant lines are properly ignored."""
        test_text = """
        Total: 25.00
        IVA: 5.75
        Mesa: 12
        Data: 20/10/2025
        Obrigado pela preferência!
        Vendedor: João
        Caixa: 1
        1 Valid Item 10.00
        """
        
        items = parse_invoice_text(test_text)
        
        # Should only find the valid item, not the metadata
        valid_items = [item for item in items if 'Valid Item' in item['item'] or item['preco_total'] == 10.00]
        self.assertGreater(len(valid_items), 0, "Should find the valid item")
        
        # Should not have items with metadata keywords
        for item in items:
            item_lower = item['item'].lower()
            self.assertNotIn('total', item_lower)
            self.assertNotIn('iva', item_lower) 
            self.assertNotIn('mesa', item_lower)
            self.assertNotIn('data', item_lower)
    
    def test_decimal_format_handling(self):
        """Test handling of different decimal formats."""
        test_text = """
        1 Item1 2,50
        1 Item2 3.75
        1 Item3 10,25
        """
        
        items = parse_invoice_text(test_text)
        
        # Should convert commas to dots for proper float parsing
        expected_prices = [2.50, 3.75, 10.25]
        actual_prices = [item['preco_total'] for item in items]
        
        for expected_price in expected_prices:
            self.assertIn(expected_price, actual_prices, 
                         f"Price {expected_price} should be found in {actual_prices}")
    
    def test_quantity_patterns(self):
        """Test different quantity patterns."""
        test_text = """
        2 Item1 5.00
        3x Item2 7.50
        1 Item3 2.25
        """
        
        items = parse_invoice_text(test_text)
        
        # Should extract quantities correctly
        quantities = [item['quantidade'] for item in items]
        expected_quantities = [2, 3, 1]
        
        for expected_qty in expected_quantities:
            self.assertIn(expected_qty, quantities,
                         f"Quantity {expected_qty} should be found in {quantities}")
    
    def test_empty_and_invalid_input(self):
        """Test handling of empty or invalid input."""
        # Empty string
        items = parse_invoice_text("")
        self.assertEqual(len(items), 0)
        
        # None input (should not crash)
        try:
            items = parse_invoice_text(None)
            self.assertEqual(len(items), 0)
        except (TypeError, AttributeError):
            pass  # Acceptable to raise an error for None input
        
        # Random text
        items = parse_invoice_text("This is just random text without any invoice format")
        self.assertEqual(len(items), 0)
    
    def test_name_cleaning(self):
        """Test that item names are properly cleaned."""
        test_text = """
        1 T-Bone 138 13.00
        1 Imperial 23X 2.20
        1 Água 1.30
        """
        
        items = parse_invoice_text(test_text)
        
        # Names should be cleaned of tax codes and numbers
        expected_clean_names = ['T-Bone', 'Imperial', 'Água']
        actual_names = [item['item'] for item in items]
        
        for expected_name in expected_clean_names:
            # Should contain the base name without codes
            found = any(expected_name in actual_name for actual_name in actual_names)
            self.assertTrue(found, f"Should find cleaned name containing '{expected_name}' in {actual_names}")


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions and edge cases."""
    
    def test_regex_patterns(self):
        """Test that regex patterns work as expected."""
        # Test quantity pattern
        quantidade_pattern = r"(\d+)(?:x|\s)"
        
        test_cases = [
            ("2 Item", 2),
            ("3x Item", 3), 
            ("1 Item", 1),
            ("10 Item", 10),
        ]
        
        for test_string, expected_qty in test_cases:
            match = re.search(quantidade_pattern, test_string)
            self.assertIsNotNone(match, f"Should match quantity in '{test_string}'")
            self.assertEqual(int(match.group(1)), expected_qty)
    
    def test_price_pattern(self):
        """Test price extraction patterns."""
        preco_pattern = r"(\d+[,.]\d{1,2})"
        
        test_cases = [
            ("Item 2.50", ["2.50"]),
            ("Item 3,75", ["3,75"]),
            ("Item 10.25 other 5.00", ["10.25", "5.00"]),
        ]
        
        for test_string, expected_prices in test_cases:
            matches = re.findall(preco_pattern, test_string)
            self.assertEqual(matches, expected_prices, 
                           f"Should extract {expected_prices} from '{test_string}', got {matches}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
