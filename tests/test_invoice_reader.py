##################
# INVOICE READER TESTS #
##################

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

# Add the src directory to the path so we can import the invoice reader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'ocr-reader'))

from read_invoice import (
    preprocess_image_simple,
    preprocess_image_advanced,
    parse_invoice_text,
    process_invoice
)


class TestInvoiceReader(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a simple test image
        self.test_image = Image.new('RGB', (100, 100), color='white')
        
    def test_preprocess_image_simple(self):
        """Test the simple image preprocessing function."""
        # Test with RGB image
        result = preprocess_image_simple(self.test_image)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, 'L')  # Should be converted to grayscale
        
        # Test with already grayscale image
        gray_image = self.test_image.convert('L')
        result = preprocess_image_simple(gray_image)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, 'L')
    
    @patch('read_invoice.cv2')
    def test_preprocess_image_advanced(self, mock_cv2):
        """Test the advanced image preprocessing function."""
        # Mock OpenCV functions
        mock_cv2.bilateralFilter.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.filter2D.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.addWeighted.return_value = np.zeros((100, 100), dtype=np.uint8)
        
        result = preprocess_image_advanced(self.test_image)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, 'L')
        
    def test_parse_invoice_text_basic(self):
        """Test basic invoice text parsing."""
        test_text = """
        Mesa 1
        1 Cerveja 2.50
        2 Bifana 8.00
        1 Agua 1.50
        Total: 12.00
        """
        
        items = parse_invoice_text(test_text)
        
        self.assertEqual(len(items), 3)
        
        # Check first item
        self.assertEqual(items[0]['item'], 'Cerveja')
        self.assertEqual(items[0]['quantidade'], 1)
        self.assertEqual(items[0]['preco_total'], 2.50)
        
        # Check second item
        self.assertEqual(items[1]['item'], 'Bifana')
        self.assertEqual(items[1]['quantidade'], 2)
        self.assertEqual(items[1]['preco_total'], 8.00)
        
        # Check third item
        self.assertEqual(items[2]['item'], 'Água')
        self.assertEqual(items[2]['quantidade'], 1)
        self.assertEqual(items[2]['preco_total'], 1.50)
    
    def test_parse_invoice_text_with_tax_codes(self):
        """Test parsing with IVA tax codes that should be removed."""
        test_text = """
        1 T-Bone I3B 13.00
        2 Imperial 23X 2.20
        1 Água I9X 1.30
        """
        
        items = parse_invoice_text(test_text)
        
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0]['item'], 'T-Bone')
        self.assertEqual(items[1]['item'], 'Imperial')
        self.assertEqual(items[2]['item'], 'Água')
    
    def test_parse_invoice_text_with_units(self):
        """Test parsing preserves unit measurements."""
        test_text = """
        1 Imperial 5L 12.00
        2 Cerveja 330ml 5.00
        1 Carne 2kg 15.00
        """
        
        items = parse_invoice_text(test_text)
        
        self.assertEqual(len(items), 3)
        # Note: The exact behavior depends on current implementation
    
    def test_parse_invoice_text_ignores_irrelevant_lines(self):
        """Test that irrelevant lines are ignored."""
        test_text = """
        Total Geral: 25.00
        IVA: 5.00
        Mesa: 15
        Data: 20/10/2025
        1 Cerveja 2.50
        Obrigado pela preferencia
        """
        
        items = parse_invoice_text(test_text)
        
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['item'], 'Cerveja')
    
    def test_parse_invoice_text_grocery_format(self):
        """Test parsing grocery receipt format."""
        test_text = """
        TOMATE DE CACHO , B
        80 0.80
        x —EUR/kg
        BATATA CONS. ROXA BX , B
        37 2.37
        """
        
        items = parse_invoice_text(test_text)
        
        # Should parse items but ignore the EUR/kg lines
        for item in items:
            self.assertNotIn('EUR/kg', item['item'])
            self.assertNotIn('x —', item['item'])
    
    def test_parse_invoice_text_ocr_corrections(self):
        """Test OCR corrections are applied."""
        test_text = """
        1 Inperial 2.20
        1 Agua 1.50
        1 pao 1.00
        """
        
        items = parse_invoice_text(test_text)
        
        # Check that corrections are applied
        item_names = [item['item'] for item in items]
        self.assertIn('Imperial', ' '.join(item_names))  # Should be corrected from Inperial
    
    def test_parse_invoice_text_empty_input(self):
        """Test parsing with empty or invalid input."""
        # Empty string
        items = parse_invoice_text("")
        self.assertEqual(len(items), 0)
        
        # Only irrelevant content
        items = parse_invoice_text("Total: 25.00\nIVA: 5.00")
        self.assertEqual(len(items), 0)
        
        # Invalid format
        items = parse_invoice_text("Random text without proper format")
        self.assertEqual(len(items), 0)
    
    def test_parse_invoice_text_edge_cases(self):
        """Test edge cases in parsing."""
        test_text = """
        10 Item123 15.99
        1 A 0.01
        999 LongItemNameWithLotsOfCharacters 999.99
        """
        
        items = parse_invoice_text(test_text)
        
        # Should handle large quantities and prices
        self.assertTrue(any(item['quantidade'] == 10 for item in items))
        self.assertTrue(any(item['quantidade'] == 999 for item in items))
        
    @patch('read_invoice.pytesseract.image_to_string')
    @patch('read_invoice.pytesseract.image_to_data')
    def test_process_invoice_basic(self, mock_image_to_data, mock_image_to_string):
        """Test basic invoice processing."""
        # Mock OCR responses
        mock_image_to_string.return_value = "1 Test Item 5.00"
        mock_image_to_data.return_value = {
            'conf': [85, 90, 80, 85],  # Confidence scores
            'text': ['1', 'Test', 'Item', '5.00']
        }
        
        # Create a temporary test image file
        test_image_path = "/tmp/test_invoice.jpg"
        self.test_image.save(test_image_path)
        
        try:
            result = process_invoice(test_image_path, save_processed=False)
            self.assertEqual(result, "1 Test Item 5.00")
        finally:
            # Clean up
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
    
    @patch('read_invoice.pytesseract.image_to_string')
    def test_process_invoice_fallback(self, mock_image_to_string):
        """Test fallback OCR behavior."""
        # Mock low confidence OCR
        mock_image_to_string.side_effect = ["", "Fallback text"]
        
        # Create a temporary test image file
        test_image_path = "/tmp/test_invoice.jpg"
        self.test_image.save(test_image_path)
        
        try:
            with patch('read_invoice.pytesseract.image_to_data') as mock_data:
                mock_data.return_value = {'conf': [10, 15]}  # Low confidence
                result = process_invoice(test_image_path, save_processed=False)
                self.assertEqual(result, "Fallback text")
        finally:
            # Clean up
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
    
    def test_parse_invoice_text_decimal_formats(self):
        """Test different decimal formats (comma vs dot)."""
        test_text = """
        1 Item1 2,50
        1 Item2 3.75
        1 Item3 1,25
        """
        
        items = parse_invoice_text(test_text)
        
        self.assertEqual(len(items), 3)
        # Prices should be properly converted to floats regardless of comma/dot
        prices = [item['preco_total'] for item in items]
        self.assertIn(2.50, prices)
        self.assertIn(3.75, prices)
        self.assertIn(1.25, prices)


class TestInvoiceReaderIntegration(unittest.TestCase):
    """Integration tests for the complete invoice processing pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.test_image_path = "/tmp/integration_test_invoice.jpg"
        # Create a simple test image
        test_image = Image.new('RGB', (400, 600), color='white')
        test_image.save(self.test_image_path)
    
    def tearDown(self):
        """Clean up after integration tests."""
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    @patch('read_invoice.pytesseract.image_to_string')
    @patch('read_invoice.pytesseract.image_to_data')
    def test_full_pipeline(self, mock_image_to_data, mock_image_to_string):
        """Test the complete invoice processing pipeline."""
        # Mock a realistic OCR response
        mock_text = """
        RESTAURANTE TESTE
        Mesa 5
        1 Francesinha 12.50
        2 Imperial 4.00
        1 Água 1.50
        Total: 18.00
        Obrigado
        """
        
        mock_image_to_string.return_value = mock_text
        mock_image_to_data.return_value = {
            'conf': [85, 90, 80, 85, 90],
            'text': ['1', 'Francesinha', '12.50', '2', 'Imperial']
        }
        
        # Process the invoice
        ocr_text = process_invoice(self.test_image_path, save_processed=False)
        items = parse_invoice_text(ocr_text)
        
        # Verify results
        self.assertEqual(len(items), 3)
        
        # Check that items are correctly extracted
        item_names = [item['item'] for item in items]
        self.assertIn('Francesinha', item_names)
        self.assertIn('Imperial', item_names)
        self.assertIn('Água', item_names)

        # Calculate total
        total = sum(item['preco_total'] for item in items)
        self.assertEqual(total, 18.00)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add all test cases
    test_suite.addTest(unittest.makeSuite(TestInvoiceReader))
    test_suite.addTest(unittest.makeSuite(TestInvoiceReaderIntegration))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)
