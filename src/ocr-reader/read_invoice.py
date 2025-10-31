##################
# INVOICE READER #
##################

import sys
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import os
import re
import numpy as np
from scipy import ndimage
import cv2

'''
Simple preprocessing function that works without external dependencies.
Args:
    img (PIL.Image): Input image.
Returns:
    PIL.Image: Preprocessed image.
'''
def preprocess_image_simple(img):
    # Convert to grayscale
    if img.mode != 'L':
        img = img.convert('L')
    
    # Light noise reduction
    img = img.filter(ImageFilter.MedianFilter(1))
    
    # Gentle sharpening
    sharpness_enhancer = ImageEnhance.Sharpness(img)
    img = sharpness_enhancer.enhance(1.2)
    
    # Light contrast enhancement
    contrast_enhancer = ImageEnhance.Contrast(img)
    img = contrast_enhancer.enhance(1.1)
    
    return img

'''
Applies gentle deblurring using OpenCV.
Args:
    img (PIL.Image): Input image.
Returns:
    PIL.Image: Deblurred image.
'''
def deblur_image(img):
    # Convert PIL to OpenCV format
    img_cv = np.array(img)
    
    # Apply gentle bilateral filter
    filtered = cv2.bilateralFilter(img_cv, 30, 50, 50)  # Gentler settings
    
    # Apply very mild sharpening kernel
    kernel = np.array([[0,-0.5,0], [-0.5,3,-0.5], [0,-0.5,0]])  # Much gentler kernel
    sharpened = cv2.filter2D(filtered, -1, kernel)
    
    # Blend original and sharpened (50/50 mix)
    result = cv2.addWeighted(img_cv, 0.5, sharpened, 0.5, 0)
    
    # Convert back to PIL
    return Image.fromarray(result)

'''
Preprocesses an image to improve OCR accuracy, especially for blurry images.
Args:
    img (PIL.Image): Input image to preprocess.
Returns:
    PIL.Image: Preprocessed image optimized for OCR.
'''
def preprocess_image_advanced(img):
    # Convert to grayscale if not already
    if img.mode != 'L':
        img = img.convert('L')
    
    # Step 1: Gentle noise reduction
    img = img.filter(ImageFilter.MedianFilter(1))  # Smaller filter size
    
    # Step 2: Very gentle sharpening for blurry text
    try:
        # Try OpenCV deblurring but with gentler settings
        img = deblur_image(img)
    except:
        # Fallback to gentle PIL sharpening
        sharpness_enhancer = ImageEnhance.Sharpness(img)
        img = sharpness_enhancer.enhance(1.3)  # Much less aggressive
    
    # Step 3: Light contrast enhancement
    contrast_enhancer = ImageEnhance.Contrast(img)
    img = contrast_enhancer.enhance(1.2)  # Reduced from 1.4
    
    # Step 4: Auto-contrast but with larger cutoff to be gentler
    img = ImageOps.autocontrast(img, cutoff=5)
    
    return img

'''
Reads and processes an invoice image to extract text using OCR.
Args:
    image_path (str): Path to the invoice image file.
    lang (str): Language code for OCR (default is Portuguese "por").
    save_processed (bool): Whether to save the processed image (default is False).
Returns:
    str: Extracted text from the invoice image.
'''
def process_invoice(image_path, lang="por", save_processed=False):
    img = Image.open(image_path)
    
    # Try advanced preprocessing, fallback to simple if it fails
    try:
        img_processed = preprocess_image_advanced(img)
    except Exception as e:
        print(f"Advanced preprocessing failed: {e}")
        print("Using simple preprocessing instead...")
        img_processed = preprocess_image_simple(img)

    # Optionally save processed image
    if save_processed:
        processed_path = os.path.splitext(image_path)[0] + "_processed.jpg"
        img_processed.save(processed_path)

    # Try multiple OCR configurations optimized for different text types
    configs = [
        r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzáàãâéêíóôõúç0123456789.,€$ ',
        r'--oem 3 --psm 4 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzáàãâéêíóôõúç0123456789.,€$ ',
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzáàãâéêíóôõúç0123456789.,€$ ',
        r'--oem 1 --psm 6',  # Try legacy OCR engine for blurry text
        r'--oem 2 --psm 6',  # Try cube OCR engine
    ]
    
    best_text = ""
    max_confidence = 0
    
    # Try different image scales but fewer and more conservative
    scales = [1.0, 1.5, 2.0]  # Removed 0.75 as it might make blur worse
    
    for scale in scales:
        # Resize image if needed
        if scale != 1.0:
            new_size = (int(img_processed.width * scale), int(img_processed.height * scale))
            scaled_img = img_processed.resize(new_size, Image.LANCZOS)
        else:
            scaled_img = img_processed
            
        for config in configs:
            try:
                # Get OCR data with confidence scores
                data = pytesseract.image_to_data(scaled_img, lang=lang, config=config, output_type=pytesseract.Output.DICT)
                
                # Calculate average confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    if avg_confidence > max_confidence:
                        max_confidence = avg_confidence
                        best_text = pytesseract.image_to_string(scaled_img, lang=lang, config=config)
            except:
                continue
    
    # Fallback to simple OCR if no good result
    if not best_text.strip() or max_confidence < 30:
        print("Trying fallback OCR methods...")
        # Try with original unprocessed image
        fallback_text = pytesseract.image_to_string(img, lang=lang)
        if len(fallback_text.strip()) > len(best_text.strip()):
            best_text = fallback_text
        
        # If still poor results, try with minimal preprocessing
        if len(best_text.strip()) < 10:
            # Just grayscale and light contrast
            simple_img = img.convert('L')
            enhancer = ImageEnhance.Contrast(simple_img)
            simple_img = enhancer.enhance(1.1)
            simple_text = pytesseract.image_to_string(simple_img, lang=lang)
            if len(simple_text.strip()) > len(best_text.strip()):
                best_text = simple_text

    print(f"Best OCR confidence: {max_confidence:.1f}%")
    return best_text


'''
Parses the extracted OCR text to identify invoice items.
Args:
    texto_ocr (str): Extracted text from the invoice image.
Returns:
    list: List of dictionaries with item details (name, quantity, total price).
'''
def parse_invoice_text(texto_ocr):
    itens = []
    linhas = texto_ocr.split("\n")

    for linha in linhas:
        linha = linha.strip()

        # Ignora linhas irrelevantes (mais completo)
        irrelevant_keywords = [
            "total", "iva", "mesa", "fatura", "consumidor", "nif", "preco", "descricao",
            "subtotal", "troco", "desconto", "taxa", "servico", "data", "hora",
            "vendedor", "caixa", "recibo", "talao", "obrigado", "volte", "sempre", 
        ]
        
        if not linha or len(linha) < 3 or any(p in linha.lower() for p in irrelevant_keywords):
            continue

        # Skip lines that are clearly price-per-unit indicators
        if any(indicator in linha.lower() for indicator in ["eur/kg", "€/kg", "/kg", "x —eur", "x —€"]):
            continue
            
        # Skip lines that are mostly placeholders or artifacts
        if "__unit_" in linha.lower() or re.match(r"^[_\s\-—]+", linha):
            continue
        
        # Padrões mais robustos para encontrar números
        # Procura por padrões como: 2 Cerveja 5.50 ou 1x Produto €3,20
        quantidade_pattern = r"(\d+)(?:x|\s)"
        preco_pattern = r"(\d+[,.]\d{1,2})"
        
        quantidade_match = re.search(quantidade_pattern, linha)
        precos_match = re.findall(preco_pattern, linha)
        
        if quantidade_match and precos_match:
            try:
                quantidade = int(quantidade_match.group(1))
                # Pega o último preço encontrado (geralmente o total)
                preco_total = float(precos_match[-1].replace(",", "."))
                
                # Extrai o nome do item de forma mais inteligente
                # Remove quantidade e preços da linha
                nome = re.sub(r"\d+x?", "", linha, count=1)  # Remove quantidade
                nome = re.sub(r"\d+[,.]\d+", "", nome)  # Remove preços
                nome = re.sub(r"[€$]", "", nome)  # Remove símbolos de moeda
                
                # Remove IVA tax codes and similar patterns
                nome = re.sub(r"\s+I\d+[A-Z%]*", "", nome)  # Remove patterns like "I3B", "I3%", "I9X"
                nome = re.sub(r"\s+\d+X", "", nome)  # Remove patterns like "23X"
                nome = re.sub(r"\s+[A-Z]+\d+[A-Z%]*", "", nome)  # Remove other tax codes
                nome = re.sub(r"\s*\|\s*", "", nome)  # Remove pipe symbols
                nome = re.sub(r"\s*['\"`»«]\s*", "", nome)  # Remove quotes and special chars
                
                # Remove numbers but preserve common units
                # Protect unit measurements first (simpler approach)
                nome = re.sub(r"(\d+(?:[.,]\d+)?)\s*(?:L|l|ML|ml|CL|cl|KG|kg|G|g)\b", r"PROTECTED_UNIT_\1_END", nome)
                
                # Now remove all other numbers
                nome = re.sub(r"\s*\d+\s*", " ", nome)  # Remove standalone numbers
                nome = re.sub(r"\d+", "", nome)  # Remove numbers attached to words
                
                # Restore protected units
                nome = re.sub(r"PROTECTED_UNIT_([^_]+)_END", r"\1", nome)
                
                nome = re.sub(r"\s{2,}", " ", nome).strip()  # Normaliza espaços
                
                # Correções de OCR mais abrangentes
                corrections = {
                    "Inperial": "Imperial",
                    "lmperial": "Imperial",
                    "Agua": "Água",
                    "agua": "água",
                    "pao": "pão",
                    "Pao": "Pão",
                    ".5h": "5L",
                    "0ne": "One",
                    "T-80ne": "T-Bone",
                    "T-B0ne": "T-Bone",
                }
                
                # Apply corrections
                for wrong, correct in corrections.items():
                    nome = nome.replace(wrong, correct)
                
                # Clean up the name further
                nome = nome.strip()
                
                # Remove trailing/leading special characters
                nome = re.sub(r"^[^\w]+|[^\w]+$", "", nome)
                
                # Only add if the name is meaningful (more than 2 chars and contains letters)
                if nome and len(nome) > 2 and re.search(r"[a-zA-ZáàãâéêíóôõúçÁÀÃÂÉÊÍÓÔÕÚÇ]", nome):
                    itens.append({
                        "item": nome,
                        "quantidade": quantidade,
                        "preco_total": preco_total
                    })
                    
            except (ValueError, IndexError):
                continue

    return itens


    
def main(argc=None, argv=None):
    if argc is None or argc != 2:
        print("[ERROR] Format: python read_invoice.py <image_path>")
        return
    
    image_path = argv[1]
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        return
    
    try:
        print(f"Processing invoice image: {image_path}")
        print("Applying image preprocessing...")
        
        # Process the invoice
        invoice_text = process_invoice(image_path, lang="por", save_processed=True)
        
        print(f"\n--- RAW OCR TEXT ---")
        print(invoice_text)
        print("--- END RAW TEXT ---\n")
        
        # Parse items from text
        print("Parsing invoice items...")
        itens = parse_invoice_text(invoice_text)
        
        if itens:
            print(f"\n--- FOUND {len(itens)} ITEMS ---")
            total_geral = 0
            for i, item in enumerate(itens, 1):
                print(f"{i}. {item['item']}")
                print(f"   Quantidade: {item['quantidade']}")
                print(f"   Preço Total: €{item['preco_total']:.2f}")
                print()
                total_geral += item['preco_total']
            
            print(f"TOTAL CALCULADO: €{total_geral:.2f}")
        else:
            print("[WARNING] No items were successfully parsed from the invoice.")
            print("Raw text has been displayed above for manual review.")
            
    except Exception as e:
        print(f"[ERROR] An error occurred while processing the invoice: {str(e)}")
        print("Please check that the image file is valid and readable.")

def read_invoice():
    main(len(sys.argv), sys.argv)

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)