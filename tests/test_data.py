##################
# MOCK DATA FOR TESTS #
##################

# Sample OCR text outputs for testing
SAMPLE_RESTAURANT_RECEIPT = """
RESTAURANTE O FADO
Mesa: 12
Data: 20/10/2025

1 T-Bone I3B 13.00
1 Peixe I3 6.50
2 Imperial 23X 2.20
1 Água SL I9X 1.30
2 pão 13 1.00
1 azeitonas 13 1.00

SubTotal: 25.00
IVA (23%): 5.75
Total: 30.75

Obrigado pela preferência!
"""

SAMPLE_GROCERY_RECEIPT = """
CONTINENTE
TOMATE DE CACHO , B
540 x 1.49 EUR/kg 0.80

BATATA CONS. ROXA BX , B  
345 x 0.99 EUR/kg 2.37

AMEIXA RAIN.CLA.EMB. , B
99 1.99

MOZZARELLA SORT. , B
89 0.89

ÁGUA DE NASCENTE , B
74 1.74

Total: 7.79
"""

SAMPLE_BAR_RECEIPT = """
CAFÉ CENTRAL
Mesa 5

2x Cerveja 330ml 5.00
1x Vinho Tinto 750ml 8.50
3x Pastel de Nata 4.50
1x Café Expresso 1.20

Total: 19.20
"""

# Expected parsing results
EXPECTED_RESTAURANT_ITEMS = [
    {"item": "T-Bone", "quantidade": 1, "preco_total": 13.00},
    {"item": "Peixe", "quantidade": 1, "preco_total": 6.50},
    {"item": "Imperial", "quantidade": 2, "preco_total": 2.20},
    {"item": "Água", "quantidade": 1, "preco_total": 1.30},
    {"item": "pão", "quantidade": 2, "preco_total": 1.00},
    {"item": "azeitonas", "quantidade": 1, "preco_total": 1.00},
]

EXPECTED_GROCERY_ITEMS = [
    {"item": "AMEIXA RAIN.CLA.EMB.", "quantidade": 99, "preco_total": 1.99},
    {"item": "MOZZARELLA SORT.", "quantidade": 89, "preco_total": 0.89},
    {"item": "ÁGUA DE NASCENTE", "quantidade": 74, "preco_total": 1.74},
]

EXPECTED_BAR_ITEMS = [
    {"item": "Cerveja 330ml", "quantidade": 2, "preco_total": 5.00},
    {"item": "Vinho Tinto 750ml", "quantidade": 1, "preco_total": 8.50},
    {"item": "Pastel de Nata", "quantidade": 3, "preco_total": 4.50},
    {"item": "Café Expresso", "quantidade": 1, "preco_total": 1.20},
]
