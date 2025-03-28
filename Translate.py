# English-Swahili Translation using Transformers
from transformers import pipeline
import warnings

# Suppress some warnings for cleaner output
warnings.filterwarnings('ignore')

# Initialize the translation pipeline
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-swc")

# Define a function for bidirectional translation
def translate_text(text, source_lang='en', target_lang='sw'):
    if source_lang == 'en' and target_lang == 'sw':
        # English to Swahili
        result = translator(text)
        return result[0]['translation_text']
    elif source_lang == 'sw' and target_lang == 'en':
        # Swahili to English (we'll use the reverse model)
        translator_sw_en = pipeline("translation", model="Helsinki-NLP/opus-mt-swc-en")
        result = translator_sw_en(text)
        return result[0]['translation_text']
    else:
        return "Unsupported language pair"

# Test examples
examples = [
    ("Hello, how are you?", 'en', 'sw'),
    ("Ninafurahi kukutana nawe", 'sw', 'en'),
    ("What is your name?", 'en', 'sw'),
    ("Naitwa John", 'sw', 'en')
]

# Run translations and print results
print("Translation Examples:")
print("=" * 50)
for text, src, tgt in examples:
    translation = translate_text(text, src, tgt)
    print(f"Source ({src}): {text}")
    print(f"Target ({tgt}): {translation}")
    print("-" * 50)