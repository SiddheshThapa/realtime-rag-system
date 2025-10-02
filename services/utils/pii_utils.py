# services/utils/pii_utils.py
import re
EMAIL = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE = re.compile(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?[\d\s-]{7,15}')

def mask_pii(text: str) -> str:
    if not text:
        return ""
    text = EMAIL.sub('[EMAIL]', text)
    text = PHONE.sub('[PHONE]', text)
    return text
