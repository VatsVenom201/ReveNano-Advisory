import re
import os
import pandas as pd
from docx import Document
import fitz  # PyMuPDF
from io import BytesIO

SOIL_PARAMETERS = [
    ("pH", "pH / પીએચ"),
    ("Electrical Conductivity", "Electrical Conductivity / વુત ચાલકતા"),
    ("Organic Carbon", "Organic Carbon / સવ કાબન"),
    ("Nitrogen", "Nitrogen / નાઇટોજન"),
    ("Phosphorus", "Phosphorus / ફોરસ"),
    ("Potassium", "Potassium / પોટેશયમ"),
    ("Sulphur", "Sulphur / ગંધક / સફર"),
    ("Boron", "Boron / બોરોન"),
    ("Copper", "Copper / તા / તાંબુ"),
    ("Ferrous", "Ferrous / લોહતવ"),
    ("Manganese", "Manganese / મેગેનીઝ"),
    ("Zinc", "Zinc / ઝીંક"),
    ("Lead Absorbance Index", "Lead Absorbance Index / સીસું શોષણ સૂચકાંક"),
    ("Termite Influence Index", "Termite Influence Index / ઉધઈ અસર સૂચકાંક"),
    ("Mercury Residual Index", "Mercury Residual Index / પારો અવશેષ સૂચકાંક"),
    ("Aflatoxin Risk Index", "Aflatoxin Risk Index / આલાટોસન ેખમ સૂચકાંક"),
    ("Relative Bacterial Index", "Relative Bacterial Index / સાપે બેટેરયલ સૂચકાંક"),
    ("Chemical Residue Index", "Chemical Residue Index / રાસાયણક અવશેષ સૂચકાંક"),
    ("Fungal Activity Index", "Fungal Activity Index / ફંગલ વૃ સૂચકાંક"),
    ("Humus Content Index", "Humus Content Index / ુમસ અંશ સૂચકાંક"),
    ("Carbon/Nitrogen", "Carbon/Nitrogen / કાબન થી નાઇટોજન ગુણોર")
]

def extract_soil_report_data(text: str) -> str:
    """
    Detects if the text is a soil report and extracts key parameters.
    Returns a cleaned string of parameters or None if not a soil report.
    """
    # Detect report markers
    low_text = text.lower()
    if "soil analysis report" not in low_text and "soil sample name" not in low_text:
        return None
        
    extracted_data = []
    extracted_data.append("--- DETECTED SOIL ANALYSIS REPORT ---")
    
    # Try to extract Report ID and Metadata
    report_id_match = re.search(r"Report ID:\s+([a-fA-F0-9-]+)", text, re.IGNORECASE)
    if report_id_match:
        extracted_data.append(f"Report ID: {report_id_match.group(1)}")
        
    sample_name_match = re.search(r"Soil Sample Name:\s+(.*?)(?=\s+Date|$)", text, re.IGNORECASE)
    if sample_name_match:
        extracted_data.append(f"Sample Name: {sample_name_match.group(1)}")

    # Extract specific parameters
    found_any = False
    for eng_name, full_name in SOIL_PARAMETERS:
        # Match English anchor followed by anything (non-greedy) and then the value.
        # We allow spaces in the English anchor name to be variable \s*
        anchor_pattern = re.escape(eng_name).replace(r"\ ", r"\s*")
        pattern = anchor_pattern + r".*?(\d+\.?\d*)"
        
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            val = match.group(1)
            extracted_data.append(f"{full_name}: {val}")
            found_any = True
            
    if not found_any:
        return None
        
    extracted_data.append("--- END OF SOIL DATA ---")
    return "\n".join(extracted_data)

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extracts text from various file formats."""
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        if ext == ".pdf":
            # Using PyMuPDF (fitz) for better robustness and speed
            doc = fitz.open(stream=file_content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
            
        elif ext == ".docx":
            doc = Document(BytesIO(file_content))
            return "\n".join([para.text for para in doc.paragraphs])
            
        elif ext in [".xlsx", ".xls"]:
            df_dict = pd.read_excel(BytesIO(file_content), sheet_name=None)
            full_text = ""
            for sheet_name, df in df_dict.items():
                full_text += f"\n--- Sheet: {sheet_name} ---\n"
                full_text += df.to_string(index=False)
            return full_text
            
        elif ext == ".txt":
            return file_content.decode("utf-8")
            
        else:
            return ""
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        return ""
