"""
DLP Name Mapping from Vana Analytics Dashboard

Source: https://vana-analytics-opendatalabs.vercel.app/
Extracted from the DLP dropdown on 2025-12-18
"""

DLP_NAME_MAPPING = {
    1: "VanaTensor",
    2: "DNA DAO",
    3: "PrimeInsights",
    4: "dFusion Social Truth",
    5: "Auto DLP",
    6: "Volara",
    7: "SixGPT",
    8: "vChars AI",
    9: "MindDAO",
    10: "YKYR",
    11: "DataPig",
    12: "PrimeInsights",
    13: "Finquarium",
    14: "de-old",
    16: "Knowhere",
    17: "UNWRAPPED",
    18: "Intra",
    19: "Defai",
    20: "Sleep.Fun",
    21: "RuiAi",
    22: "Voogle",
    23: "InspectDLP",
    24: "Deoracle",
    25: "SwipeLabel",
    26: "Vanagent",
    27: "KLEO NETWORK",
    28: "KLEO.NETWORK",
    29: "zLayer",
    30: "Second Brain",
    31: "Seekers",
    32: "GPT Data DAO",
    33: "AsteriskDLP",
    34: "WorldAIHealth",
    35: "Barbarika",
    36: "Audata-Test",
    37: "Devdock-Test",
    38: "Audata",
    39: "CredMont",
    40: "r/datadao",
    41: "!Vpoints",
    42: "Vpoints",
    43: "Vpoints!",
    44: "VDataDao",
}


def get_dlp_name(dlp_id):
    """
    Get DLP name from mapping, with fallback.
    
    Args:
        dlp_id: DLP ID (int, str, or float)
        
    Returns:
        DLP name string
    """
    if isinstance(dlp_id, str):
        try:
            dlp_id = int(float(dlp_id))
        except (ValueError, TypeError):
            return f"DLP {dlp_id}"
    
    try:
        dlp_id = int(dlp_id)
    except (ValueError, TypeError):
        return f"DLP {dlp_id}"
    
    return DLP_NAME_MAPPING.get(dlp_id, f"DLP {dlp_id}")


def get_all_dlp_names():
    """
    Get all DLP names as a dictionary.
    
    Returns:
        Dictionary mapping DLP IDs to names
    """
    return DLP_NAME_MAPPING.copy()

