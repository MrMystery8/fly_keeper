import json
def frame(payload):
    """Compact observational protocol; no controls or full graph state."""
    return json.dumps({'type':'frame',**payload},separators=(',',':'))
