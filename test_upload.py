import requests

pdf_bytes = b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n149\n%EOF\n"

with open("dummy.pdf", "wb") as f:
    f.write(pdf_bytes)

url = "http://localhost:8000/upload"
files = {"file": ("dummy.pdf", open("dummy.pdf", "rb"), "application/pdf")}

try:
    response = requests.post(url, files=files)
    print("STATUS:", response.status_code)
    print("RESPONSE:", response.text)
except Exception as e:
    print("ERROR:", str(e))
