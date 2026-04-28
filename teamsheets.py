import re
import pandas as pd
from pypdf import PdfReader

# CONFIG: Load the PDF file
pdf_path = "U.S. Swimming Team Club Ranking 2022.pdf"
reader = PdfReader(pdf_path)
data = []

print("Starting extraction...")

for page in reader.pages:
    # Extract text preserving physical layout (critical for table parsing)
    text = page.extract_text(extraction_mode="layout") #layout keeps the space
    lines = text.split('\n')
    
    for line in lines:
        # Split by multi-space delimiters to separate columns
        parts = re.split(r'\s{2,}', line.strip())
        parts = [p for p in parts if p] # Remove empty strings
        
        # VALIDATION LOGIC:
        # Valid rows must start with a Rank (digit) and end with State (2 chars)
        # Structure: [Rank, QRank, Score, LSC, Team Name, City, State]
        if len(parts) >= 7 and parts[0].replace(',', '').isdigit() and len(parts[-1]) == 2:
            
            # Extract the last 3 columns (State, City, Team Name)
            state = parts[-1]
            city = parts[-2]
            
            # Team Name is everything between 'LSC' (index 3) and 'City' (index -2)
            # We rejoin them in case the name itself had large spaces
            if len(parts) == 7:
                team_name = parts[4]
            else:
                team_name = " ".join(parts[4:-2])
                
            data.append([team_name, city, state])

# Export to CSV
df = pd.DataFrame(data, columns=['Team Name', 'City', 'State'])
df.to_csv("teams_master_list.csv", index=False)
print(f"Extraction Complete. Saved {len(df)} teams to 'teams_master_list.csv'")