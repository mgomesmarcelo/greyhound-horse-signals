import sys
from bs4 import BeautifulSoup

def main():
    with open('timeform_sample.html', encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text('\n', strip=True)
    with open('timeform_sample.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    
    # Try to find class/grade and traps manually by CSS classes
    # Timeform usually uses something like .rpf-runner or .rpf-race-title
    print("Classificacao provavel:")
    for el in soup.select("h1, h2, h3, .race-title, .rpf-race-title, .rpf-header"):
        print("Header/Title:", el.get_text(strip=True))
        
    print("\nRunners provaveis:")
    for el in soup.select("li, tr, .runner, .rpf-runner"):
        text = el.get_text(" | ", strip=True)
        if "Trap" in text or text.startswith("1 |") or text.startswith("2 |"):
            print("Runner:", text)

if __name__ == "__main__":
    main()
