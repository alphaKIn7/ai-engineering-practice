from markdown_pdf import MarkdownPdf, Section
from pathlib import Path

base = Path(__file__).parent
md_path = base / "phase1_notes.md"
pdf_path = base / "phase1_notes.pdf"

text = md_path.read_text()

pdf = MarkdownPdf(toc_level=2, optimize=True)

css = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 11pt; line-height: 1.5; color: #1f2328; }
h1 { font-size: 22pt; border-bottom: 2px solid #d0d7de; padding-bottom: 6px; margin-top: 24px; }
h2 { font-size: 16pt; border-bottom: 1px solid #d8dee4; padding-bottom: 4px; margin-top: 20px; }
h3 { font-size: 13pt; margin-top: 16px; color: #24292f; }
h4 { font-size: 11pt; margin-top: 12px; }
code { background: #f6f8fa; padding: 1px 4px; border-radius: 3px; font-family: 'SF Mono', Menlo, monospace; font-size: 9.5pt; }
pre { background: #f6f8fa; padding: 10px; border-radius: 6px; font-size: 9pt; overflow-x: auto; }
pre code { background: transparent; padding: 0; font-size: 9pt; }
table { border-collapse: collapse; margin: 8px 0; font-size: 10pt; }
th, td { border: 1px solid #d0d7de; padding: 5px 8px; text-align: left; }
th { background: #f6f8fa; }
blockquote { border-left: 3px solid #d0d7de; padding-left: 12px; color: #57606a; margin: 8px 0; }
hr { border: 0; border-top: 1px solid #d0d7de; margin: 20px 0; }
strong { color: #0a0a0a; }
"""

pdf.add_section(Section(text, toc=True), user_css=css)
pdf.meta["title"] = "Phase 1 Study Notes — AI Engineering"
pdf.meta["author"] = "Shubham"

pdf.save(str(pdf_path))
print(f"Wrote {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")
