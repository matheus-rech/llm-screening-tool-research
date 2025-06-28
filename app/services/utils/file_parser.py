import csv
import io
import os
import re
import rispy
import bibtexparser
from lxml import etree
from Bio import Entrez, Medline
from typing import List, Dict, Optional, Tuple
import operator
import os
try:
    from nltk.metrics import edit_distance
except ImportError:
    def edit_distance(s1, s2):
        """Simple edit distance implementation if NLTK is not available"""
        if len(s1) < len(s2):
            return edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

# --- FILE PARSING AND DATA LOADING ---

def parse_ris_file(file_content: str) -> List[Dict]:
    """Enhanced RIS parsing with better field handling and error recovery"""
    try:
        entries = rispy.loads(file_content)
        studies = []
        for entry in entries:
            authors = entry.get("authors", [])
            studies.append({
                "title": entry.get("title", entry.get("primary_title", "")),
                "abstract": entry.get("abstract", ""),
                "authors": ', '.join(authors) if isinstance(authors, list) else str(authors),
                "year": entry.get("year", ""),
                "journal_name": entry.get("journal_name", ""),
                "pmid": entry.get("pmid", "")
                                  })
        return studies
    except Exception as e:
        print(f"rispy parsing failed, trying manual parsing: {e}")
        return parse_ris_manual(file_content)

def parse_ris_manual(ris_data: str) -> List[Dict]:
    """
    Manual RIS parsing based on abstrackr-web approach for better compatibility.
    Handles edge cases that rispy might miss.
    """
    try:
        if ris_data.startswith('\ufeff'):
            ris_data = ris_data[1:]
        
        cur_id = 1
        ris_dict = {}
        
        re_pattern = re.compile(r'([A-Z][A-Z0-9]\s{2}-\s)', re.DOTALL)
        
        lines_list = re.split(re_pattern, ris_data)
        
        lines_list = [var for var in lines_list if var.strip()]
        
        current_citation = None
        cur_authors, cur_keywords = [], []
        
        for idx, line in enumerate(lines_list):
            if line == "TY  - ":
                current_citation = {
                    "title": "", "abstract": "", "journal_name": "",
                    "keywords": "", "pmid": "", "authors": "", "year": ""
                }
                cur_authors, cur_keywords = [], []
                
                while cur_id in ris_dict:
                    cur_id += 1
                
                ris_dict[cur_id] = current_citation
                
            elif current_citation is not None:
                if line in ("AU  - ", "A1  - ") and idx + 1 < len(lines_list):
                    cur_authors.append(lines_list[idx + 1].strip())
                elif line in ("T1  - ", "TI  - ") and idx + 1 < len(lines_list):
                    current_citation["title"] = lines_list[idx + 1].strip()[:500]  # Limit title length
                elif re.match(r"^J[A-Z0-9]\s{2}-\s|^T2\s{2}-\s", line) and idx + 1 < len(lines_list):
                    current_citation["journal_name"] = lines_list[idx + 1].strip()
                elif line == "KW  - " and idx + 1 < len(lines_list):
                    cur_keywords.extend([x.strip() for x in lines_list[idx + 1].splitlines() if x.strip()])
                elif line in ("N2  - ", "AB  - ") and idx + 1 < len(lines_list):
                    current_citation["abstract"] = lines_list[idx + 1].strip()
                elif line == "AN  - " and idx + 1 < len(lines_list):
                    current_citation["pmid"] = lines_list[idx + 1].strip()
                elif line == "PY  - " and idx + 1 < len(lines_list):
                    year_text = lines_list[idx + 1].strip()
                    year_match = re.search(r'\b(19|20)\d{2}\b', year_text)
                    if year_match:
                        current_citation["year"] = year_match.group()
                elif line == "ER  - ":
                    if current_citation:
                        current_citation["authors"] = ', '.join(cur_authors) if cur_authors else ''
                        current_citation["keywords"] = ', '.join(cur_keywords) if cur_keywords else ''
        
        studies = []
        for citation in ris_dict.values():
            if citation and citation.get("title"):  # Only include citations with titles
                studies.append({
                    "title": citation["title"],
                    "abstract": citation["abstract"],
                    "authors": citation["authors"],
                    "year": citation["year"],
                    "journal_name": citation["journal_name"],
                    "pmid": citation["pmid"]
                })
        
        return studies
        
    except Exception as e:
        print(f"Manual RIS parsing failed: {e}")
        return []

def parse_tsv_file(file_content: str) -> List[Dict]:
    studies = []
    file_like_object = io.StringIO(file_content)
    reader = csv.DictReader(file_like_object, delimiter='\t')
    for row in reader:
        studies.append({
            "studyid": row.get("id", row.get("pmid", "")),
            "title": row.get("title", ""),
            "abstract": row.get("abstract", "")
        })
    return studies

def parse_csv_file(file_content: str) -> List[Dict]:
    """Parse CSV content or file path into study dictionaries."""
    studies = []

    # Check if input is a file path
    if os.path.exists(file_content):
        # Security check: prevent path traversal attacks
        if ".." in file_content:
            raise ValueError("Path traversal not allowed")
        
        # Allow relative paths and temp files (for testing)
        if not os.path.isabs(file_content) or file_content.startswith('/tmp/'):
            with open(file_content, "r", encoding="utf-8") as f:
                file_content = f.read()
        else:
            # For absolute paths, ensure they're within allowed directories
            resolved_path = os.path.abspath(file_content)
            current_dir = os.path.abspath(".")
            if resolved_path.startswith(current_dir):
                with open(file_content, "r", encoding="utf-8") as f:
                    file_content = f.read()
            else:
                raise ValueError("File path not allowed for security reasons")

    """Parse CSV data from a file path or raw content."""
    studies = []

    # Allow passing a file path for convenience in tests
    if os.path.exists(file_content) and "\n" not in file_content:
        with open(file_content, "r", encoding="utf-8") as f:
            file_content = f.read()

    file_like_object = io.StringIO(file_content)
    reader = csv.DictReader(file_like_object)
    for row in reader:
        authors = row.get("authors", "")
        studies.append({
            "studyid": row.get("id", row.get("pmid", "")),
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
            "authors": ', '.join(authors.split(';')) if ';' in authors else str(authors),
            "year": row.get("year", ""),
            "journal_name": row.get("journal_name", ""),
            "pmid": row.get("pmid", ""),
            "keywords": row.get("keywords", "")
        })
    return studies

def parse_medline_txt_file(file_content: str) -> List[Dict]:
    studies = []
    file_like_object = io.StringIO(file_content)
    records = Medline.parse(file_like_object)
    for record in records:
        studies.append({
            "title": record.get("TI", ""),
            "abstract": record.get("AB", ""),
            "authors": record.get("AU", []),
            "year": record.get("DP", "").split()[0] if "DP" in record else "",
            "journal_name": record.get("JT", "")
        })
    return studies


def parse_xml_file(file_content: str) -> List[Dict]:
    studies = []
    try:
        root = etree.fromstring(file_content.encode('utf-8'))
        def get_text(parent, tagnames):
            for tagname in tagnames:
                element = parent.find(f".//{tagname}")
                if element is not None and element.text:
                    return element.text.strip()
            return ""
        for article_element in root.findall('.//PubmedArticle'):
            title = get_text(article_element, ['ArticleTitle'])
            abstract = get_text(article_element, ['AbstractText'])
            studies.append({'title': title, 'abstract': abstract})
    except etree.XMLSyntaxError as e:
        print(f"Error parsing XML: {e}")
    return studies

def parse_pmid_file_and_fetch(file_content: str, email: str) -> List[Dict]:
    Entrez.email = email
    pmids = [line.strip() for line in file_content.splitlines() if line.strip()]
    if not pmids:
        return []
    
    return batch_fetch_pubmed_articles(pmids)

def batch_fetch_pubmed_articles(pmids: List[str], batch_size: int = 20) -> List[Dict]:
    """Enhanced batch fetching with better error handling and progress tracking"""
    all_studies = []
    total = len(pmids)
    fetched_so_far = 0
    
    while fetched_so_far < total:
        batch_pmids = pmids[fetched_so_far:fetched_so_far + batch_size]
        try:
            studies = fetch_pubmed_articles(batch_pmids)
            all_studies.extend(studies)
            fetched_so_far += batch_size
            print(f"Fetched {min(fetched_so_far, total)} of {total} articles from PubMed")
        except Exception as e:
            print(f"Error fetching batch {fetched_so_far}-{fetched_so_far + batch_size}: {e}")
            fetched_so_far += batch_size
            continue
    
    return all_studies

def fetch_pubmed_articles(pmids: List[str]) -> List[Dict]:
    """Fetch articles from PubMed with enhanced error handling"""
    studies = []
    try:
        handle = Entrez.efetch(db="pubmed", id=pmids, rettype="medline", retmode="text")
        records = Medline.parse(handle)
        for record in records:
            studies.append({
                "title": record.get("TI", ""),
                "abstract": record.get("AB", ""),
                "authors": record.get("AU", []),
                "year": record.get("DP", "").split()[0] if "DP" in record else "",
                "journal_name": record.get("JT", ""),
                "pmid": record.get("PMID", "")
            })
    except Exception as e:
        print(f"Error fetching PubMed articles: {e}")
        return []
    
    return studies

def get_pmid_from_title(title: str, distance_threshold: int = 7) -> Optional[str]:
    """
    Enhanced title-to-PMID matching with fuzzy search capabilities.
    Searches PubMed title field for the given text. If a match is found with an edit
    distance of distance_threshold or less, return its PMID; otherwise return None.
    """
    try:
        handle = Entrez.esearch(db="pubmed", term=f'"{title}"', field="ti")
        records = Entrez.read(handle)
        id_list = records["IdList"]
        
        if len(id_list) == 0:
            handle = Entrez.esearch(db="pubmed", term=title)
            records = Entrez.read(handle)
            id_list = records["IdList"]
        
        if len(id_list) == 0:
            print(f"No records found for '{title}'")
            return None
        
        clean_title = title.rstrip('.').strip().lower()
        
        sorted_pmids = rank_pmids_by_edit_distance(clean_title, id_list)
        if not sorted_pmids:
            return None
        
        best_pmid, best_distance = sorted_pmids[0]
        
        if best_distance <= distance_threshold:
            return best_pmid
        
        print(f"No close match found - closest match ({best_pmid}) had edit distance of {best_distance}")
        return None
        
    except Exception as e:
        print(f"Error in title-to-PMID matching: {e}")
        return None

def rank_pmids_by_edit_distance(target_title: str, pmids: List[str]) -> List[Tuple[str, int]]:
    """
    Ranks PMIDs by their title's proximity to target_title using edit distance.
    Returns list of (pmid, distance) tuples sorted by distance.
    """
    scores = {}
    
    for pmid in pmids:
        try:
            articles = fetch_pubmed_articles([pmid])
            if not articles:
                continue
                
            article_title = articles[0].get("title", "").lower()
            if not article_title:
                continue
                
            scores[pmid] = edit_distance(target_title, article_title)
            
        except Exception as e:
            print(f"Error processing PMID {pmid}: {e}")
            continue
    
    return sorted(scores.items(), key=operator.itemgetter(1))

def parse_bibtex_file(file_content: str) -> List[Dict]:
    bib_database = bibtexparser.loads(file_content)
    studies = []
    for entry in bib_database.entries:
        studies.append({
            "title": entry.get("title", ""),
            "abstract": entry.get("abstract", ""),
            "authors": entry.get("author", "").split(" and "),
            "year": entry.get("year", ""),
            "journal_name": entry.get("journal", "")
        })
    return studies


def detect_file_format(filename: str, content: str = "") -> str:
    """Enhanced format detection using both filename and content analysis"""
    filename_lower = filename.lower()
    
    if filename_lower.endswith(".ris"):
        return "ris"
    if filename_lower.endswith(".csv"):
        return "csv"
    if filename_lower.endswith(".tsv"):
        return "tsv"
    if filename_lower.endswith(".xml"):
        return "xml"
    if filename_lower.endswith(".bib"):
        return "bibtex"
    if "pmid" in filename_lower and filename_lower.endswith(".txt"):
        return "pmid_list"
    if filename_lower.endswith(".txt"):
        if content:
            if looks_like_ris(content):
                return "ris"
            elif looks_like_medline(content):
                return "medline"
            elif looks_like_pmid_list(content):
                return "pmid_list"
            elif looks_like_tsv(content):
                return "tsv"
        return "medline"  # Default for .txt files
    
    if content:
        if looks_like_ris(content):
            return "ris"
        elif looks_like_tsv(content):
            return "tsv"
        elif looks_like_xml(content):
            return "xml"
        elif looks_like_bibtex(content):
            return "bibtex"
        elif looks_like_pmid_list(content):
            return "pmid_list"
        elif looks_like_medline(content):
            return "medline"
    
    return "unknown"

def looks_like_ris(content: str) -> bool:
    """Check if content appears to be RIS format"""
    if not content:
        return False
    
    if 'TY  - ' not in content:
        return False
    
    ris_pattern = re.compile(r'^[A-Z][A-Z0-9]\s{2}-\s', re.MULTILINE)
    matches = ris_pattern.findall(content)
    
    ris_specific_tags = ['TY  - ', 'ER  - ']  # Type and End Record are RIS-specific
    common_tags = ['TI  - ', 'AU  - ']  # Title and Author are common
    
    ris_specific_count = sum(1 for tag in ris_specific_tags if tag in content)
    common_tag_count = sum(1 for tag in common_tags if tag in content)
    
    print(f"DEBUG looks_like_ris: matches={len(matches)}, ris_specific_count={ris_specific_count}, common_tag_count={common_tag_count}")
    print(f"DEBUG content preview: {repr(content[:200])}")
    
    return len(matches) > 0 and ris_specific_count >= 1 and common_tag_count >= 1

def looks_like_tsv(content: str) -> bool:
    """Check if content appears to be TSV format"""
    if not content:
        return False
    
    lines = content.strip().split('\n')
    if len(lines) < 2:
        return False
    
    header_line = lines[0]
    headers = [x.lower().strip() for x in header_line.split('\t')]
    
    required_fields = ['title', 'abstract']
    optional_fields = ['author', 'year', 'journal', 'pmid', 'doi']
    
    has_required = all(any(field in header for header in headers) for field in required_fields)
    has_optional = any(any(field in header for header in headers) for field in optional_fields)
    
    return has_required and len(headers) > 1

def looks_like_xml(content: str) -> bool:
    """Check if content appears to be XML format"""
    if not content:
        return False
    
    content = content.strip()
    return content.startswith('<?xml') or content.startswith('<') and content.endswith('>')

def looks_like_bibtex(content: str) -> bool:
    """Check if content appears to be BibTeX format"""
    if not content:
        return False
    
    bibtex_pattern = re.compile(r'@\w+\s*\{', re.IGNORECASE)
    return bool(bibtex_pattern.search(content))

def looks_like_pmid_list(content: str) -> bool:
    """Check if content appears to be a list of PMIDs"""
    if not content:
        return False
    
    lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
    if len(lines) < 1:
        return False
    
    numeric_lines = sum(1 for line in lines if line.isdigit() and len(line) >= 6)
    return numeric_lines / len(lines) > 0.7

def looks_like_medline(content: str) -> bool:
    """Check if content appears to be Medline format"""
    if not content:
        return False
    
    medline_pattern = re.compile(r'^[A-Z]{2,4}\s{2}-', re.MULTILINE)
    matches = medline_pattern.findall(content)
    
    common_tags = ['PMID-', 'TI  -', 'AB  -', 'AU  -']
    tag_count = sum(1 for tag in common_tags if tag in content)
    
    return len(matches) > 0 and tag_count >= 2


def load_studies(file_content: str, filename: str, entrez_email: str = "") -> List[Dict]:
    """
    Enhanced study loading with improved format detection and error handling.
    """
    if entrez_email:
        Entrez.email = entrez_email
    
    file_format = detect_file_format(filename, file_content)
    
    print(f"Detected file format: {file_format} for file: {filename}")

    try:
        if file_format == "ris":
            return parse_ris_file(file_content)
        elif file_format == "csv":
            return parse_csv_file(file_content)
        elif file_format == "tsv":
            return parse_tsv_file(file_content)
        elif file_format == "xml":
            return parse_xml_file(file_content)
        elif file_format == "bibtex":
            return parse_bibtex_file(file_content)
        elif file_format == "medline":
            return parse_medline_txt_file(file_content)
        elif file_format == "pmid_list":
            if not entrez_email:
                raise ValueError("Email is required for fetching PubMed data.")
            return parse_pmid_file_and_fetch(file_content, entrez_email)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    except Exception as e:
        print(f"Error parsing {file_format} file: {e}")
        return try_fallback_parsing(file_content, filename, entrez_email)

def try_fallback_parsing(file_content: str, filename: str, entrez_email: str = "") -> List[Dict]:
    """Try multiple parsing methods as fallback when primary detection fails"""
    print("Attempting fallback parsing methods...")
    
    parsers = [
        ("RIS", parse_ris_file),
        ("TSV", parse_tsv_file),
        ("CSV", parse_csv_file),
        ("Medline", parse_medline_txt_file),
        ("XML", parse_xml_file),
        ("BibTeX", parse_bibtex_file)
    ]
    
    for parser_name, parser_func in parsers:
        try:
            print(f"Trying {parser_name} parser...")
            studies = parser_func(file_content)
            if studies and len(studies) > 0:
                print(f"Successfully parsed with {parser_name} parser: {len(studies)} studies")
                return studies
        except Exception as e:
            print(f"{parser_name} parser failed: {e}")
            continue
    
    if entrez_email and looks_like_pmid_list(file_content):
        try:
            print("Trying PMID list parser as last resort...")
            return parse_pmid_file_and_fetch(file_content, entrez_email)
        except Exception as e:
            print(f"PMID list parser failed: {e}")
    
    print("All parsing methods failed")
    return []

def load_studies_with_source_tracking(file_content: str, filename: str, entrez_email: str = "") -> tuple:
    """
    Enhanced study loading with source tracking and improved error handling.
    Returns: (studies, source_database)
    """
    if entrez_email:
        Entrez.email = entrez_email
    
    file_format = detect_file_format(filename, file_content)
    source_database = detect_source_database(filename, file_content)
    
    print(f"Detected format: {file_format}, source: {source_database}")
    
    try:
        studies = load_studies(file_content, filename, entrez_email)
        
        # Override source database for PMID lists
        if file_format == "pmid_list":
            source_database = "PubMed"
        
        return studies, source_database
        
    except Exception as e:
        print(f"Error in load_studies_with_source_tracking: {e}")
        return [], "Unknown"

def detect_source_database(filename: str, content: str) -> str:
    """Detect source database from filename or content patterns."""
    filename_lower = filename.lower()
    
    if "pubmed" in filename_lower or "medline" in filename_lower:
        return "PubMed"
    elif "scopus" in filename_lower:
        return "Scopus"
    elif "embase" in filename_lower:
        return "Embase"
    elif "webofscience" in filename_lower or "wos" in filename_lower:
        return "Web of Science"
    elif "cochrane" in filename_lower:
        return "Cochrane Library"
    
    if content.startswith("TY  -"):
        if "DB  - PubMed" in content or "DP  -" in content:
            return "PubMed"
        elif "DB  - Scopus" in content:
            return "Scopus"
        elif "DB  - Embase" in content:
            return "Embase"
    
    return "Unknown"

def search_and_enrich_studies(studies: List[Dict], entrez_email: str = "") -> List[Dict]:
    """
    Enrich studies by attempting to find PMIDs for studies that don't have them.
    This uses the enhanced title-to-PMID matching from abstrackr-web.
    """
    if entrez_email:
        Entrez.email = entrez_email
    
    enriched_studies = []
    
    for study in studies:
        enriched_study = study.copy()
        
        if not study.get("pmid") and study.get("title"):
            title = study["title"].strip()
            if len(title) > 10:  # Only try for reasonable length titles
                print(f"Searching PMID for: {title[:50]}...")
                pmid = get_pmid_from_title(title)
                if pmid:
                    enriched_study["pmid"] = pmid
                    print(f"Found PMID: {pmid}")
                    
                    try:
                        pubmed_data = fetch_pubmed_articles([pmid])
                        if pubmed_data:
                            pubmed_study = pubmed_data[0]
                            if not enriched_study.get("abstract") and pubmed_study.get("abstract"):
                                enriched_study["abstract"] = pubmed_study["abstract"]
                            if not enriched_study.get("authors") and pubmed_study.get("authors"):
                                enriched_study["authors"] = pubmed_study["authors"]
                            if not enriched_study.get("year") and pubmed_study.get("year"):
                                enriched_study["year"] = pubmed_study["year"]
                    except Exception as e:
                        print(f"Error enriching study with PMID {pmid}: {e}")
        
        enriched_studies.append(enriched_study)
    
    return enriched_studies                                                                                                                                                                                                