import modal
import json
import xml.etree.ElementTree as ET
from typing import Dict

cls = modal.Cls.lookup("grobid-modal", "Model")
obj = cls()


def parse_tei_xml(tei_content: str) -> Dict:
    # Define the namespaces
    namespaces = {
        "tei": "http://www.tei-c.org/ns/1.0",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xlink": "http://www.w3.org/1999/xlink",
    }

    root = ET.fromstring(tei_content)

    # Extract basic metadata
    result = {
        "title": "",
        "authors": [],
        "published_date": "",
        "abstract": "",
        "references": [],
    }

    # Get title (using namespace)
    title = root.find(".//tei:titleStmt/tei:title", namespaces)
    if title is not None:
        result["title"] = title.text.strip()

    # Get authors
    for author in root.findall(".//tei:author", namespaces):
        author_info = {"first_name": "", "last_name": "", "affiliation": ""}

        forename = author.find(".//tei:forename", namespaces)
        surname = author.find(".//tei:surname", namespaces)
        affiliation = author.find(".//tei:affiliation/tei:orgName", namespaces)

        if forename is not None:
            author_info["first_name"] = forename.text
        if surname is not None:
            author_info["last_name"] = surname.text
        if affiliation is not None:
            author_info["affiliation"] = affiliation.text

        result["authors"].append(author_info)

    # Get publication date
    date = root.find(".//tei:publicationStmt/tei:date", namespaces)
    if date is not None:
        result["published_date"] = date.get("when", date.text)

    # Get abstract
    abstract = root.find(".//tei:abstract", namespaces)
    if abstract is not None:
        result["abstract"] = " ".join(
            [p.text.strip() for p in abstract.findall(".//tei:p", namespaces) if p.text]
        )

    return result


def parse_document(fname: bytes) -> dict:
    response = obj.parse_document.remote(fname)
    if response["status"] == "success":
        return parse_tei_xml(response["tei"])
    return {"error": "Failed to parse document"}


if __name__ == "__main__":
    pdf_path = "/Users/pranaviyer/Desktop/AstralisData/E5_Paper.pdf"
    with open(pdf_path, "rb") as f:
        output = parse_document(f.read())
        with open("output.json", "w") as f:
            json.dump(output, f, indent=4)
