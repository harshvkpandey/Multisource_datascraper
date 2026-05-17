from __future__ import annotations
import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any
from Bio import Entrez

from scraper.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

# NCBI requires an email for all API usage
Entrez.email = os.getenv("NCBI_EMAIL", "researcher@example.com")
if _api_key := os.getenv("NCBI_API_KEY"):
    Entrez.api_key = _api_key
    logger.info("[PubMedScraper] NCBI API key loaded — rate limit: 10 req/s")
else:
    logger.info("[PubMedScraper] No NCBI API key — rate limit: 3 req/s")

# PageRank ceiling: log-normalisation denominator for C_score
CITATION_COUNT_MAX: int = int(os.getenv("CITATION_COUNT_MAX", "50000"))

_MONTH_MAP: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_month(raw: str) -> int:
    """Convert month string (numeric or abbreviated name) to integer."""
    try:
        return int(raw)
    except ValueError:
        return _MONTH_MAP.get(raw.lower()[:3], 1)


class PubMedScraper(BaseScraper):
    async def scrape(self, url: str) -> dict[str, Any]:
        """
        Scrape a PubMed article.

        Args:
            url: PubMed URL (https://pubmed.ncbi.nlm.nih.gov/{PMID}/)
                 or raw PMID string.
        """
        logger.info("[PubMedScraper] Scraping: %s", url)
        pmid = self._extract_pmid(url)
        if not pmid:
            raise ValueError(f"[PubMedScraper] Cannot extract PMID from: {url}")

        article = self._fetch_article(pmid)
        article["citation_count"] = self._fetch_citation_count(pmid)
        article["source_url"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        article["source_type"] = "pubmed"
        article["domain"] = "pubmed.ncbi.nlm.nih.gov"
        article["domain_authority"] = 1.0  # PubMed is a .gov seed domain

        return article
    @staticmethod
    def _extract_pmid(url: str) -> str | None:
        """Extract PMID from a PubMed URL or return raw numeric string."""
        import re
        match = re.search(r"\b(\d{7,9})\b", url)
        return match.group(1) if match else None
    def _fetch_article(self, pmid: str) -> dict[str, Any]:
        """
        Fetch structured article data via Entrez efetch XML.

        Returns a dict with: title, author, authors_list, journal,
        abstract, text, published_date, doi.
        """
        result: dict[str, Any] = {
            "title": None,
            "author": None,
            "authors_list": [],
            "journal": None,
            "abstract": "",
            "text": "",
            "published_date": None,
            "doi": None,
        }

        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
            xml_bytes = handle.read()
            handle.close()

            root = ET.fromstring(xml_bytes)
            art = root.find(".//PubmedArticle/MedlineCitation/Article")
            if art is None:
                logger.error("[PubMedScraper] No <Article> element for PMID %s", pmid)
                return result

            # Title
            t_el = art.find("ArticleTitle")
            if t_el is not None:
                result["title"] = "".join(t_el.itertext()).strip()

            # Authors — multiple-author handling: average credibility approach
            authors: list[str] = []
            for a_el in art.findall(".//AuthorList/Author"):
                last = a_el.findtext("LastName", "")
                fore = a_el.findtext("ForeName", "")
                name = f"{fore} {last}".strip()
                if name:
                    authors.append(name)
            result["authors_list"] = authors

            if authors:
                display = authors[0]
                if len(authors) > 1:
                    display += f" et al. (+{len(authors) - 1})"
                result["author"] = display
            else:
                logger.warning(
                    "[PubMedScraper] No authors for PMID %s — A_cred penalty applies.",
                    pmid,
                )

            # Journal
            j_el = art.find(".//Journal/Title")
            if j_el is not None:
                result["journal"] = j_el.text.strip()

            # Abstract (may have labelled sections: BACKGROUND, METHODS, etc.)
            parts: list[str] = []
            for abs_el in art.findall(".//Abstract/AbstractText"):
                label = abs_el.get("Label", "")
                body = "".join(abs_el.itertext()).strip()
                if body:
                    parts.append(f"{label}: {body}" if label else body)
            result["abstract"] = "\n\n".join(parts)
            result["text"] = result["abstract"]

            # Publication date
            pd_el = art.find(".//Journal/JournalIssue/PubDate")
            if pd_el is not None:
                year_str = pd_el.findtext("Year", "")
                month_str = pd_el.findtext("Month", "1")
                day_str = pd_el.findtext("Day", "1")
                if year_str:
                    try:
                        dt = datetime(
                            int(year_str),
                            _parse_month(month_str),
                            int(day_str),
                        )
                        result["published_date"] = dt.isoformat()
                    except (ValueError, TypeError):
                        result["published_date"] = year_str

            # DOI
            for aid in root.findall(".//ArticleId"):
                if aid.get("IdType") == "doi":
                    result["doi"] = aid.text
                    break

        except Exception as exc:  # noqa: BLE001
            logger.error("[PubMedScraper] efetch error for PMID %s: %s", pmid, exc)

        return result

    def _fetch_citation_count(self, pmid: str) -> int:
        
        try:
            handle = Entrez.elink(
                dbfrom="pubmed",
                db="pmc",
                id=pmid,
                linkname="pubmed_pmc_refs",
            )
            record = Entrez.read(handle)
            handle.close()

            link_sets = record[0].get("LinkSetDb", [])
            if link_sets:
                count = len(link_sets[0].get("Link", []))
                logger.info("[PubMedScraper] Citation count for PMID %s: %d", pmid, count)
                return count

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[PubMedScraper] elink failed for PMID %s: %s. Defaulting to 0.",
                pmid,
                exc,
            )
        return 0
