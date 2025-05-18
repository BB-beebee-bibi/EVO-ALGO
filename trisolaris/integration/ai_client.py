"""
Client for AI assistants to interact with the central document.
"""
import requests
import json
import logging
from typing import Dict, Any, Optional

class TrisolarisDocs:
    """Client for AI assistants to interact with the central document."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Initialize the client.
        
        Args:
            base_url: URL of the webhook server
        """
        self.base_url = base_url
        self.content_cache = None
        
        # Configure logging
        self.logger = logging.getLogger('trisolaris_docs_client')
    
    def get_document(self) -> str:
        """
        Get the current document content.
        
        Returns:
            Document content as text
        """
        try:
            response = requests.get(f"{self.base_url}/content")
            response.raise_for_status()
            
            data = response.json()
            self.content_cache = data.get('content', '')
            
            return self.content_cache
            
        except Exception as e:
            self.logger.error(f"Error getting document: {e}")
            return self.content_cache or ""
    
    def update_document(self, content: str, position: int = 1) -> bool:
        """
        Update the document content.
        
        Args:
            content: New content to write
            position: Position to insert (1 for start of document)
            
        Returns:
            Success status
        """
        try:
            response = requests.post(
                f"{self.base_url}/update",
                json={"content": content, "position": position}
            )
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating document: {e}")
            return False
    
    def extract_section(self, section_name: str) -> Optional[str]:
        """
        Extract a specific section from the document.
        
        Args:
            section_name: Name of the section to extract
            
        Returns:
            Section content or None if not found
        """
        content = self.get_document()
        if not content:
            return None
            
        # Try to find section
        section_start = content.find(f"# {section_name}")
        if section_start == -1:
            section_start = content.find(f"## {section_name}")
            
        if section_start == -1:
            return None
            
        # Find end of section (next heading or end of document)
        next_heading = -1
        for marker in ["# ", "## "]:
            pos = content.find(marker, section_start + len(section_name) + 3)
            if pos != -1 and (next_heading == -1 or pos < next_heading):
                next_heading = pos
                
        # Extract section content
        if next_heading != -1:
            return content[section_start:next_heading].strip()
        else:
            return content[section_start:].strip()
    
    def update_section(self, section_name: str, content: str) -> bool:
        """
        Update a specific section in the document.
        
        Args:
            section_name: Name of the section to update
            content: New content for the section
            
        Returns:
            Success status
        """
        doc_content = self.get_document()
        if not doc_content:
            return False
            
        # Try to find section
        section_start = doc_content.find(f"# {section_name}")
        is_h1 = section_start != -1
        
        if not is_h1:
            section_start = doc_content.find(f"## {section_name}")
            
        if section_start == -1:
            # Section doesn't exist, append it
            heading = f"# {section_name}" if is_h1 else f"## {section_name}"
            return self.update_document(f"\n\n{heading}\n\n{content}", len(doc_content))
            
        # Find end of section
        next_heading = -1
        for marker in ["# ", "## "]:
            pos = doc_content.find(marker, section_start + len(section_name) + 3)
            if pos != -1 and (next_heading == -1 or pos < next_heading):
                next_heading = pos
                
        # Create updated document
        if next_heading != -1:
            updated_content = (
                doc_content[:section_start] +
                (f"# {section_name}" if is_h1 else f"## {section_name}") +
                "\n\n" + content + "\n\n" +
                doc_content[next_heading:]
            )
        else:
            updated_content = (
                doc_content[:section_start] +
                (f"# {section_name}" if is_h1 else f"## {section_name}") +
                "\n\n" + content
            )
            
        # Update entire document
        return self.update_document(updated_content, 1) 