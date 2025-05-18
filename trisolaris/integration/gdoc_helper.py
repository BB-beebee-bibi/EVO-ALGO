#!/usr/bin/env python3
"""
Helper module for simplified Google Docs integration.
This provides a minimal interface for AI agents to interact with the central document.
"""
import os
from typing import Optional
from trisolaris.integration.google_docs_sync import GoogleDocsSyncService

# The central document ID
DOC_ID = "1fk0TkyC7xsKgw2yGV6lzTn99zU6u2SKuNU2RdYxMj9w"

def gdoc_service(ai_id: str = "local-ai") -> GoogleDocsSyncService:
    """
    Create a Google Docs sync service instance for an AI agent.
    
    Args:
        ai_id: Unique identifier for the AI agent
        
    Returns:
        Configured GoogleDocsSyncService instance
    """
    credentials_path = os.getenv("TRISOLARIS_GDOC_KEY")
    if not credentials_path:
        raise ValueError("TRISOLARIS_GDOC_KEY environment variable not set")
        
    return GoogleDocsSyncService(
        doc_id=DOC_ID,
        user_id=ai_id,
        user_type="AI",
        credentials_path=credentials_path,
        update_interval=60  # Check for updates every minute
    )

async def read_section(ai_id: str, section_id: str) -> Optional[str]:
    """
    Read the content of a specific section from the document.
    
    Args:
        ai_id: Unique identifier for the AI agent
        section_id: ID of the section to read
        
    Returns:
        Section content if found, None otherwise
    """
    doc = gdoc_service(ai_id)
    content = doc.fetch_document(force=True)
    
    # Extract section content using the standard format
    start_marker = f"#§ {section_id}:"
    end_marker = f"#§ END_{section_id}"
    
    try:
        start_idx = content.index(start_marker) + len(start_marker)
        end_idx = content.index(end_marker)
        return content[start_idx:end_idx].strip()
    except ValueError:
        return None

async def write_section(ai_id: str, section_id: str, content: str, message: str = "AI contribution") -> bool:
    """
    Write content to a specific section in the document.
    
    Args:
        ai_id: Unique identifier for the AI agent
        section_id: ID of the section to write to
        content: Content to write
        message: Optional message describing the edit
        
    Returns:
        True if the edit was queued successfully
    """
    doc = gdoc_service(ai_id)
    try:
        await doc.queue_edit(section_id, content, message)
        return True
    except Exception as e:
        print(f"Error queueing edit: {e}")
        return False

# Example usage:
if __name__ == "__main__":
    import asyncio
    
    async def demo():
        # Read a section
        content = await read_section("test-ai", "TEST_SECTION")
        print(f"Current content: {content}")
        
        # Write to a section
        success = await write_section(
            "test-ai",
            "TEST_SECTION",
            "This is a test edit from the helper module.",
            "Testing the helper interface"
        )
        print(f"Edit queued: {success}")
    
    asyncio.run(demo()) 