"""
Module for synchronizing with Google Docs as a centralized development registry.
"""
import json
import os
import time
import threading
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import backoff  # For retry mechanism

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('google_docs_sync')

@dataclass
class EditRequest:
    """Represents a document edit request."""
    section_id: str
    content: str
    message: str
    timestamp: str
    requestor: str
    version: Optional[str] = None

class GoogleDocsSyncService:
    """Service to sync with Google Docs as a centralized registry."""
    
    def __init__(self, 
                 doc_id: str,
                 user_id: str,
                 user_type: str = 'AI',
                 credentials_path: str = "credentials.json",
                 update_interval: int = 300,
                 local_fallback_path: str = "local_doc_backup.json"):
        """
        Initialize the sync service.
        
        Args:
            doc_id: Google Doc ID
            user_id: Unique identifier for this user/agent
            user_type: Type of user (e.g., 'AI', 'Human', 'System')
            credentials_path: Path to service account credentials
            update_interval: Seconds between auto-updates (0 to disable)
            local_fallback_path: Path to local backup file
        """
        self.doc_id = doc_id
        self.user_id = user_id
        self.user_type = user_type
        self.credentials_path = credentials_path
        self.update_interval = update_interval
        self.local_fallback_path = local_fallback_path
        
        # Cache of document content and metadata
        self.document_content = None
        self.last_update = 0
        self.last_sync_version = None
        
        # Edit queue and processing state
        self.edit_queue: List[EditRequest] = []
        self.is_processing_edits = False
        
        # Subscribers to changes
        self.subscribers = []
        
        # Initialize API client
        self.docs_service = None
        self._initialize_service()
        
        # Load local backup if exists
        self._load_local_backup()
        
        # Start auto-update thread if enabled
        if update_interval > 0:
            self.update_thread = threading.Thread(target=self._auto_update_loop, daemon=True)
            self.update_thread.start()
        else:
            self.update_thread = None
            
    def _initialize_service(self) -> None:
        """Initialize the Google Docs API client."""
        try:
            if not os.path.exists(self.credentials_path):
                logger.error(f"Credentials file not found: {self.credentials_path}")
                return
                
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path, scopes=['https://www.googleapis.com/auth/documents']
            )
            
            self.docs_service = build('docs', 'v1', credentials=credentials)
            logger.info("Google Docs API client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Docs API client: {e}")
            self.docs_service = None
            
    def _load_local_backup(self) -> None:
        """Load document content from local backup if available."""
        try:
            if os.path.exists(self.local_fallback_path):
                with open(self.local_fallback_path, 'r') as f:
                    data = json.load(f)
                    self.document_content = data.get('content', '')
                    self.last_update = data.get('timestamp', 0)
                    self.last_sync_version = data.get('version', None)
                    logger.info("Loaded document content from local backup")
        except Exception as e:
            logger.error(f"Failed to load local backup: {e}")
            
    def _save_local_backup(self) -> None:
        """Save document content to local backup."""
        try:
            with open(self.local_fallback_path, 'w') as f:
                json.dump({
                    'content': self.document_content,
                    'timestamp': time.time(),
                    'version': self.last_sync_version
                }, f)
            logger.info("Saved document content to local backup")
        except Exception as e:
            logger.error(f"Failed to save local backup: {e}")
    
    @backoff.on_exception(backoff.expo, HttpError, max_tries=3)
    def _fetch_document_remote(self) -> Optional[str]:
        """Fetch document content from Google Docs with retry."""
        if not self.docs_service:
            return None
            
        try:
            document = self.docs_service.documents().get(documentId=self.doc_id).execute()
            
            content = ""
            for element in document.get('body', {}).get('content', []):
                if 'paragraph' in element:
                    for paragraph_element in element['paragraph']['elements']:
                        if 'textRun' in paragraph_element:
                            content += paragraph_element['textRun']['content']
                            
            return content
            
        except HttpError as e:
            if e.resp.status == 429:  # Rate limit
                logger.warning("Rate limit exceeded, will retry with backoff")
                raise  # Let backoff handle retry
            logger.error(f"Failed to fetch document: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching document: {e}")
            return None
    
    def _auto_update_loop(self) -> None:
        """Background thread for automatic updates."""
        while True:
            try:
                self.fetch_document()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in auto-update loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def fetch_document(self, force: bool = False) -> str:
        """
        Fetch the document content from Google Docs.
        
        Args:
            force: Force refresh even if recently updated
            
        Returns:
            Document content as plain text
        """
        current_time = time.time()
        
        # Check if we need to refresh
        if not force and self.document_content is not None and \
           current_time - self.last_update < self.update_interval:
            return self.document_content
            
        # Try to fetch from Google Docs
        content = self._fetch_document_remote()
        
        if content is not None:
            # Update cache and notify subscribers
            old_content = self.document_content
            self.document_content = content
            self.last_update = current_time
            self.last_sync_version = datetime.utcnow().isoformat()
            self._save_local_backup()
            
            if old_content != content:
                self._notify_subscribers({
                    'type': 'CONTENT_UPDATED',
                    'content': content,
                    'version': self.last_sync_version
                })
                
            return content
        else:
            # Return cached content or empty string
            return self.document_content or ""
    
    @backoff.on_exception(backoff.expo, HttpError, max_tries=3)
    def update_document(self, content: str, position: int = 1) -> bool:
        """
        Update the document content.
        
        Args:
            content: New content to write
            position: Position to insert (1 for start of document)
            
        Returns:
            Success status
        """
        if not self.docs_service:
            logger.error("Google Docs API client not initialized")
            return False
            
        try:
            # Create the update request
            requests = [
                {
                    'insertText': {
                        'location': {
                            'index': position
                        },
                        'text': content
                    }
                }
            ]
            
            # Execute the update
            result = self.docs_service.documents().batchUpdate(
                documentId=self.doc_id,
                body={'requests': requests}
            ).execute()
            
            # Force refresh content
            self.fetch_document(force=True)
            
            return True
            
        except HttpError as e:
            if e.resp.status == 429:  # Rate limit
                logger.warning("Rate limit exceeded, will retry with backoff")
                raise  # Let backoff handle retry
            logger.error(f"Failed to update document: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error updating document: {e}")
            return False
    
    def subscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe to document events.
        
        Args:
            callback: Function to call with event data
        """
        self.subscribers.append(callback)
        
        # Call immediately with current content
        if self.document_content:
            callback({
                'type': 'INITIAL_CONTENT',
                'content': self.document_content,
                'version': self.last_sync_version
            })
    
    def _format_edit(self, edit_request: EditRequest) -> str:
        """Format an edit with proper markers and metadata."""
        timestamp = datetime.fromisoformat(edit_request.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        return f"\n{edit_request.content}\n[{self.user_type} Edit: {timestamp}] - {edit_request.message}\n"

    async def queue_edit(self, section_id: str, content: str, message: str) -> EditRequest:
        """
        Queue an edit to be applied to the document.
        
        Args:
            section_id: Identifier for the section to edit
            content: New content to write
            message: Description of the edit
            
        Returns:
            The created EditRequest
        """
        edit_request = EditRequest(
            section_id=section_id,
            content=content,
            message=message,
            timestamp=datetime.utcnow().isoformat(),
            requestor=f"{self.user_type}:{self.user_id}",
            version=self.last_sync_version
        )
        
        self.edit_queue.append(edit_request)
        logger.info(f"Queued edit for section {section_id} from {edit_request.requestor}")
        
        if not self.is_processing_edits:
            await self._process_edit_queue()
            
        return edit_request

    async def _process_edit_queue(self) -> None:
        """Process the edit queue with rate limiting and conflict resolution."""
        if not self.edit_queue:
            self.is_processing_edits = False
            return
            
        self.is_processing_edits = True
        edit_request = self.edit_queue.pop(0)
        
        try:
            # Always refresh before editing to avoid conflicts
            self.fetch_document(force=True)
            
            # Apply edit with proper formatting
            formatted_edit = self._format_edit(edit_request)
            success = self.update_document(
                formatted_edit,
                self._find_section_position(edit_request.section_id)
            )
            
            if success:
                self._notify_subscribers({
                    'type': 'EDIT_APPLIED',
                    'edit': edit_request
                })
            else:
                # Requeue the edit if it failed
                self.edit_queue.insert(0, edit_request)
                
        except Exception as e:
            logger.error(f"Error processing edit: {e}")
            # Requeue the edit on error
            self.edit_queue.insert(0, edit_request)
            
        # Continue processing queue with rate limiting
        time.sleep(2)  # Rate limiting
        await self._process_edit_queue()

    def _find_section_position(self, section_id: str) -> int:
        """
        Find the position of a section in the document.
        
        Args:
            section_id: The section identifier to find
            
        Returns:
            The index where the section starts
        """
        if not self.document_content:
            return 1
            
        # Look for section marker
        marker = f"#§ {section_id}:"
        pos = self.document_content.find(marker)
        return pos + 1 if pos >= 0 else 1

    def _notify_subscribers(self, event: Dict[str, Any]) -> None:
        """Notify all subscribers of an event."""
        for callback in self.subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}") 