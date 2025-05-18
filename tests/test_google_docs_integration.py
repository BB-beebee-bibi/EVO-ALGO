"""
Test script for Google Docs integration.
"""
import os
import pytest
import time
import json
from unittest.mock import Mock, patch
from googleapiclient.errors import HttpError
from trisolaris.integration.google_docs_sync import GoogleDocsSyncService

@pytest.fixture
def mock_docs_service():
    """Create a mock Google Docs service."""
    with patch('googleapiclient.discovery.build') as mock_build:
        mock_service = Mock()
        mock_build.return_value = mock_service
        yield mock_service

@pytest.fixture
def local_backup_path(tmp_path):
    """Create a temporary local backup file."""
    return str(tmp_path / "local_doc_backup.json")

@pytest.mark.skipif(not os.path.exists('credentials.json'), reason="No credentials.json found")
def test_google_docs_sync():
    """Test the GoogleDocsSyncService with real credentials."""
    # Use test doc ID or default
    doc_id = os.environ.get('GOOGLE_DOC_ID', '1fk0TkyC7xsKgw2yGV6lzTn99zU6u2SKuNU2RdYxMj9w')
    
    # Create service
    service = GoogleDocsSyncService(
        doc_id=doc_id,
        credentials_path='credentials.json',
        update_interval=0  # Disable auto-update
    )
    
    # Fetch document
    content = service.fetch_document()
    assert content is not None, "Failed to fetch document content"
    
    # Update document with test content
    test_content = f"Test content {time.time()}"
    success = service.update_document(f"\n\n# TEST SECTION\n\n{test_content}\n\n", 1)
    assert success, "Failed to update document"
    
    # Fetch again to verify
    updated_content = service.fetch_document(force=True)
    assert test_content in updated_content, "Updated content not found in document"

def test_local_fallback(mock_docs_service, local_backup_path):
    """Test local fallback when Google Docs is unavailable."""
    # Create service with mock
    service = GoogleDocsSyncService(
        doc_id="test_doc_id",
        credentials_path="test_credentials.json",
        update_interval=0,
        local_fallback_path=local_backup_path
    )
    
    # Save some test content
    test_content = "Test content for local fallback"
    with open(local_backup_path, 'w') as f:
        json.dump({
            'content': test_content,
            'timestamp': time.time()
        }, f)
    
    # Mock service to fail
    mock_docs_service.documents().get().execute.side_effect = HttpError(
        resp=Mock(status=500),
        content=b'Internal Server Error'
    )
    
    # Fetch should return local content
    content = service.fetch_document()
    assert content == test_content, "Failed to use local fallback"

def test_retry_mechanism(mock_docs_service):
    """Test retry mechanism for rate limits."""
    # Create service with mock
    service = GoogleDocsSyncService(
        doc_id="test_doc_id",
        credentials_path="test_credentials.json",
        update_interval=0
    )
    
    # Mock service to fail twice then succeed
    mock_docs_service.documents().get().execute.side_effect = [
        HttpError(resp=Mock(status=429), content=b'Rate Limit Exceeded'),
        HttpError(resp=Mock(status=429), content=b'Rate Limit Exceeded'),
        {'body': {'content': [{'paragraph': {'elements': [{'textRun': {'content': 'Success'}}]}}]}}
    ]
    
    # Fetch should eventually succeed
    content = service.fetch_document()
    assert content == "Success", "Failed to retry after rate limit"
    assert mock_docs_service.documents().get().execute.call_count == 3, "Wrong number of retries"

def test_subscriber_notification(mock_docs_service):
    """Test subscriber notification on content changes."""
    # Create service with mock
    service = GoogleDocsSyncService(
        doc_id="test_doc_id",
        credentials_path="test_credentials.json",
        update_interval=0
    )
    
    # Create mock subscriber
    subscriber = Mock()
    service.subscribe(subscriber)
    
    # Mock service to return different content
    mock_docs_service.documents().get().execute.return_value = {
        'body': {'content': [{'paragraph': {'elements': [{'textRun': {'content': 'New Content'}}]}}]}
    }
    
    # Fetch should notify subscriber
    service.fetch_document(force=True)
    subscriber.assert_called_once_with("New Content")

def test_error_handling(mock_docs_service):
    """Test error handling for various failure cases."""
    # Create service with mock
    service = GoogleDocsSyncService(
        doc_id="test_doc_id",
        credentials_path="test_credentials.json",
        update_interval=0
    )
    
    # Test invalid credentials
    mock_docs_service.documents().get().execute.side_effect = HttpError(
        resp=Mock(status=401),
        content=b'Invalid Credentials'
    )
    content = service.fetch_document()
    assert content == "", "Should return empty string on auth error"
    
    # Test document not found
    mock_docs_service.documents().get().execute.side_effect = HttpError(
        resp=Mock(status=404),
        content=b'Document Not Found'
    )
    content = service.fetch_document()
    assert content == "", "Should return empty string on not found error"
    
    # Test unexpected error
    mock_docs_service.documents().get().execute.side_effect = Exception("Unexpected error")
    content = service.fetch_document()
    assert content == "", "Should return empty string on unexpected error" 