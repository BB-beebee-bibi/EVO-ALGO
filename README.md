# TRISOLARIS Document Collaboration System

This system enables AI agents (including Claude) to collaborate on a shared Google Doc through a secure broker service.

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   ```bash
   # For the broker service
   export TRI_SHARED_SECRET="your-secure-secret-key"  # Change this!
   export TRISOLARIS_GDOC_KEY="/path/to/your/credentials.json"
   
   # For Claude's helper
   export TRI_BROKER_URL="http://localhost:5000"  # Or your deployed broker URL
   ```

3. **Start the Broker Service**
   ```bash
   python broker_service.py
   ```

4. **Test the Integration**
   ```bash
   python claude_helper.py
   ```

## Security Notes

1. The `TRI_SHARED_SECRET` must be:
   - At least 32 characters long
   - Kept secure and not shared publicly
   - The same value for both the broker and Claude's helper

2. The broker service should be:
   - Deployed to a secure, HTTPS-enabled endpoint
   - Protected by proper firewall rules
   - Monitored for unusual activity

## Usage

### Reading a Section
```python
from claude_helper import read_section

result = read_section("SECTION_ID")
if result:
    print(result["content"])
```

### Writing to a Section
```python
from claude_helper import write_section

result = write_section(
    "SECTION_ID",
    "Your content here",
    "Description of the edit"
)
if result:
    print(f"Edit status: {result['status']}")
```

## Troubleshooting

1. **Connection Errors**
   - Verify the broker service is running
   - Check the `TRI_BROKER_URL` is correct
   - Ensure network connectivity

2. **Authentication Errors**
   - Verify `TRI_SHARED_SECRET` matches between broker and helper
   - Check Google credentials are valid
   - Ensure proper permissions on the Google Doc

3. **Section Not Found**
   - Verify section IDs are correctly formatted
   - Check the section exists in the document
   - Ensure proper section markers are present

## Support

For issues or questions, please contact the TRISOLARIS team.
