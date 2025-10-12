#!/usr/bin/env python3
"""Fix tracking calls in sponsor_integrations.py"""

import re

# Read the file
with open('agents/sponsor_integrations.py', 'r') as f:
    content = f.read()

# Pattern to fix nested dict structure in track_event calls
# This will find patterns like track_event("name": { ... }) and fix them

# Fix pattern 1: track_event("event", { "actual_event": { ... } })
content = re.sub(
    r'track_event\("event", \{\s*"(\w+)":\s*\{([^}]+)\}\s*\}\)',
    r'track_event("\1", {\2})',
    content,
    flags=re.MULTILINE | re.DOTALL
)

# Fix pattern 2: track_event("name": { ... }) - missing comma
content = re.sub(
    r'track_event\("(\w+)":\s*\{',
    r'track_event("\1", {',
    content
)

# Fix pattern 3: track_event("copilotkit_suggestions": suggestions })
content = re.sub(
    r'track_event\("(\w+)":\s*(\w+)\s*\}\)',
    r'track_event("\1", \2)',
    content
)

# Fix pattern 4: track_event("copilotkit_insights": insights })
content = re.sub(
    r'track_event\("(\w+)":\s*(\w+)\s*\}\)',
    r'track_event("\1", \2)',
    content
)

# Write back
with open('agents/sponsor_integrations.py', 'w') as f:
    f.write(content)

print("Fixed tracking calls in sponsor_integrations.py")