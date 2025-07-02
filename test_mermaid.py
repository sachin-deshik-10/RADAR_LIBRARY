#!/usr/bin/env python3
"""
Simple test to check if our Mermaid diagrams are valid
"""
import re
import os

def test_basic_mermaid_syntax(file_path):
    """Test basic Mermaid syntax validation"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all mermaid code blocks
    pattern = r'```mermaid\n(.*?)\n```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    print(f"\n🔍 Testing: {file_path}")
    print(f"📊 Found {len(matches)} diagrams")
    
    for i, diagram in enumerate(matches, 1):
        lines = diagram.strip().split('\n')
        first_line = lines[0].strip() if lines else ""
        
        # Check for common Mermaid diagram types
        valid_types = [
            'graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 
            'stateDiagram', 'pie', 'timeline', 'mindmap', 'quadrantChart',
            'gitgraph', 'journey', 'sankey', 'xychart', 'block'
        ]
        
        # Check for init directives
        if first_line.startswith('%%{init:'):
            print(f"  📊 Diagram {i}: INIT directive found")
            # Get the actual diagram type from the second line
            if len(lines) > 1:
                first_line = lines[1].strip()
        
        # Check if it starts with a valid type
        is_valid = any(first_line.startswith(vtype) for vtype in valid_types)
        
        if is_valid:
            print(f"  ✅ Diagram {i}: {first_line}")
        else:
            print(f"  ❌ Diagram {i}: Invalid type: {first_line}")
            print(f"     First few lines: {lines[:3]}")

# Test the files that were reported as having issues
test_files = [
    'docs/advanced_technologies.md',
    'docs/datasets_and_benchmarks.md', 
    'docs/literature_review.md',
    'docs/research_gaps_and_architectures.md'
]

for file_path in test_files:
    if os.path.exists(file_path):
        test_basic_mermaid_syntax(file_path)
