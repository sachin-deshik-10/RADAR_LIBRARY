#!/usr/bin/env python3
"""
Debug specific diagrams to find issues
"""
import re
import os

def find_mermaid_diagrams(file_path):
    """Find all mermaid diagrams in a file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all mermaid code blocks
    pattern = r'```mermaid\n(.*?)\n```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    return matches

def debug_specific_diagrams():
    """Debug specific problematic diagrams"""
    problematic_files = {
        'docs/advanced_technologies.md': [12],
        'docs/datasets_and_benchmarks.md': [16, 17],
        'docs/literature_review.md': [14, 21, 23],
        'docs/research_gaps_and_architectures.md': [11]
    }
    
    for file_path, diagram_nums in problematic_files.items():
        if os.path.exists(file_path):
            print(f"\n🔍 File: {file_path}")
            diagrams = find_mermaid_diagrams(file_path)
            
            for num in diagram_nums:
                if num - 1 < len(diagrams):
                    print(f"\n📊 Diagram {num}:")
                    print("=" * 50)
                    diagram_content = diagrams[num - 1]
                    print(diagram_content[:200] + "..." if len(diagram_content) > 200 else diagram_content)
                    print("=" * 50)
                else:
                    print(f"\n❌ Diagram {num} not found (only {len(diagrams)} diagrams in file)")

if __name__ == "__main__":
    debug_specific_diagrams()
