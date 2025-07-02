#!/usr/bin/env python3
"""
Mermaid Diagram Validation and Error Prevention Script
Comprehensive validation for Phase 1 implementation
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Data class for validation results"""
    file_path: str
    diagram_index: int
    is_valid: bool
    issues: List[str]
    content_preview: str
    suggested_fixes: List[str]

class MermaidDiagramValidator:
    """
    Advanced Mermaid diagram validator with comprehensive error detection
    """
    
    def __init__(self):
        # Define problematic patterns that cause "PS" parsing errors
        self.problematic_patterns = {
            'html_breaks': r'<br\s*/?>\s*',
            'shape_definitions': r'Shape:\s*\([^)]*\)',
            'unescaped_angles': r'(?<![-=~])[<>](?![a-zA-Z\-\[\]()])',
            'malformed_nodes': r'[A-Za-z0-9_]+\[[^\]]*<[^\]]*\]',
            'invalid_characters': r'[^\w\s\-\[\](){}.,;:!?+=*/_#&|~<>"\']',
            'unbalanced_brackets': r'\[[^\]]*\n[^\]]*\]',
        }
        
        # Safe replacement patterns
        self.safe_replacements = {
            r'<br\s*/?>\s*': '\\n',
            r'Shape:\s*\([^)]*\)': '',
            r'(?<![-=~])<(?![a-zA-Z\-\[\](])': '&lt;',
            r'(?<![-=~])>(?![a-zA-Z\-\[\](])': '&gt;',
        }
        
        # Valid Mermaid diagram types
        self.valid_diagram_types = {
            'graph', 'flowchart', 'sequenceDiagram', 'classDiagram',
            'stateDiagram', 'erDiagram', 'journey', 'gantt',
            'pie', 'gitgraph', 'mindmap', 'timeline'
        }
        
        # Valid node shapes
        self.valid_node_shapes = {
            '[]', '()', '([])', '[[]]', '[()]', '(())',
            '>', '{', '}', '{{}}', '[//]', '[\\]',
            '[(', ')]', '([)]'
        }
    
    def validate_all_files(self, directory: str = ".") -> Dict:
        """Validate all Mermaid diagrams in markdown files"""
        
        print(f"🔍 Scanning for Mermaid diagrams in {directory}")
        
        results = []
        total_diagrams = 0
        valid_diagrams = 0
        
        # Find all markdown files
        md_files = list(Path(directory).rglob("*.md"))
        
        for md_file in md_files:
            file_results = self.validate_file(str(md_file))
            if file_results['diagrams']:
                results.append(file_results)
                total_diagrams += file_results['total_diagrams']
                valid_diagrams += file_results['valid_diagrams']
        
        success_rate = (valid_diagrams / total_diagrams * 100) if total_diagrams > 0 else 100
        
        return {
            'total_files': len(md_files),
            'files_with_diagrams': len(results),
            'total_diagrams': total_diagrams,
            'valid_diagrams': valid_diagrams,
            'invalid_diagrams': total_diagrams - valid_diagrams,
            'success_rate': success_rate,
            'file_results': results
        }
    
    def validate_file(self, file_path: str) -> Dict:
        """Validate all Mermaid diagrams in a single file"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            diagrams = self.extract_mermaid_diagrams(content)
            
            if not diagrams:
                return {
                    'file_path': file_path,
                    'diagrams': [],
                    'total_diagrams': 0,
                    'valid_diagrams': 0
                }
            
            validation_results = []
            valid_count = 0
            
            for i, diagram in enumerate(diagrams):
                result = self.validate_single_diagram(diagram, i)
                validation_results.append(result)
                if result.is_valid:
                    valid_count += 1
            
            return {
                'file_path': file_path,
                'diagrams': validation_results,
                'total_diagrams': len(diagrams),
                'valid_diagrams': valid_count
            }
            
        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e),
                'diagrams': [],
                'total_diagrams': 0,
                'valid_diagrams': 0
            }
    
    def extract_mermaid_diagrams(self, content: str) -> List[str]:
        """Extract Mermaid diagram blocks from markdown"""
        
        # Pattern to match mermaid code blocks
        pattern = r'```mermaid\s*\n(.*?)\n```'
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        
        return [match.strip() for match in matches]
    
    def validate_single_diagram(self, content: str, index: int) -> ValidationResult:
        """Validate a single Mermaid diagram"""
        
        issues = []
        
        # Check diagram type
        if not self.has_valid_diagram_type(content):
            issues.append("Invalid or missing diagram type")
        
        # Check for problematic patterns
        for pattern_name, pattern in self.problematic_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f"Found {pattern_name.replace('_', ' ')}")
        
        # Check for balanced brackets
        if not self.check_balanced_brackets(content):
            issues.append("Unbalanced brackets detected")
        
        # Check node definitions
        node_issues = self.validate_nodes(content)
        issues.extend(node_issues)
        
        # Check arrow syntax
        arrow_issues = self.validate_arrows(content)
        issues.extend(arrow_issues)
        
        # Generate fixes
        suggested_fixes = self.generate_fixes(content, issues)
        
        return ValidationResult(
            file_path="",
            diagram_index=index,
            is_valid=len(issues) == 0,
            issues=issues,
            content_preview=content[:100].replace('\n', ' '),
            suggested_fixes=suggested_fixes
        )
    
    def has_valid_diagram_type(self, content: str) -> bool:
        """Check if diagram has a valid type declaration"""
        
        first_line = content.split('\n')[0].strip().lower()
        
        # Check for explicit diagram types
        for diagram_type in self.valid_diagram_types:
            if first_line.startswith(diagram_type.lower()):
                return True
        
        return False
    
    def check_balanced_brackets(self, content: str) -> bool:
        """Check for balanced brackets and parentheses"""
        
        bracket_pairs = {'[': ']', '(': ')', '{': '}'}
        stack = []
        
        for char in content:
            if char in bracket_pairs:
                stack.append(bracket_pairs[char])
            elif char in bracket_pairs.values():
                if not stack or stack.pop() != char:
                    return False
        
        return len(stack) == 0
    
    def validate_nodes(self, content: str) -> List[str]:
        """Validate node definitions"""
        
        issues = []
        
        # Find node definitions
        node_pattern = r'([A-Za-z][A-Za-z0-9_]*)\s*(\[[^\]]*\]|\([^)]*\)|\{[^}]*\})'
        matches = re.findall(node_pattern, content)
        
        for node_id, node_shape in matches:
            # Check node ID format
            if not re.match(r'^[A-Za-z][A-Za-z0-9_]*$', node_id):
                issues.append(f"Invalid node ID format: {node_id}")
            
            # Check for problematic content in node labels
            if '<' in node_shape or '>' in node_shape:
                if not self.is_valid_html_in_node(node_shape):
                    issues.append(f"Problematic characters in node {node_id}")
        
        return issues
    
    def validate_arrows(self, content: str) -> List[str]:
        """Validate arrow syntax"""
        
        issues = []
        
        # Valid arrow patterns
        valid_arrows = [
            '-->', '--->', '-.', '-.->',
            '==>', '===>', '~~>', '~~~>',
            '--', '---', '==', '===',
            '~~', '~~~', '.-'
        ]
        
        # Find potential arrows
        arrow_pattern = r'[-=~.]{2,}>?'
        arrows = re.findall(arrow_pattern, content)
        
        for arrow in set(arrows):  # Remove duplicates
            if arrow not in valid_arrows and len(arrow) >= 2:
                issues.append(f"Invalid arrow syntax: {arrow}")
        
        return issues
    
    def is_valid_html_in_node(self, node_content: str) -> bool:
        """Check if HTML content in node is valid for Mermaid"""
        
        # Allow some basic HTML entities
        allowed_html = ['&lt;', '&gt;', '&amp;', '&quot;', '&#39;']
        
        # Remove allowed HTML entities
        cleaned = node_content
        for html_entity in allowed_html:
            cleaned = cleaned.replace(html_entity, '')
        
        # Check if any angle brackets remain
        return '<' not in cleaned and '>' not in cleaned
    
    def generate_fixes(self, content: str, issues: List[str]) -> List[str]:
        """Generate suggested fixes for issues"""
        
        fixes = []
        
        if any("html breaks" in issue.lower() for issue in issues):
            fixes.append("Replace <br> tags with \\n")
        
        if any("shape definitions" in issue.lower() for issue in issues):
            fixes.append("Remove Shape: definitions from labels")
        
        if any("unescaped angles" in issue.lower() for issue in issues):
            fixes.append("Replace < with &lt; and > with &gt;")
        
        if any("balanced brackets" in issue.lower() for issue in issues):
            fixes.append("Check all brackets are properly closed")
        
        if any("node id" in issue.lower() for issue in issues):
            fixes.append("Use alphanumeric characters for node IDs")
        
        if any("arrow syntax" in issue.lower() for issue in issues):
            fixes.append("Use standard Mermaid arrow syntax (-->, ---, etc.)")
        
        return fixes
    
    def auto_fix_diagram(self, content: str) -> str:
        """Automatically fix common issues"""
        
        fixed_content = content
        
        # Apply safe replacements
        for pattern, replacement in self.safe_replacements.items():
            fixed_content = re.sub(pattern, replacement, fixed_content, flags=re.IGNORECASE)
        
        return fixed_content
    
    def print_validation_report(self, results: Dict):
        """Print a comprehensive validation report"""
        
        print("\n" + "="*60)
        print("📊 MERMAID DIAGRAM VALIDATION REPORT")
        print("="*60)
        
        print(f"\n📁 Files scanned: {results['total_files']}")
        print(f"📄 Files with diagrams: {results['files_with_diagrams']}")
        print(f"📊 Total diagrams: {results['total_diagrams']}")
        print(f"✅ Valid diagrams: {results['valid_diagrams']}")
        print(f"❌ Invalid diagrams: {results['invalid_diagrams']}")
        print(f"📈 Success rate: {results['success_rate']:.1f}%")
        
        if results['invalid_diagrams'] > 0:
            print(f"\n🔧 ISSUES FOUND:")
            
            for file_result in results['file_results']:
                file_path = file_result['file_path']
                invalid_diagrams = [d for d in file_result['diagrams'] if not d.is_valid]
                
                if invalid_diagrams:
                    print(f"\n📄 {file_path}")
                    
                    for diagram in invalid_diagrams:
                        print(f"  📊 Diagram {diagram.diagram_index + 1}:")
                        for issue in diagram.issues:
                            print(f"    ❌ {issue}")
                        
                        if diagram.suggested_fixes:
                            print(f"    🔧 Suggested fixes:")
                            for fix in diagram.suggested_fixes:
                                print(f"      - {fix}")
        
        else:
            print(f"\n🎉 All diagrams are valid! No issues found.")

def main():
    """Main function to run validation"""
    
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "."
    
    validator = MermaidDiagramValidator()
    results = validator.validate_all_files(directory)
    validator.print_validation_report(results)
    
    # Exit with error code if any diagrams are invalid
    if results['invalid_diagrams'] > 0:
        print(f"\n❌ Validation failed. Fix {results['invalid_diagrams']} diagram(s) before proceeding.")
        sys.exit(1)
    else:
        print(f"\n✅ All validations passed! Ready for deployment.")
        sys.exit(0)

if __name__ == "__main__":
    main()
