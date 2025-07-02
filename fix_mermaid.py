#!/usr/bin/env python3
"""
Automated Mermaid Diagram Fixer
Fixes common issues that cause parsing errors like "PS" errors
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

class MermaidDiagramFixer:
    """
    Automatically fix common Mermaid diagram syntax errors
    """
    
    def __init__(self):
        # Comprehensive fix patterns
        self.fix_patterns = [
            # Replace HTML breaks with proper line breaks
            (r'<br\s*/?>\s*', '\\n'),
            
            # Remove problematic Shape: definitions
            (r'Shape:\s*\([^)]*\)', ''),
            
            # Fix problematic angle brackets in node labels
            (r'\[([^<>\]]*)<([^<>\]]*)\]', r'[\1&lt;\2]'),
            (r'\[([^<>\]]*)<([^<>\]]*)<([^<>\]]*)\]', r'[\1&lt;\2&lt;\3]'),
            (r'\[([^<>\]]*>)([^<>\]]*)\]', r'[\1&gt;\2]'),
            (r'\[([^<>\]]*>)([^<>\]]*>)([^<>\]]*)\]', r'[\1&gt;\2&gt;\3]'),
            
            # Fix specific problematic patterns
            (r'\[ADC Data<br/>Shape: \([^)]*\)\]', '[ADC Data]'),
            (r'\[([^]]*)<br/>([^]]*)\]', r'[\1\\n\2]'),
            (r'\[([^]]*)<([^]]*)\]', r'[\1&lt;\2]'),
            (r'\[([^]]*>)([^]]*)\]', r'[\1&gt;\2]'),
            
            # Clean up multiple line breaks
            (r'\\n\\n+', '\\n'),
            
            # Fix malformed node definitions
            (r'\[([^]]*)\s*<br/>\s*([^]]*)\]', r'[\1\\n\2]'),
        ]
        
        # Diagram type fixes
        self.diagram_type_fixes = {
            'xychart-beta': 'pie',  # Replace unsupported diagram types
            'requirement': 'graph TD',
            'c4Context': 'graph TB',
            'c4Container': 'graph TB',
            'c4Component': 'graph TB',
            'c4Dynamic': 'flowchart TD'
        }
    
    def fix_all_files(self, directory: str = ".") -> dict:
        """Fix all Mermaid diagrams in markdown files"""
        
        print(f"🔧 Fixing Mermaid diagrams in {directory}")
        
        results = []
        total_files = 0
        fixed_files = 0
        total_diagrams_fixed = 0
        
        # Find all markdown files
        md_files = list(Path(directory).rglob("*.md"))
        
        for md_file in md_files:
            total_files += 1
            file_result = self.fix_file(str(md_file))
            
            if file_result['diagrams_fixed'] > 0:
                fixed_files += 1
                total_diagrams_fixed += file_result['diagrams_fixed']
                results.append(file_result)
                
                print(f"  ✅ Fixed {file_result['diagrams_fixed']} diagrams in {md_file.name}")
        
        return {
            'total_files_processed': total_files,
            'files_with_fixes': fixed_files,
            'total_diagrams_fixed': total_diagrams_fixed,
            'file_results': results
        }
    
    def fix_file(self, file_path: str) -> dict:
        """Fix all Mermaid diagrams in a single file"""
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Find and fix all Mermaid diagrams
            fixed_content, diagrams_fixed = self.fix_mermaid_diagrams(original_content)
            
            # Only write if changes were made
            if diagrams_fixed > 0:
                # Create backup
                backup_path = file_path + '.backup'
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
            
            return {
                'file_path': file_path,
                'diagrams_fixed': diagrams_fixed,
                'success': True
            }
            
        except Exception as e:
            return {
                'file_path': file_path,
                'diagrams_fixed': 0,
                'success': False,
                'error': str(e)
            }
    
    def fix_mermaid_diagrams(self, content: str) -> Tuple[str, int]:
        """Find and fix all Mermaid diagrams in content"""
        
        # Pattern to match mermaid code blocks
        pattern = r'(```mermaid\s*\n)(.*?)(\n```)'
        
        diagrams_fixed = 0
        
        def fix_diagram_match(match):
            nonlocal diagrams_fixed
            
            start_marker = match.group(1)
            diagram_content = match.group(2)
            end_marker = match.group(3)
            
            # Fix the diagram content
            fixed_diagram = self.fix_single_diagram(diagram_content)
            
            # Check if any fixes were applied
            if fixed_diagram != diagram_content:
                diagrams_fixed += 1
            
            return start_marker + fixed_diagram + end_marker
        
        # Apply fixes to all diagrams
        fixed_content = re.sub(pattern, fix_diagram_match, content, flags=re.DOTALL)
        
        return fixed_content, diagrams_fixed
    
    def fix_single_diagram(self, diagram_content: str) -> str:
        """Fix a single Mermaid diagram"""
        
        fixed_content = diagram_content
        
        # Apply all fix patterns
        for pattern, replacement in self.fix_patterns:
            fixed_content = re.sub(pattern, replacement, fixed_content, flags=re.IGNORECASE)
        
        # Fix diagram types
        lines = fixed_content.split('\n')
        if lines:
            first_line = lines[0].strip()
            for old_type, new_type in self.diagram_type_fixes.items():
                if first_line.startswith(old_type):
                    lines[0] = lines[0].replace(old_type, new_type)
                    break
        
        fixed_content = '\n'.join(lines)
        
        # Additional specific fixes
        fixed_content = self.apply_specific_fixes(fixed_content)
        
        return fixed_content
    
    def apply_specific_fixes(self, content: str) -> str:
        """Apply specific fixes for known problematic patterns"""
        
        # Fix specific problematic node labels
        specific_fixes = [
            # Fix ADC Data pattern
            (r'\[ADC Data<br/>Shape: \(samples, chirps, rx_antennas\)\]', '[ADC Data\\nShape: samples x chirps x rx_antennas]'),
            
            # Fix any remaining HTML in brackets
            (r'\[([^]]*)<([^>]*)>([^]]*)\]', r'[\1&lt;\2&gt;\3]'),
            
            # Fix shape definitions in node labels
            (r'\[([^]]*)\s*Shape:\s*\([^)]*\)([^]]*)\]', r'[\1\2]'),
            
            # Clean up excessive whitespace
            (r'\s*\\n\s*', '\\n'),
            
            # Fix timeline syntax if present
            (r'timeline\s*\n\s*title\s*([^\n]+)\s*\n', r'timeline\n    title \1\n'),
        ]
        
        for pattern, replacement in specific_fixes:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        return content
    
    def validate_fixes(self, directory: str = ".") -> dict:
        """Validate that fixes were successful"""
        
        from validate_mermaid import MermaidDiagramValidator
        
        validator = MermaidDiagramValidator()
        results = validator.validate_all_files(directory)
        
        return {
            'validation_passed': results['invalid_diagrams'] == 0,
            'remaining_issues': results['invalid_diagrams'],
            'success_rate': results['success_rate']
        }

def main():
    """Main function to run the fixer"""
    
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "docs/"
    
    print("🚀 Starting Mermaid Diagram Auto-Fixer")
    print("=" * 50)
    
    # Fix all diagrams
    fixer = MermaidDiagramFixer()
    fix_results = fixer.fix_all_files(directory)
    
    # Print results
    print(f"\n📊 Fix Results:")
    print(f"  📁 Total files processed: {fix_results['total_files_processed']}")
    print(f"  🔧 Files with fixes: {fix_results['files_with_fixes']}")
    print(f"  📊 Total diagrams fixed: {fix_results['total_diagrams_fixed']}")
    
    if fix_results['total_diagrams_fixed'] > 0:
        print(f"\n✅ Successfully fixed {fix_results['total_diagrams_fixed']} diagrams!")
        
        # Validate the fixes
        print(f"\n🔍 Validating fixes...")
        validation_results = fixer.validate_fixes(directory)
        
        if validation_results['validation_passed']:
            print(f"✅ All diagrams now pass validation!")
            print(f"📈 Success rate: {validation_results['success_rate']:.1f}%")
        else:
            print(f"⚠️  {validation_results['remaining_issues']} diagrams still have issues")
            print(f"📈 Success rate: {validation_results['success_rate']:.1f}%")
            print(f"💡 Manual review may be needed for remaining issues")
    
    else:
        print(f"\nℹ️  No diagrams needed fixing!")
    
    print(f"\n🎉 Phase 1 diagram fixing complete!")

if __name__ == "__main__":
    main()
