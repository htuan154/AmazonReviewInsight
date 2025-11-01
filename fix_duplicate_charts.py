"""
Fix duplicate Visual Analysis charts in HTML reports
Removes the second occurrence of chart images that were added by add_image_links.py
"""
import re

def fix_day3_report():
    """Fix day3_v2_report.html - remove duplicate chart section"""
    filepath = 'docs_v2/day3_v2_report.html'
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match the duplicate section starting from line 3083
    # From "<h3> Visual Analysis Charts</h3>" to just before "<h2>🎯 Final Recommendation"
    pattern = r'(\s*<h3>\s*📊\s*Visual Analysis Charts</h3>\s*<div class="chart-container">.*?</div>\s*</div>\s*)(?=\s*<h2>🎯 Final Recommendation)'
    
    # Count matches
    matches = list(re.finditer(pattern, content, re.DOTALL))
    print(f"Found {len(matches)} duplicate chart sections in day3_v2_report.html")
    
    if matches:
        # Remove the duplicate section
        content_fixed = re.sub(pattern, '', content, flags=re.DOTALL)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content_fixed)
        
        print(f"✓ Removed duplicate chart section from day3_v2_report.html")
        print(f"  Original size: {len(content):,} bytes")
        print(f"  Fixed size: {len(content_fixed):,} bytes")
        print(f"  Removed: {len(content) - len(content_fixed):,} bytes")
    else:
        print("No duplicate section found in day3_v2_report.html")

def fix_all_in_one():
    """Fix v2_all_in_one.html - remove duplicate chart section"""
    filepath = 'docs_v2/v2_all_in_one.html'
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Similar pattern for v2_all_in_one.html
    pattern = r'(\s*<h3>\s*📊\s*Visual Analysis Charts</h3>\s*<div class="chart-container">.*?</div>\s*</div>\s*)(?=\s*<h2>🎯 Final Recommendation)'
    
    matches = list(re.finditer(pattern, content, re.DOTALL))
    print(f"\nFound {len(matches)} duplicate chart sections in v2_all_in_one.html")
    
    if matches:
        content_fixed = re.sub(pattern, '', content, flags=re.DOTALL)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content_fixed)
        
        print(f"✓ Removed duplicate chart section from v2_all_in_one.html")
        print(f"  Original size: {len(content):,} bytes")
        print(f"  Fixed size: {len(content_fixed):,} bytes")
        print(f"  Removed: {len(content) - len(content_fixed):,} bytes")
    else:
        print("No duplicate section found in v2_all_in_one.html")

if __name__ == '__main__':
    print("=" * 60)
    print("Fixing Duplicate Visual Analysis Charts")
    print("=" * 60)
    
    fix_day3_report()
    fix_all_in_one()
    
    print("\n" + "=" * 60)
    print("✅ Fix completed! Both HTML files cleaned.")
    print("=" * 60)
