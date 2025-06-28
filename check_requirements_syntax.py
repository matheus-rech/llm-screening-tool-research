#!/usr/bin/env python3

def check_requirements_syntax():
    """Check requirements.txt for syntax issues that might prevent proper installation."""
    try:
        with open('requirements.txt', 'r') as f:
            lines = f.readlines()
        
        print("Checking requirements.txt syntax...")
        issues_found = False
        
        for i, line in enumerate(lines, 1):
            original_line = line
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
                
            if '=' in line or '>' in line or '<' in line:
                package_name = line.split('=')[0].split('>')[0].split('<')[0].split('!')[0].strip()
                
                if not package_name:
                    print(f"Line {i}: Invalid format - empty package name: '{original_line.strip()}'")
                    issues_found = True
                    continue
                    
                if any(char in package_name for char in [' ', '\t']):
                    print(f"Line {i}: Package name contains whitespace: '{package_name}'")
                    issues_found = True
                    
                print(f"Line {i}: OK - {package_name}")
            else:
                print(f"Line {i}: No version specifier: '{line}'")
        
        psycopg2_found = False
        for line in lines:
            if 'psycopg2' in line.lower() and not line.strip().startswith('#'):
                psycopg2_found = True
                print(f"Found PostgreSQL driver: {line.strip()}")
                break
        
        if not psycopg2_found:
            print("ERROR: No psycopg2 or psycopg2-binary found in requirements.txt!")
            issues_found = True
        
        if not issues_found:
            print("✅ Requirements.txt syntax check passed")
            return True
        else:
            print("❌ Issues found in requirements.txt")
            return False
            
    except Exception as e:
        print(f"Error checking requirements.txt: {e}")
        return False

if __name__ == '__main__':
    success = check_requirements_syntax()
    exit(0 if success else 1)
