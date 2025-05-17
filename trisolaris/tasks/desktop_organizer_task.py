#!/usr/bin/env python3
"""
Task for evolving code that organizes desktop files by content type.
"""
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from trisolaris.tasks.base import TaskInterface
# from trisolaris.core import CodeGenome  # Unused import removed


class DesktopOrganizerTask(TaskInterface):
    """Task for evolving code that organizes desktop files by content type."""
    
    def __init__(self, template_path=None):
        super().__init__()
        self.name = "desktop_organizer"
        self.description = "Evolve code to organize desktop files by content type"
        self.template_path = template_path
        
    def get_template(self) -> str:
        """Return a template for the desktop organizer code."""
        template = '''#!/usr/bin/env python3
"""
Desktop File Organizer
Organizes files on the desktop by their content type.
"""
import os
import shutil
from pathlib import Path
from typing import List, Dict

# Define file categories by extension
CATEGORIES = {
    'images': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'},
    'documents': {'.pdf', '.doc', '.docx', '.txt', '.rtf'},
    'audio': {'.mp3', '.wav', '.ogg', '.midi', '.m4a'},
    'video': {'.mp4', '.mov', '.avi', '.mkv', '.wmv'},
    'code': {'.py', '.java', '.cpp', '.c', '.js', '.html', '.css'},
    'archives': {'.zip', '.rar', '.7z', '.tar', '.gz'},
    'spreadsheets': {'.xls', '.xlsx', '.csv'},
    'presentations': {'.ppt', '.pptx', '.key'}
}

def get_file_type(file_path: str) -> str:
    """Determine the type of file based on its extension."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    for category, extensions in CATEGORIES.items():
        if ext in extensions:
            return category
    return 'unknown'

def create_category_dirs(base_path: str) -> None:
    """Create directories for each category."""
    for category in CATEGORIES.keys():
        category_path = os.path.join(base_path, category)
        os.makedirs(category_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, 'unknown'), exist_ok=True)

def organize_files(desktop_path: str) -> Dict[str, List[str]]:
    """Organize files into categories."""
    organized_files = {category: [] for category in CATEGORIES.keys()}
    organized_files['unknown'] = []
    
    create_category_dirs(desktop_path)
    
    for filename in os.listdir(desktop_path):
        file_path = os.path.join(desktop_path, filename)
        
        if os.path.isdir(file_path):
            continue
            
        category = get_file_type(file_path)
        target_dir = os.path.join(desktop_path, category)
        target_path = os.path.join(target_dir, filename)
        
        shutil.move(file_path, target_path)
        organized_files[category].append(target_path)
    
    return organized_files

def main():
    """Main function to organize desktop files."""
    desktop_path = os.path.expanduser("~/Desktop")
    organized_files = organize_files(desktop_path)
    
    print("File organization complete!")
    for category, files in organized_files.items():
        if files:
            print(f"\n{category.upper()}:")
            for file in files:
                print(f"  - {os.path.basename(file)}")

if __name__ == "__main__":
    main()
'''
        return template
    
    def get_evolution_params(self) -> Dict:
        """Get parameters for the evolution process."""
        return {
            'population_size': 200,  # Increased for more diversity
            'num_generations': 200,  # More generations for better convergence
            'mutation_rate': 0.4,    # Higher mutation rate for more exploration
            'crossover_rate': 0.9,   # Higher crossover rate for better mixing
            'tournament_size': 3     # Smaller tournament size for more diversity
        }
    
    def get_fitness_weights(self) -> Dict[str, float]:
        """Get weights for different fitness criteria."""
        return {
            'functionality': 0.8,    # Increased weight on functionality
            'efficiency': 0.1,       # Basic efficiency requirement
            'robustness': 0.05,      # Basic robustness requirement
            'maintainability': 0.05  # Basic maintainability requirement
        }
    
    def evaluate_fitness(self, code: str) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the fitness of a given code solution.
        Returns a tuple of (total_fitness, individual_scores).
        """
        # Initialize scores dictionary
        scores = {
            'functionality': 0.0,
            'efficiency': 0.0,
            'robustness': 0.0,
            'maintainability': 0.0
        }
        
        # Step 1: Basic syntax check
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            print(f"Syntax error in evolved code: {e}")
            return 0.0, scores
        
        # Step 2: Create a temporary test directory
        test_dir = Path("test_desktop")
        test_dir.mkdir(exist_ok=True)
        
        # Step 3: Create test files with known extensions
        test_files = {
            'test.txt': 'documents',
            'test.jpg': 'images',
            'test.pdf': 'documents',
            'test.py': 'code',
            'test.mp3': 'audio'
        }
        
        for filename in test_files:
            (test_dir / filename).touch()
        
        try:
            # Step 4: Execute the code in a safe environment
            local_vars = {}
            exec(code, {'__builtins__': __builtins__}, local_vars)
            
            # Step 5: Verify required functions exist
            required_functions = ['organize_files', 'get_file_type', 'create_category_dirs']
            for func in required_functions:
                if func not in local_vars:
                    print(f"Missing required function: {func}")
                    return 0.0, scores
            
            # Step 6: Run the organization
            try:
                result = local_vars['organize_files'](str(test_dir))
            except Exception as e:
                print(f"Error running organize_files: {e}")
                return 0.0, scores
            
            # Step 7: Verify files were actually moved
            if not result:
                print("No files were organized")
                return 0.0, scores
            
            # Step 8: Check if files were moved to correct categories
            correct_moves = 0
            total_files = len(test_files)
            
            for category, files in result.items():
                for file_path in files:
                    filename = os.path.basename(file_path)
                    if filename in test_files:
                        expected_category = test_files[filename]
                        if category == expected_category:
                            correct_moves += 1
            
            # Step 9: Calculate functionality score
            functionality_score = correct_moves / total_files if total_files > 0 else 0.0
            scores['functionality'] = functionality_score
            
            # If functionality score is 0, return immediately
            if functionality_score == 0.0:
                return 0.0, scores
            
            # Only evaluate other criteria if the code is functional
            scores['efficiency'] = 0.8  # Placeholder
            scores['robustness'] = self._evaluate_robustness(code)
            scores['maintainability'] = self._evaluate_maintainability(code)
            
            # Calculate total fitness
            weights = self.get_fitness_weights()
            total_fitness = sum(score * weights[criteria] for criteria, score in scores.items())
            
            return total_fitness, scores
            
        except Exception as e:
            print(f"Error evaluating fitness: {e}")
            return 0.0, scores
        finally:
            # Clean up test directory
            try:
                shutil.rmtree(test_dir)
            except Exception:
                pass
    
    def _evaluate_robustness(self, code: str) -> float:
        """Evaluate how well the code handles errors."""
        # Check for error handling
        error_handling_score = 0.0
        
        # Check for try-except blocks
        if 'try' in code and 'except' in code:
            error_handling_score += 0.5
        
        # Check for specific error handling
        if 'Exception' in code:
            error_handling_score += 0.3
        
        # Check for file existence checks
        if 'os.path.exists' in code:
            error_handling_score += 0.2
        
        return error_handling_score
    
    def _evaluate_maintainability(self, code: str) -> float:
        """Evaluate code maintainability."""
        maintainability_score = 0.0
        
        # Check for docstrings
        if '"""' in code or "'''" in code:
            maintainability_score += 0.3
        
        # Check for type hints
        if ': str' in code or ': List' in code or ': Dict' in code:
            maintainability_score += 0.3
        
        # Check for meaningful variable names
        if 'file_path' in code and 'category' in code:
            maintainability_score += 0.2
        
        # Check for code organization
        if 'def ' in code and 'if __name__' in code:
            maintainability_score += 0.2
        
        return maintainability_score
    
    def _get_category_from_extension(self, extension: str) -> str:
        """Get the category name for a given file extension."""
        # Use the same CATEGORIES as in the template
        categories = {
            'images': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'},
            'documents': {'.pdf', '.doc', '.docx', '.txt', '.rtf'},
            'audio': {'.mp3', '.wav', '.ogg', '.midi', '.m4a'},
            'video': {'.mp4', '.mov', '.avi', '.mkv', '.wmv'},
            'code': {'.py', '.java', '.cpp', '.c', '.js', '.html', '.css'},
            'archives': {'.zip', '.rar', '.7z', '.tar', '.gz'},
            'spreadsheets': {'.xls', '.xlsx', '.csv'},
            'presentations': {'.ppt', '.pptx', '.key'}
        }
        for category, extensions in categories.items():
            if extension in extensions:
                return category
        return 'unknown'
    
    def get_required_boundaries(self) -> Dict:
        """Get required boundaries for the evolution process."""
        return {
            'max_file_size': {"max_file_size": 1000000},  # 1MB
            'allowed_imports': {"allowed_imports": [
                'os', 'shutil', 'pathlib', 'mimetypes', 'magic',
                'typing', 'datetime', 'json'
            ]},
            'forbidden_imports': {"forbidden_imports": [
                'subprocess', 'sys', 'ctypes', 'socket'
            ]}
        }
    
    def get_allowed_imports(self) -> List[str]:
        """Get list of allowed imports."""
        return self.get_required_boundaries()['allowed_imports']['allowed_imports']
    
    def post_process(self, code: str) -> str:
        """Post-process the evolved code."""
        # Add shebang if missing
        if not code.startswith('#!/usr/bin/env python3'):
            code = '#!/usr/bin/env python3\n' + code
            
        # Add docstring if missing
        if '"""' not in code[:100]:
            code = '"""\nDesktop File Organizer\nOrganizes files on the desktop by their content type.\n"""\n' + code
            
        return code

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description 