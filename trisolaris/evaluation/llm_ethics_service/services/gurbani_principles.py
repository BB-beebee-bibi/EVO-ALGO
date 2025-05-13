"""
Gurbani Principles Service for the LLM Ethics Service.

This module provides functionality to load, parse, and apply Gurbani principles
to code evaluation. It serves as a core component of the ethics evaluation
system, ensuring that code aligns with Gurbani-inspired design principles.
"""

import os
import logging
import re
from typing import Dict, Any, List, Optional, Tuple

from ..config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GurbaniPrinciplesError(Exception):
    """Exception raised when Gurbani principles processing fails."""
    pass

class GurbaniPrinciples:
    """
    Service for loading and applying Gurbani principles to code evaluation.
    
    This class handles the loading, parsing, and application of Gurbani principles
    from the design principles markdown file, providing methods to evaluate code
    against these principles.
    """
    
    def __init__(self, principles_path: Optional[str] = None):
        """
        Initialize the Gurbani principles service.
        
        Args:
            principles_path: Path to the Gurbani principles markdown file
                            (defaults to config setting if not provided)
        """
        self.principles_path = principles_path or os.path.join(
            os.path.dirname(__file__), 
            settings.GURBANI_PRINCIPLES_PATH
        )
        self.principles = {}
        self.raw_principles = ""
        self._load_principles()
        logger.info("Initialized Gurbani principles service")
    
    def _load_principles(self) -> None:
        """
        Load Gurbani principles from the markdown file.
        
        Raises:
            GurbaniPrinciplesError: If principles cannot be loaded
        """
        try:
            with open(self.principles_path, 'r') as f:
                self.raw_principles = f.read()
            
            # Parse principles from markdown
            self.principles = self._parse_principles(self.raw_principles)
            logger.info(f"Loaded {len(self.principles)} Gurbani principles")
            
        except FileNotFoundError:
            logger.warning(f"Gurbani principles file not found at {self.principles_path}. Using default principles.")
            # Use default principles if file not found
            self.raw_principles = self._get_default_principles()
            self.principles = self._parse_principles(self.raw_principles)
        except Exception as e:
            error_msg = f"Error loading Gurbani principles: {str(e)}"
            logger.error(error_msg)
            raise GurbaniPrinciplesError(error_msg)
    
    def _parse_principles(self, markdown: str) -> Dict[str, Dict[str, Any]]:
        """
        Parse principles from markdown text.
        
        Args:
            markdown: Markdown text containing principles
            
        Returns:
            Dictionary mapping principle names to their details
        """
        principles = {}
        
        # Extract principle sections using regex
        principle_pattern = r'## (\d+)\. ([^\n]+)\n\n\*\*Principle:\*\* ([^\n]+)\n\n\*\*Technical Implementation:\*\*\n(.*?)(?=\n\n\*\*Examples by Project:|$)'
        matches = re.finditer(principle_pattern, markdown, re.DOTALL)
        
        for match in matches:
            number = match.group(1)
            name = match.group(2).strip()
            description = match.group(3).strip()
            
            # Extract implementation points
            implementation_text = match.group(4).strip()
            implementation_points = [
                point.strip('- \n') 
                for point in implementation_text.split('\n') 
                if point.strip().startswith('-')
            ]
            
            # Create a key for the principle (lowercase with underscores)
            key = name.lower().replace(' ', '_').replace('-', '_')
            
            principles[key] = {
                'number': int(number),
                'name': name,
                'description': description,
                'implementation': implementation_points
            }
        
        return principles
    
    def _get_default_principles(self) -> str:
        """
        Get default Gurbani principles as markdown text.
        
        Returns:
            Markdown text containing default principles
        """
        return """
        # Gurbani-Inspired Design Principles for Technology Development
        
        ## Core Philosophy
        These design principles draw from the spiritual wisdom of Gurbani to inform technical decisions across all projects. By consciously incorporating these teachings, we create technology that not only functions effectively but also embodies wisdom, compassion, and ethical awareness.
        
        ## 1. Unity in Design
        
        **Principle:** Create systems that recognize and respect interconnection.
        
        **Technical Implementation:**
        - Use modular architecture with clear interfaces to enable seamless interaction
        - Implement consistent design patterns across components
        - Design data structures that maintain relationship information
        - Prioritize interoperability and open standards
        - Reduce unnecessary duplication of functionality
        
        ## 2. Natural Flow
        
        **Principle:** Work with rather than against natural patterns and limitations.
        
        **Technical Implementation:**
        - Design interfaces that follow human cognitive patterns
        - Create algorithms that respect computational constraints
        - Structure data to reflect its inherent organization
        - Build systems that gracefully handle edge cases and exceptions
        - Implement graceful degradation under resource constraints
        
        ## 3. Truth and Transparency
        
        **Principle:** Create systems that embody and facilitate truthfulness.
        
        **Technical Implementation:**
        - Provide clear error messages that accurately describe problems
        - Create logs and monitoring that give true system state
        - Design interfaces that accurately represent capabilities and limitations
        - Implement verification systems for data integrity
        - Avoid manipulative patterns that mislead users
        
        ## 4. Service-Oriented Architecture
        
        **Principle:** Design with the sincere intention to serve users and the broader community.
        
        **Technical Implementation:**
        - Prioritize features based on genuine user needs, not merely engagement
        - Design interfaces that empower rather than manipulate users
        - Create accessibility features for diverse user capabilities
        - Build in sustainability to serve future generations
        - Include mechanisms for community feedback and improvement
        
        ## 5. Balance and Harmony
        
        **Principle:** Find the middle path between competing concerns.
        
        **Technical Implementation:**
        - Balance performance with readability and maintainability
        - Find equilibrium between flexibility and standardization
        - Balance automation with human oversight
        - Create systems with both structure and adaptability
        - Balance immediate needs with long-term sustainability
        
        ## 6. Ego-Free Development
        
        **Principle:** Create with awareness of how technology can either reinforce or transcend ego.
        
        **Technical Implementation:**
        - Design systems that focus on functionality over branding
        - Create attribution systems that recognize all contributors
        - Implement feedback mechanisms without judgment
        - Design social features that don't amplify status-seeking
        - Build systems that serve collective well-being, not just individual desires
        
        ## 7. Universal Design
        
        **Principle:** Create technology accessible and beneficial to all, without distinction.
        
        **Technical Implementation:**
        - Follow accessibility standards (WCAG, etc.) as minimum requirements
        - Test with diverse users across different contexts and abilities
        - Design for low-bandwidth and resource-constrained environments
        - Create documentation for different technical literacy levels
        - Implement internationalization and localization
        
        ## 8. Mindful Resource Usage
        
        **Principle:** Use only what is necessary, without excess or waste.
        
        **Technical Implementation:**
        - Optimize for efficiency in computation, memory, and storage
        - Minimize network traffic and bandwidth usage
        - Reduce power consumption and carbon footprint
        - Create lightweight implementations where possible
        - Only collect data that serves a clear purpose
        """
    
    def get_principles_text(self) -> str:
        """
        Get the full text of the Gurbani principles.
        
        Returns:
            Raw principles text
        """
        return self.raw_principles
    
    def get_principles_summary(self) -> Dict[str, str]:
        """
        Get a summary of all principles.
        
        Returns:
            Dictionary mapping principle names to descriptions
        """
        return {name: data['description'] for name, data in self.principles.items()}
    
    def get_principle(self, name: str) -> Dict[str, Any]:
        """
        Get details for a specific principle.
        
        Args:
            name: Name or key of the principle
            
        Returns:
            Dictionary containing principle details
            
        Raises:
            GurbaniPrinciplesError: If principle not found
        """
        # Normalize name to key format
        key = name.lower().replace(' ', '_').replace('-', '_')
        
        if key in self.principles:
            return self.principles[key]
        
        # Try partial matching
        for principle_key in self.principles:
            if key in principle_key or principle_key in key:
                return self.principles[principle_key]
        
        raise GurbaniPrinciplesError(f"Principle not found: {name}")
    
    def evaluate_code_against_principles(self, code: str) -> Dict[str, Any]:
        """
        Perform a basic evaluation of code against Gurbani principles.
        
        This is a simple rule-based evaluation that checks for basic patterns
        in the code that might indicate alignment or misalignment with principles.
        For more sophisticated evaluation, use an LLM-based approach.
        
        Args:
            code: Source code to evaluate
            
        Returns:
            Dictionary containing evaluation results
        """
        results = {
            "overall_score": 0.0,
            "principle_scores": {},
            "concerns": [],
            "recommendations": []
        }
        
        # Unity in Design
        unity_score = 0.8
        if code.count("import") > 10:
            unity_score -= 0.2
            results["concerns"].append("Many imports may indicate poor unity in design")
            results["recommendations"].append("Consider consolidating imports and reducing dependencies")
        
        # Natural Flow
        flow_score = 0.8
        if code.count("if ") > 15 or code.count("for ") > 10:
            flow_score -= 0.2
            results["concerns"].append("Complex control flow may violate natural flow principle")
            results["recommendations"].append("Simplify control flow and break down complex functions")
        
        # Truth and Transparency
        transparency_score = 0.8
        if code.count('"""') < 2 and code.count("def ") > 3:
            transparency_score -= 0.2
            results["concerns"].append("Lack of documentation affects transparency")
            results["recommendations"].append("Add docstrings and comments to improve transparency")
        
        # Service-Oriented
        service_score = 0.8
        if "except" in code and "error_msg" not in code.lower():
            service_score -= 0.1
            results["concerns"].append("Lack of user-friendly error messages affects service orientation")
            results["recommendations"].append("Add clear error messages to help users understand issues")
        
        # Balance and Harmony
        balance_score = 0.8
        if len(code.split("\n")) > 300:
            balance_score -= 0.1
            results["concerns"].append("Very large file may indicate lack of balance")
            results["recommendations"].append("Consider breaking down large files into smaller modules")
        
        # Ego-Free Development
        ego_free_score = 0.8
        if "password" in code.lower() or "api_key" in code.lower():
            ego_free_score -= 0.2
            results["concerns"].append("Hardcoded credentials may indicate ego-centric development")
            results["recommendations"].append("Use environment variables or secure storage for credentials")
        
        # Universal Design
        universal_score = 0.8
        if "color" in code.lower() and "contrast" not in code.lower():
            universal_score -= 0.1
            results["concerns"].append("UI elements may not consider accessibility")
            results["recommendations"].append("Ensure UI elements have proper contrast and accessibility features")
        
        # Mindful Resource Usage
        resource_score = 0.8
        if "open(" in code and "close(" not in code and "with" not in code:
            resource_score -= 0.2
            results["concerns"].append("Resources may not be properly managed")
            results["recommendations"].append("Use context managers (with statements) for resource management")
        
        # Compile principle scores
        results["principle_scores"] = {
            "unity_in_design": unity_score,
            "natural_flow": flow_score,
            "truth_and_transparency": transparency_score,
            "service_oriented": service_score,
            "balance_and_harmony": balance_score,
            "ego_free_development": ego_free_score,
            "universal_design": universal_score,
            "mindful_resource_usage": resource_score
        }
        
        # Calculate overall score
        results["overall_score"] = sum(results["principle_scores"].values()) / len(results["principle_scores"])
        
        return results
    
    def get_evaluation_prompt(self) -> str:
        """
        Get a prompt for LLM-based evaluation against Gurbani principles.
        
        Returns:
            Prompt text for LLM evaluation
        """
        principles_summary = "\n\n".join([
            f"{i+1}. {data['name']}: {data['description']}"
            for i, (_, data) in enumerate(sorted(self.principles.items(), key=lambda x: x[1]['number']))
        ])
        
        return f"""
        Please evaluate the given code against the following Gurbani-inspired design principles:
        
        {principles_summary}
        
        For each principle, provide:
        1. A score from 0.0 to 1.0 indicating alignment
        2. Specific aspects of the code that align or misalign with the principle
        3. Recommendations for better alignment
        
        Format your response as a JSON object with the following structure:
        {{
            "score": float,  // Overall alignment score from 0.0 to 1.0
            "principle_scores": {{
                "unity_in_design": float,
                "natural_flow": float,
                "truth_and_transparency": float,
                "service_oriented": float,
                "balance_and_harmony": float,
                "ego_free_development": float,
                "universal_design": float,
                "mindful_resource_usage": float
            }},
            "concerns": [string],  // List of specific concerns
            "recommendations": [string]  // List of recommendations for better alignment
        }}
        """