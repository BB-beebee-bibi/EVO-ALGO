PROJECT: EVO-ALGO/TRISOLARIS FRAMEWORK
===================================

UNDERSTANDING
------------
TRISOLARIS = The evolutionary algorithm framework (the engine)
minja_usb_scan.py = Example program TO BE evolved BY the framework
run.py = Main entry point for TRISOLARIS engine

CURRENT STRUCTURE
----------------
- Root directory contains main framework files
- trisolaris/ contains core implementation
- minja/ contains input programs for evolution  
- evolved_minja/ contains evolved outputs

PRIORITY IMPROVEMENTS
--------------------
1. REORGANIZE DIRECTORY STRUCTURE
   - Move minja_usb_scan.py to minja/ directory
   - Implement timestamped output folders: 
     outputs/run_YYYYMMDD_HHMMSS/generation_N/
   
2. RESOURCE MANAGEMENT
   - Add Resource Steward implementation
   - Ensure system stability during long runs
   - Monitor CPU/memory usage

3. ETHICAL BOUNDARIES
   - Strengthen ethical_filter.py safety checks
   - Ensure sandboxed execution of evolved code
   - No harmful code generation

4. PERFORMANCE OPTIMIZATION
   - Review evolutionary loop efficiency
   - Optimize selection/mutation operations
   - Minimize unnecessary operations

WORKFLOW
--------
1. Analyze existing code thoroughly
2. Ask clarifying questions first
3. Propose changes incrementally  
4. Verify functionality after changes
5. Document all modifications

CODING PRINCIPLES
----------------
- Follow Python best practices
- Maintain clear separation of concerns
- Use descriptive variable/function names
- Include comprehensive docstrings
- Write unit tests for new features