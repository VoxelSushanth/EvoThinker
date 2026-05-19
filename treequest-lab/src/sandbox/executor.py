"""
Sandboxed Code Executor for TreeQuest Lab

Provides safe execution of generated experiment code with:
- Timeout limits
- Resource constraints
- Output capture
- Error isolation
"""

import logging
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution."""
    
    success: bool
    stdout: str
    stderr: str
    return_code: int
    duration_seconds: float
    memory_usage_mb: Optional[float] = None
    error_type: Optional[str] = None
    metrics: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "duration_seconds": self.duration_seconds,
            "memory_usage_mb": self.memory_usage_mb,
            "error_type": self.error_type,
            "metrics": self.metrics
        }


class SandboxConfig(BaseModel):
    """Configuration for sandboxed execution."""
    
    timeout_seconds: int = Field(
        default=3600,
        gt=0,
        description="Maximum execution time"
    )
    max_memory_mb: int = Field(
        default=8192,
        gt=0,
        description="Maximum memory usage in MB"
    )
    use_subprocess: bool = Field(
        default=True,
        description="Use subprocess isolation"
    )
    capture_output: bool = Field(
        default=True,
        description="Capture stdout/stderr"
    )
    working_dir: Optional[str] = Field(
        default=None,
        description="Working directory for execution"
    )
    environment_vars: dict = Field(
        default_factory=dict,
        description="Additional environment variables"
    )
    allow_network: bool = Field(
        default=False,
        description="Allow network access (discouraged)"
    )
    python_path: str = Field(
        default=sys.executable,
        description="Python interpreter path"
    )
    
    class Config:
        extra = "ignore"


class SandboxExecutor:
    """
    Sandboxed code executor for safe experiment running.
    
    Provides:
    - Process isolation via subprocess
    - Timeout enforcement
    - Resource limits (memory, CPU)
    - Output capture and parsing
    - Error handling and reporting
    """
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self._execution_count = 0
        
        # Set up restricted environment
        self.env = os.environ.copy()
        self.env.update(self.config.environment_vars)
        
        # Remove dangerous variables if not allowing network
        if not self.config.allow_network:
            for key in ['http_proxy', 'https_proxy', 'ftp_proxy']:
                self.env.pop(key, None)
        
        logger.info(f"SandboxExecutor initialized with timeout={self.config.timeout_seconds}s")
    
    def execute(
        self,
        code_path: Path | str,
        args: Optional[list[str]] = None,
        cwd: Optional[Path | str] = None
    ) -> ExecutionResult:
        """
        Execute a Python script in sandboxed environment.
        
        Args:
            code_path: Path to Python script
            args: Command-line arguments
            cwd: Working directory
            
        Returns:
            ExecutionResult with outputs and metrics
        """
        import time
        
        self._execution_count += 1
        code_path = Path(code_path)
        
        if not code_path.exists():
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"File not found: {code_path}",
                return_code=-1,
                duration_seconds=0,
                error_type="FileNotFoundError"
            )
        
        logger.info(f"Executing {code_path} (run #{self._execution_count})")
        
        start_time = time.time()
        
        try:
            # Build command
            cmd = [self.config.python_path, str(code_path)]
            if args:
                cmd.extend(args)
            
            # Set working directory
            work_dir = Path(cwd) if cwd else Path(self.config.working_dir) if self.config.working_dir else code_path.parent
            work_dir.mkdir(parents=True, exist_ok=True)
            
            # Execute with subprocess
            if self.config.use_subprocess:
                result = self._execute_subprocess(cmd, work_dir)
            else:
                result = self._execute_in_process(code_path)
            
            # Calculate duration
            duration = time.time() - start_time
            result.duration_seconds = duration
            
            # Parse metrics from output
            result.metrics = self._parse_metrics(result.stdout)
            
            logger.info(f"Execution complete: success={result.success}, duration={duration:.2f}s")
            
            return result
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.error(f"Execution timed out after {duration:.2f}s")
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Timeout after {self.config.timeout_seconds}s",
                return_code=-2,
                duration_seconds=duration,
                error_type="TimeoutError"
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Execution failed: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-3,
                duration_seconds=duration,
                error_type=type(e).__name__
            )
    
    def _execute_subprocess(
        self,
        cmd: list[str],
        cwd: Path
    ) -> ExecutionResult:
        """Execute using subprocess with isolation."""
        
        # Set resource limits on Unix systems
        preexec_fn = None
        if sys.platform != 'win32':
            preexec_fn = self._set_resource_limits
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE if self.config.capture_output else None,
                stderr=subprocess.PIPE if self.config.capture_output else None,
                cwd=str(cwd),
                env=self.env,
                preexec_fn=preexec_fn,
                text=True
            )
            
            # Wait with timeout
            stdout, stderr = process.communicate(timeout=self.config.timeout_seconds)
            
            return ExecutionResult(
                success=process.returncode == 0,
                stdout=stdout or "",
                stderr=stderr or "",
                return_code=process.returncode
            )
            
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
    
    def _set_resource_limits(self):
        """Set Unix resource limits for subprocess."""
        import resource
        
        # Set memory limit
        max_mem_bytes = self.config.max_memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (max_mem_bytes, max_mem_bytes))
        except (ValueError, resource.error):
            pass  # May not be supported on all systems
        
        # Set CPU time limit (slightly more than timeout)
        cpu_limit = int(self.config.timeout_seconds * 1.5)
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        except (ValueError, resource.error):
            pass
    
    def _execute_in_process(self, code_path: Path) -> ExecutionResult:
        """Execute code in current process (less isolated)."""
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        # Capture outputs
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Read and execute code
            with open(code_path, 'r') as f:
                code = f.read()
            
            # Set up execution globals
            exec_globals = {
                '__name__': '__main__',
                '__file__': str(code_path),
                '__doc__': None
            }
            
            # Execute with output capture
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compile(code, str(code_path), 'exec'), exec_globals)
            
            return ExecutionResult(
                success=True,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                return_code=0
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout=stdout_capture.getvalue(),
                stderr=traceback.format_exc(),
                return_code=-4,
                error_type=type(e).__name__
            )
    
    def _parse_metrics(self, stdout: str) -> dict:
        """Parse metrics from stdout."""
        import json
        import re
        
        metrics = {}
        
        # Look for JSON blocks in output
        json_pattern = r'\{[^{}]*"score"[^{}]*\}'
        matches = re.findall(json_pattern, stdout, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match)
                if isinstance(data, dict):
                    metrics.update(data)
            except json.JSONDecodeError:
                pass
        
        # Look for specific metric patterns
        patterns = {
            'accuracy': r'accuracy[:\s]+([0-9.]+)',
            'loss': r'loss[:\s]+([0-9.]+)',
            'f1': r'f1[:\s]+([0-9.]+)',
            'precision': r'precision[:\s]+([0-9.]+)',
            'recall': r'recall[:\s]+([0-9.]+)'
        }
        
        for name, pattern in patterns.items():
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                try:
                    metrics[name] = float(match.group(1))
                except ValueError:
                    pass
        
        return metrics
    
    def execute_from_string(
        self,
        code: str,
        script_name: str = "experiment.py"
    ) -> ExecutionResult:
        """
        Execute code from string.
        
        Args:
            code: Python code as string
            script_name: Name for temporary file
            
        Returns:
            ExecutionResult
        """
        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            dir=self.config.working_dir
        ) as f:
            f.write(code)
            temp_path = Path(f.name)
        
        try:
            return self.execute(temp_path)
        finally:
            # Clean up
            try:
                temp_path.unlink()
            except:
                pass
    
    def get_statistics(self) -> dict:
        """Get executor statistics."""
        return {
            "total_executions": self._execution_count,
            "timeout_seconds": self.config.timeout_seconds,
            "max_memory_mb": self.config.max_memory_mb,
            "use_subprocess": self.config.use_subprocess
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("SandboxExecutor Test")
    print("=" * 60)
    
    # Test with simple script
    test_code = '''
print("Hello from sandbox!")
import json
print(json.dumps({"score": 0.85, "accuracy": 0.92}))
'''
    
    executor = SandboxExecutor(SandboxConfig(timeout_seconds=10))
    result = executor.execute_from_string(test_code)
    
    print(f"\nSuccess: {result.success}")
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")
    print(f"Duration: {result.duration_seconds:.3f}s")
    print(f"Metrics: {result.metrics}")
    print(f"Return code: {result.return_code}")
