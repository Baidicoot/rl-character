"""Docker container-based environment for code execution."""

import docker
import io
import tarfile
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .base import Environment

logger = logging.getLogger(__name__)


class DockerEnvironment(Environment):
    """Docker container-based environment for isolated code execution."""
    
    def __init__(self, image_name: str, working_dir: str = "/workspace"):
        """
        Initialize Docker environment.
        
        Args:
            image_name: Name of the Docker image to use
            working_dir: Working directory inside the container
        """
        self.client = docker.from_env()
        self.image_name = image_name
        self.working_dir = working_dir
        self.container = None
        self._start_container()
    
    def _start_container(self):
        """Start the Docker container."""
        try:
            self.container = self.client.containers.run(
                self.image_name,
                command="/bin/bash",
                working_dir=self.working_dir,
                tty=True,
                stdin_open=True,
                detach=True,
                auto_remove=False
            )
            logger.info(f"Started container {self.container.id[:12]} from image {self.image_name}")
        except Exception as e:
            logger.error(f"Failed to start container: {e}")
            raise
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Return file system and command execution tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute a bash command in the container",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The bash command to execute"
                            }
                        },
                        "required": ["command"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            }
                        },
                        "required": ["path"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file (creates file if it doesn't exist)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["path", "content"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Replace text in a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to edit"
                            },
                            "old_text": {
                                "type": "string",
                                "description": "Exact text to replace"
                            },
                            "new_text": {
                                "type": "string", 
                                "description": "New text to insert"
                            }
                        },
                        "required": ["path", "old_text", "new_text"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files and directories",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to list (defaults to current directory)",
                                "default": "."
                            }
                        },
                        "additionalProperties": False
                    }
                }
            }
        ]
    
    async def execute_action(self, action: Dict[str, Any]) -> Any:
        """Execute tool action in container."""
        tool_name = action["tool"]
        args = action.get("arguments", {})
        
        if tool_name == "bash":
            return self._exec_command(args["command"])
        elif tool_name == "read_file":
            return self._read_file(args["path"])
        elif tool_name == "write_file":
            return self._write_file(args["path"], args["content"])
        elif tool_name == "edit_file":
            return self._edit_file(args["path"], args["old_text"], args["new_text"])
        elif tool_name == "list_files":
            return self._list_files(args.get("path", "."))
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def _exec_command(self, command: str, timeout: int = 300) -> str:
        """Execute a command in the container."""
        try:
            result = self.container.exec_run(
                command,
                workdir=self.working_dir,
                demux=False,
                tty=False
            )
            output = result.output.decode('utf-8', errors='replace')
            
            if result.exit_code != 0:
                return f"Command failed with exit code {result.exit_code}:\n{output}"
            
            return output
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    def _read_file(self, path: str) -> str:
        """Read a file from the container."""
        try:
            # Get file content using cat
            result = self._exec_command(f"cat {path}")
            return result
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def _write_file(self, path: str, content: str) -> str:
        """Write content to a file in the container."""
        try:
            # Create a tar archive in memory
            tar_stream = io.BytesIO()
            tar = tarfile.open(fileobj=tar_stream, mode='w')
            
            # Add file to archive
            file_data = content.encode('utf-8')
            tarinfo = tarfile.TarInfo(name=Path(path).name)
            tarinfo.size = len(file_data)
            tar.addfile(tarinfo, io.BytesIO(file_data))
            tar.close()
            
            # Put archive in container
            tar_stream.seek(0)
            self.container.put_archive(
                str(Path(path).parent),
                tar_stream
            )
            
            return f"File written successfully: {path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    def _edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Edit a file by replacing text."""
        try:
            # Read current content
            content = self._read_file(path)
            if content.startswith("Error"):
                return content
            
            # Check if old_text exists
            if old_text not in content:
                return f"Error: Text to replace not found in {path}"
            
            # Replace text
            new_content = content.replace(old_text, new_text)
            
            # Write back
            return self._write_file(path, new_content)
        except Exception as e:
            return f"Error editing file: {str(e)}"
    
    def _list_files(self, path: str) -> str:
        """List files and directories."""
        try:
            result = self._exec_command(f"ls -la {path}")
            return result
        except Exception as e:
            return f"Error listing files: {str(e)}"
    
    def evaluate(self) -> bool:
        """Base implementation always returns False - override in subclasses."""
        return False
    
    def get_initial_context(self) -> str:
        """Base implementation - override in subclasses."""
        return "Environment is ready."
    
    def reset(self) -> None:
        """Reset the container to initial state."""
        if self.container:
            self.container.stop()
            self.container.remove()
        self._start_container()
    
    def cleanup(self) -> None:
        """Clean up the Docker container."""
        if self.container:
            try:
                self.container.stop()
                self.container.remove()
                logger.info(f"Cleaned up container {self.container.id[:12]}")
            except Exception as e:
                logger.error(f"Error cleaning up container: {e}")