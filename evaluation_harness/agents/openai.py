"""OpenAI API-based agent implementation."""

import json
import logging
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from .base import Agent
from ..environments.base import Environment
from ..utils.retry import exponential_backoff_retry

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are an expert software engineer tasked with solving coding problems.
You will be given a problem description and access to a codebase via tools.
Your goal is to understand the problem, explore the codebase, and implement a solution.
Work step by step, test your changes, and ensure the solution is correct.
When you believe the task is complete, respond with exactly 'Task Completed.'"""


class OpenAIAgent(Agent):
    """OpenAI API-based agent with retry logic."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_turns: int = 50,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ):
        """
        Initialize OpenAI agent.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (e.g., 'gpt-4-turbo-preview')
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens per response
            max_turns: Maximum conversation turns
            system_prompt: System prompt to use
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.system_prompt = system_prompt
    
    @exponential_backoff_retry(max_retries=5, base_delay=1.0, max_delay=60.0)
    async def _call_openai_with_retry(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        """
        Call OpenAI API with automatic retry on failure.
        
        Args:
            messages: Conversation messages
            tools: Tool definitions (optional)
            
        Returns:
            OpenAI API response
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        return await self.client.chat.completions.create(**kwargs)
    
    async def run(self, environment: Environment) -> List[Dict[str, Any]]:
        """
        Run agent on environment until task completion or max turns.
        
        Args:
            environment: The environment to solve
            
        Returns:
            List of message dictionaries representing the full conversation
        """
        # Initialize conversation with system prompt and initial context
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": environment.get_initial_context()}
        ]
        
        # Get tools from environment
        tools = environment.get_tools()
        
        turn_count = 0
        
        while turn_count < self.max_turns:
            turn_count += 1
            logger.info(f"Turn {turn_count}/{self.max_turns}")
            
            try:
                # Call OpenAI API with exponential backoff
                response = await self._call_openai_with_retry(messages, tools)
                
                # Process assistant response
                assistant_message = response.choices[0].message
                
                # Convert to dict format
                assistant_dict = {
                    "role": "assistant",
                    "content": assistant_message.content
                }
                
                # Add tool calls if present
                if assistant_message.tool_calls:
                    assistant_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                
                messages.append(assistant_dict)
                
                # Handle tool calls if present
                if assistant_message.tool_calls:
                    for tool_call in assistant_message.tool_calls:
                        try:
                            # Parse arguments
                            args = json.loads(tool_call.function.arguments)
                            
                            # Execute tool in environment
                            result = await environment.execute_action({
                                "tool": tool_call.function.name,
                                "arguments": args
                            })
                            
                            # Add tool result to messages
                            tool_message = {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": str(result)
                            }
                            messages.append(tool_message)
                            
                            logger.info(f"Executed tool: {tool_call.function.name}")
                            
                        except json.JSONDecodeError as e:
                            # Handle JSON parsing error
                            error_msg = f"Error parsing tool arguments: {str(e)}"
                            logger.error(error_msg)
                            
                            tool_message = {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": error_msg
                            }
                            messages.append(tool_message)
                            
                        except Exception as e:
                            # Handle tool execution error
                            error_msg = f"Error executing {tool_call.function.name}: {str(e)}"
                            logger.error(error_msg)
                            
                            tool_message = {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": error_msg
                            }
                            messages.append(tool_message)
                else:
                    # No tool calls - check if agent thinks it's done
                    if assistant_message.content and assistant_message.content.strip() == "Task Completed.":
                        logger.info("Agent declared task completed")
                        break
                    
                    # Otherwise prompt to continue
                    continue_message = {
                        "role": "user",
                        "content": "Continue working on the task. If you believe the task is complete, respond with exactly 'Task Completed.'"
                    }
                    messages.append(continue_message)
                    
            except Exception as e:
                logger.error(f"Error in agent loop: {str(e)}")
                raise
        
        return messages