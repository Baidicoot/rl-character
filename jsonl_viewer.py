#!/usr/bin/env python3
"""
JSONL Viewer - A web-based viewer for JSONL files with proper text formatting

Example usage:
python jsonl_viewer.py ./results/completion_basic_apps_ft:gpt-4.1-nano_hack.jsonl
"""

import json
import argparse
from pathlib import Path
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Global variable to store the data
data = []
filename = ""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>JSONL Viewer - {{ filename }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
        }
        .navigation {
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .nav-button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 0 10px;
            border-radius: 4px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
        }
        .nav-button:hover {
            background-color: #0056b3;
        }
        .nav-button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .field {
            margin-bottom: 25px;
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
        }
        .field-header {
            background-color: #e9ecef;
            padding: 10px 15px;
            font-weight: bold;
            color: #495057;
            border-bottom: 1px solid #ddd;
        }
        .field-content {
            padding: 15px;
            background-color: white;
            overflow-x: auto;
        }
        .text-content {
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: Arial, sans-serif;
            font-size: 14px;
            line-height: 1.4;
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 3px;
            border: 1px solid #e9ecef;
        }
        .json-content {
            white-space: pre;
            font-family: Arial, sans-serif;
            font-size: 12px;
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 3px;
            border: 1px solid #e9ecef;
            overflow-x: auto;
        }
        .list-content {
            padding: 5px 0;
        }
        .list-item {
            margin: 10px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 3px;
            border: 1px solid #e9ecef;
        }
        .message-block {
            margin: 15px 0;
            border-radius: 5px;
            overflow: hidden;
            border: 1px solid #dee2e6;
        }
        .message-role {
            background-color: #343a40;
            color: white;
            padding: 8px 12px;
            font-weight: bold;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .message-content {
            background-color: #f8f9fa;
            padding: 15px;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            line-height: 1.5;
            border-top: 1px solid #dee2e6;
        }
        .page-info {
            font-size: 18px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>JSONL Viewer</h1>
            <p class="page-info">File: {{ filename }}</p>
            <p class="page-info">Record {{ current_index + 1 }} of {{ total_records }}</p>
        </div>
        
        <div class="navigation">
            {% if current_index > 0 %}
                <a href="?index={{ current_index - 1 }}" class="nav-button">← Previous</a>
            {% else %}
                <button class="nav-button" disabled>← Previous</button>
            {% endif %}
            
            <span style="margin: 0 20px;">{{ current_index + 1 }} / {{ total_records }}</span>
            
            {% if current_index < total_records - 1 %}
                <a href="?index={{ current_index + 1 }}" class="nav-button">Next →</a>
            {% else %}
                <button class="nav-button" disabled>Next →</button>
            {% endif %}
        </div>

        <div class="content">
            {% for field_name, field_value in record_data %}
                <div class="field">
                    <div class="field-header">{{ field_name }}</div>
                    <div class="field-content">
                        {% if field_value.type == 'text' %}
                            <div class="text-content">{{ field_value.content }}</div>
                        {% elif field_value.type == 'json' %}
                            <div class="json-content">{{ field_value.content }}</div>
                        {% elif field_value.type == 'list' %}
                            <div class="list-content">
                                {% for item in field_value.content %}
                                    <div class="list-item">
                                        {% if item.type == 'text' %}
                                            <div class="text-content">{{ item.content }}</div>
                                        {% elif item.type == 'json' %}
                                            <div class="json-content">{{ item.content }}</div>
                                        {% else %}
                                            {{ item.content }}
                                        {% endif %}
                                    </div>
                                {% endfor %}
                            </div>
                        {% elif field_value.type == 'messages' %}
                            <div class="messages-content">
                                {% for message in field_value.content %}
                                    <div class="message-block">
                                        <div class="message-role">{{ message.role }}</div>
                                        <div class="message-content">{{ message.content }}</div>
                                    </div>
                                {% endfor %}
                            </div>
                        {% else %}
                            {{ field_value.content }}
                        {% endif %}
                    </div>
                </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""


def is_messages_list(value):
    """Check if a value is a list of message objects"""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(
            isinstance(item, dict) and "role" in item and "content" in item
            for item in value
        )
    )


def format_field_value(value):
    """Format a field value for display"""
    if isinstance(value, str):
        # Check if it's a multi-line string or contains special characters
        if "\n" in value or len(value) > 100:
            return {"type": "text", "content": value}
        else:
            return {"type": "text", "content": value}
    elif isinstance(value, dict):
        # Check if this dict contains a "messages" field that looks like chat messages
        if "messages" in value and is_messages_list(value["messages"]):
            # This is likely a chat completion format, let's format the messages specially
            messages = []
            for msg in value["messages"]:
                messages.append(
                    {
                        "role": msg.get("role", "unknown"),
                        "content": msg.get("content", ""),
                    }
                )
            return {"type": "messages", "content": messages}
        else:
            return {"type": "json", "content": json.dumps(value, indent=2)}
    elif isinstance(value, list):
        # Check if this is a list of messages
        if is_messages_list(value):
            messages = []
            for msg in value:
                messages.append(
                    {
                        "role": msg.get("role", "unknown"),
                        "content": msg.get("content", ""),
                    }
                )
            return {"type": "messages", "content": messages}
        else:
            # Handle other lists of objects
            formatted_items = []
            for item in value:
                if isinstance(item, dict):
                    formatted_items.append(
                        {"type": "json", "content": json.dumps(item, indent=2)}
                    )
                elif isinstance(item, str) and ("\n" in item or len(item) > 100):
                    formatted_items.append({"type": "text", "content": item})
                else:
                    formatted_items.append({"type": "text", "content": str(item)})
            return {"type": "list", "content": formatted_items}
    else:
        return {"type": "text", "content": str(value)}


@app.route("/")
def index():
    current_index = int(request.args.get("index", 0))
    current_index = max(0, min(current_index, len(data) - 1))

    if not data:
        return "No data loaded"

    record = data[current_index]

    # Format the record for display
    record_data = []
    for key, value in record.items():
        formatted_value = format_field_value(value)
        record_data.append((key, formatted_value))

    return render_template_string(
        HTML_TEMPLATE,
        filename=filename,
        current_index=current_index,
        total_records=len(data),
        record_data=record_data,
    )


def load_jsonl(filepath):
    """Load JSONL file into memory"""
    global data, filename
    data = []
    filename = Path(filepath).name

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue

    print(f"Loaded {len(data)} records from {filepath}")


def main():
    parser = argparse.ArgumentParser(description="View JSONL files in a web interface")
    parser.add_argument("jsonl_file", help="Path to the JSONL file to view")
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to run the server on (default: 127.0.0.1)",
    )

    args = parser.parse_args()

    if not Path(args.jsonl_file).exists():
        print(f"Error: File {args.jsonl_file} does not exist")
        return

    load_jsonl(args.jsonl_file)

    if not data:
        print("No valid JSON records found in the file")
        return

    print(f"Starting server at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
