from typing import Dict, Any


def get_mcp_prompt(app_config: Dict[str, Any], workflow: str, agent_name: str) -> str:
    """Return MCP prompt for a given agent within a workflow."""
    mcp_cfg = app_config.get("mcp", {})
    workflow_cfg = mcp_cfg.get("workflow_settings", {}).get(workflow, {})
    tool_name = workflow_cfg.get(agent_name)
    if not tool_name:
        tool_name = mcp_cfg.get("agent_defaults", {}).get(agent_name)
    if not tool_name:
        tool_name = mcp_cfg.get("default_tool")
    if not tool_name:
        return ""
    for tool in mcp_cfg.get("tools", []):
        if tool.get("name") == tool_name:
            return tool.get("prompt", "")
    return ""
