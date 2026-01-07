from collections import defaultdict

from wxo_agentic_evaluation.inference_backend import WXOClient, is_saas_url


class ResourceMap:
    def __init__(self, wxo_client: WXOClient):
        self.wxo_client = wxo_client
        self.agent2tools, self.tools2agents = self.init_mapping()
        self.all_agents = list(self.agent2tools.keys())

    def init_mapping(self):
        agent2tools = defaultdict(set)
        tools2agents = defaultdict(set)
        if is_saas_url(self.wxo_client.service_url):
            # TO-DO: this is not validated after the v1 prefix change
            # need additional validation
            tools_path = "v1/orchestrate/tools"
            agents_path = "v1/orchestrate/agents"
        else:
            tools_path = "v1/tools/"
            agents_path = "v1/orchestrate/agents/"

        tool_map = {}

        resp = self.wxo_client.get(tools_path)
        if resp.status_code == 200:
            tools = resp.json()
            tool_map = {tool["id"]: tool["name"] for tool in tools}
        else:
            resp.raise_for_status()

        resp = self.wxo_client.get(agents_path)

        if resp.status_code == 200:
            agents = resp.json()
            for agent in agents:
                agent_name = agent["name"]
                tools = [tool_map[id] for id in agent["tools"]]
                for tool in tools:
                    agent2tools[agent_name].add(tool)
                    tools2agents[tool].add(agent_name)
        else:
            resp.raise_for_status()

        agent2tools = dict(agent2tools)
        tools2agents = dict(tools2agents)
        return agent2tools, tools2agents
