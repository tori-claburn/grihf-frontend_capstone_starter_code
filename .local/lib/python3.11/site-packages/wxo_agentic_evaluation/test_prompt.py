from wxo_agentic_evaluation.watsonx_provider import WatsonXProvider



def parse_json_string(input_string):
    json_char_count = 0
    json_objects = []
    current_json = ""
    brace_level = 0
    inside_json = False

    for i, char in enumerate(input_string):
        if char == "{":
            brace_level += 1
            inside_json = True
            json_char_count += 1
        if inside_json:
            current_json += char
            json_char_count += 1
        if char == "}":
            json_char_count += 1
            brace_level -= 1
            if brace_level == 0:
                inside_json = False
                try:
                    json_objects.append(json.loads(current_json))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                current_json = ""  # Reset current JSON string
    # some threshold to say there are some non-funct calling step
    is_thinking_step = len(input_string) - json_char_count > 10
    return json_objects

wai_client = WatsonXProvider(model_id="meta-llama/llama-3-405b-instruct")

prompt =  """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are trying to make tool calls. Given a raw input and tool output. Try to extract the information to make the tool call

Example:
    Tool description:
    def get_payslips(user_id: str) -> PayslipsResponse:
    Gets a user's payslips from Workday.

    :param user_id: The user's id uniquely identifying them within the Workday API.
    :return: The user's payslips.
 
 Raw inputs:\{"tool_name": "get_payslips", "args": {"user_id": '$get_user_workday_ids'}}
 tool output: {'user_id': UserWorkdayIDs(person_id='', user_id='6dcb8106e8b74b5aabb1fc3ab8ef2b92')}
 <|start_header_id|>ipython<|end_header_id|>
 {"tool_name": "get_payslips", "args": {"user_id": "6dcb8106e8b74b5aabb1fc3ab8ef2b92"}}
 <|eot_id|>
 
"""

test_sample1 = """
<|start_header_id|>assistant<|end_header_id|>
    Tool description:
    def update_direct_reports(email_id: str, members: List[str], notification:bool) -> PayslipsResponse:
    update direct reports for a given user
    :param email_id: The user's email-id uniquely identifying them within the Workday API.
    :param members: a list of user ids to be added as direct reports
    :param notification: do we send the notification to all members

 Raw inputs:  {"tool_name": "update_direct_reports", "args": {"email_id": '$get_email_id', 'members': $get_user_by_dvision]}}
 tool output: {"email_id": 'jalenm3@163.com'}
              {'members': [UserProfile(name="Lan Smith", user_id="46873f8i93", email="lan_smith@gmail.com"), UserProfile(name="Mary Rubic", user_id="34sss31", email="MaryRobic@gmail.com"), UserProfile(name="Jason Dai", user_id="8e8ewer3", email="jd@gmail.com"])}
 <|start_header_id|>ipython<|end_header_id|>"""


test_sample2 = """
<|start_header_id|>assistant<|end_header_id|>
    Tool description:
    def book_meeting(location: str, date: str, time: str) -> bool:
    update direct reports for a given user
    :param email_id: The user's email-id uniquely identifying them within the Workday API.
    :param members: a list of user ids to be added as direct reports
    :param notification: do we send the notification to all members

 Raw inputs:  {"tool_name": "book_meeting", "args": {"email_id": '$get_email_id', 'members': $get_user_by_dvision]}}
 tool output: {"email_id": 'jalenm3@163.com'}
              {'members': [UserProfile(name="Lan Smith", user_id="46873f8i93", email="lan_smith@gmail.com"), UserProfile(name="Mary Rubic", user_id="34sss31", email="MaryRobic@gmail.com"), UserProfile(name="Jason Dai", user_id="8e8ewer3", email="jd@gmail.com"])}
 <|start_header_id|>ipython<|end_header_id|>"""



outputs = wai_client.query(prompt + test_sample1)

import json
print(outputs["generated_text"])

json_obj = parse_json_string(outputs["generated_text"])[0]

print(json_obj)