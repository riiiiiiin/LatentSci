# %%
import json
import regex as re

def tranform_str_to_json(str_input):
    if str_input is None:
        return None
    ## 假如LLM输出的是类似json的字符串, 我需要设定一个逻辑, 把字符串重新转换成json
    ## o1-mini的感觉, 是要移除字符串里面的\n，并且把所有的\"都改成 "
    if "</think>\n\n" in str_input:
        str_input = str_input.split("</think>\n\n")[-1]
        
    if "```json\n" in str_input:
        str_input = str_input.split("```json\n")[1]
        str_input = str_input.replace("\n```", '')
    
    unescaped_str = str_input.replace('\n    ', '').replace('\n', '').replace('\"', '"')
    
    pattern = re.compile(
        r'("Major Product"\s*:\s*"(?:\\.|[^"\\])*"\s*)(?=\s*"Byproduct\(s\)")',
        flags=re.DOTALL
    )
    
    '''{
    "Major Product": "CC(C)(C)OC(=O)N[C@H](C=O)CC1CCC1""Byproduct(s)": "CCOC(C)=O.CC(=O)O"
    }'''

    unescaped_str = re.sub(pattern, r'\1,', unescaped_str)
    
    try:
        json_obj = json.loads(unescaped_str)
    except json.JSONDecodeError as e:
        return None
    
    def _replace_outputs(obj):
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                if k == 'outputs':
                    if isinstance(v, list):
                        new_value = v[0] if len(v) > 0 else None
                    else:
                        new_value = v
                    new_obj[k] = new_value
                else:
                    new_obj[k] = v
            return new_obj
        else: return obj

    processed = _replace_outputs(json_obj)
    return processed
# %%

answer_pattern = re.compile(r'<answer\s*>(.*?)</answer\s*>', flags=re.S)

def extract_answer(str_input):
    matches = re.findall(answer_pattern, str_input)
    last_content = matches[-1].strip() if matches else None
    return last_content