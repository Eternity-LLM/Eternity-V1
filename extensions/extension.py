import re

ext_list = []
def register_extension(ext):
    name = type(ext).__name__
    begin_label = ext.begin_label
    end_label = ext.end_label
    system_prompt = ext.system_prompt
    ext_list.append({
        'name' : name,
        'begin' : begin_label,
        'end' : end_label,
        'system' : system_prompt,
        '__call__' : ext
    })

class Extension:
    begin_label = None
    end_label = None
    system_prompt = None
    output_prompt = None
    def __init__(self):
        register_extension(self)
        self.pattern = re.compile(f'{self.begin_label}(.*?){self.end_label}')
    def forward(self, input_str):
        pass
    def __call__(self, input_str):
        self.pattern.findall(input_str)
        return self.forward(input_str) + self.output_prompt