import re

class Extension:
    begin_label = None
    end_label = None
    system_prompt = None
    output_prompt = None
    def __init__(self):
        pattern = re.compile(f'{self.begin_label}.*?{self.end_label}')
    def forward(self, input_str):
        pass
    def __call__(self, input_str):
        pass
