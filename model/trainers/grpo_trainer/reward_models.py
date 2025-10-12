from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Literal, Optional, Union
import numpy as np

class RewardModelBaseClass:
    def __init__(self, max_reward:int, begin_label:str, end_label:str) -> None:
        self.max_reward = max_reward
        self.begin_label = begin_label
        self.end_label = end_label
        self.__begin_label_length = len(begin_label)
    def __get_answer(self, model_result:str) -> str:
        if self.begin_label not in model_result or self.end_label not in model_result:
            return model_result
        begin_idx = model_result.find(self.begin_label) + self.__begin_label_length
        end_idx = model_result.find(self.end_label)
        result = model_result[begin_idx:end_idx]
        return result
    def reward(self, pred:str, truth:str) -> List[str]:
        return 0.
    def __call__(self, pred:Union[str, List[str]], truth:str) -> List[float]:
        if not isinstance(pred, list):
            pred = [pred]
        ans = [self.__get_answer(prd) for prd in pred]
        r = [self.reward(ans_, truth) for ans_ in ans]
        r = np.array(r, dtype='float')
        r -= r.min()
        r /= r.max()
        r *= self.max_reward
        return r.tolist()

class AccuracyRewardModelForMathProblems(RewardModelBaseClass):
    def reward(self, pred:str, truth:str) -> float:
        total = 0
        for c in pred:
            idx = truth.find(c)
            if idx < 0:
                continue
            else:
                total += 1
                truth = truth[idx+1:]
        return total

class AccuracyRewardModelForCodingProblems(RewardModelBaseClass):
    # Still developing
    pass

class ReasoningFormatRewardModel(RewardModelBaseClass):
    def __init__(self, max_reward:float, deepseek_distill_qwen3_path:str = 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'):
        self.max_reward = max_reward
        self.tokenizer = AutoTokenizer.from_pretrained(deepseek_distill_qwen3_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(deepseek_distill_qwen3_path, trust_remote_code=True)
    def reward(self, pred:str)->float:
        prompt = f'''下面是对于某个问题，由AI大模型生成的结果。其中思考过程（即思维链，CoT）应嵌套在<think>...</think>中。
        <think>...</think>标签中应嵌套思考过程而不是最终回答。最终回答在</think>结束后。如果回答很简单，例如对“你好”的回答，<think>...</think>标签内部可以为空。
        请检查该AI在回答中是否做到这一点，并根据以上规则进行打分。
        0 表示很差
        1 表示较差
        2 表示一般
        3 表示较好
        4 表示很好
        请只打分，不要回答其他无关内容。不要尝试修改回答。打分请嵌套在`...`中。例如，`4`。下面是该AI大模型生成的结果：\n{pred}\n请打分。注意只有最终结果可以使用`...`，其他部分不允许使用此符号。
        最终打分及其外部的`...`请放在回答的最后6个字符内。
        '''
        inputs = ''   # Still developing, not completed
        outputs = ''  # Still developing, not completed
        outputs = outputs[-10:]
        begin_idx = outputs.find('`')+1
        end_idx = outputs[begin_idx:].find('`')
        try:
            r = outputs[begin_idx:end_idx]
            return float(r)
        except:
            return 2.0
    def __call__(self, pred:Union[str, List[str]]) -> List[float]:
        if not isinstance(pred, list):
            pred = [pred]
        r = [self.reward(ans) for ans in pred]
        r = np.array(r, dtype='float')
        r -= r.min()
        r /= r.max()
        r *= self.max_reward
        return r.tolist()

class RewardModelForCreativeWriting(RewardModelBaseClass):
    def __init__(self, max_reward:float, deepseek_distill_qwen3_path:str = 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'):
        self.max_reward = max_reward
        self.tokenizer = AutoTokenizer.from_pretrained(deepseek_distill_qwen3_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(deepseek_distill_qwen3_path, trust_remote_code=True)
    def reward(self, pred:str)->float:
        prompt = f'''下面是对于某个问题，由AI大模型生成的结果。类型：Creative Writing
        请打分，要求必须在0~50内。可以从多个方面进行打分，再全部加起来。最终得分要求必须在0~50内，并嵌套在`...`内部，如`40`。
        如果分多个方面，这些方面的得分不得嵌套在`...`，只有总分可以嵌套在`...`中。请只打分，不要生成任何无关内容，不要尝试改进AI生成的结果。
        下面是该AI大模型生成的结果：\n{pred}\n请打分。注意只有最终结果可以使用`...`，其他部分不允许使用此符号。最终打分及其外部的`...`请放在回答的最后6个字符内。
        '''
        inputs = ''   # Still developing, not completed
        outputs = ''  # Still developing, not completed
        outputs = outputs[-10:]
        begin_idx = outputs.find('`')+1
        end_idx = outputs[begin_idx:].find('`')
        try:
            r = outputs[begin_idx:end_idx]
            return float(r)
        except:
            return 25.0
    def __call__(self, pred:Union[str, List[str]]) -> List[float]:
        if not isinstance(pred, list):
            pred = [pred]
        r = [self.reward(ans) for ans in pred]
        r = np.array(r, dtype='float')
        r -= r.min()
        r /= r.max()
        r *= self.max_reward
        return r.tolist()

class RewardModelForLanguageMixing(RewardModelBaseClass):
    def __init__(self, max_reward:float, deepseek_distill_qwen3_path:str = 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'):
        self.max_reward = max_reward
        self.tokenizer = AutoTokenizer.from_pretrained(deepseek_distill_qwen3_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(deepseek_distill_qwen3_path, trust_remote_code=True)
    def reward(self, pred:str, quesion:str)->float:
        prompt = f'''下面是对于某个问题，由AI大模型生成的结果。
        请检查是否存在语言混合或标点混乱的问题。回答内容与用户问题语言不同，也算语言混合。
        注意：
            - 个别只有英文的专业术语如Transformer，Python除外。
            - 代码部分不考虑。
            - 中文使用的标点符号是 ，。？：；“”‘’、《》（）！ ······
            - 英文等其他语言使用的标点符号是 ,.?:'"'()! ...
            - 如果是Markdown或LaTeX语法使用了其他标点，不考虑
            - 英文不能用书名号《》
            - 中文在标题部分不可以用书名号《》，但可以用Markdown语法加粗。在其他部分引用时才可以用书名号。
        打分范围：0~20，0表示语言混合和标点混乱严重，20表示回答使用单一语言，标点使用规范。分数请嵌套在`...`中，如`15`。
        请只打分，不要生成任何无关内容，不要尝试改进AI生成的结果，不要回答下面的用户问题。
        下面是用户的问题：{quesion}
        下面是该AI大模型生成的结果：\n{pred}\n请打分。注意只有最终结果可以使用`...`，其他部分不允许使用此符号。最终打分及其外部的`...`请放在回答的最后6个字符内。
        '''
        inputs = ''   # Still developing, not completed
        outputs = ''  # Still developing, not completed
        outputs = outputs[-10:]
        begin_idx = outputs.find('`')+1
        end_idx = outputs[begin_idx:].find('`')
        try:
            r = outputs[begin_idx:end_idx]
            return float(r)
        except:
            return 10.0
    def __call__(self, pred:Union[str, List[str]], quesion:str) -> List[float]:
        if not isinstance(pred, list):
            pred = [pred]
        r = [self.reward(ans, quesion) for ans in pred]
        r = np.array(r, dtype='float')
        r -= r.min()
        r /= r.max()
        r *= self.max_reward
        return r.tolist()

class RewardModel:
    def __init__(
            self,
            task_type:Literal['math', 'coding', 'creative_writing', 'other'],
            begin_label:Optional[str] = None,
            end_label:Optional[str] = None
        ):
        if task_type in ['math', 'coding'] and (begin_label is None or end_label is None):
            raise TypeError('For math and coding problems, begin_label and end_label must not be None.')
        self.format_reward = ReasoningFormatRewardModel(20)
        self.language_reward = RewardModelForLanguageMixing(20)
        if task_type == 'math':
            self.task_reward = AccuracyRewardModelForMathProblems(60, begin_label, end_label)
            self.__requires_truth = True
        elif task_type == 'coding':
            self.task_reward = AccuracyRewardModelForCodingProblems(60, begin_label, end_label)
            self.__requires_truth = True
        elif task_type == 'creative_writing':
            self.task_reward = RewardModelForCreativeWriting(60)
            self.__requires_truth = False
        else:
            self.task_reward = lambda ans:45
            self.__requires_truth = False
    def __call__(self, question:str, results:Union[str, List[str]], truth:Optional[str]=None):
        format_reward = self.format_reward(results)
        lang_reward = self.language_reward(results, question)
        if self.__requires_truth:
            acc_reward = self.task_reward(results, truth)
        else:
            acc_reward = self.task_reward(results)
        final_reward = format_reward + lang_reward + acc_reward
        return final_reward