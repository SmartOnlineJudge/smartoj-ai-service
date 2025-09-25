import os
from pathlib import Path
from typing import Dict


class PromptManager:
    """管理所有提示词的类，自动加载 prompts 目录下的所有提示词文件"""
    
    _instance = None
    _prompts: Dict[str, str] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_prompts()
        return cls._instance
    
    def _load_prompts(self):
        """自动加载 prompts 目录下的所有提示词文件"""
        prompts_dir = Path(__file__).parent
        self._traverse_directory(prompts_dir)
    
    def _traverse_directory(self, directory: Path):
        """遍历目录，加载所有提示词文件"""
        for item in directory.iterdir():
            if item.is_dir():
                # 递归遍历子目录
                self._traverse_directory(item)
            elif item.is_file() and item.suffix in ['.txt', '.md', '.prompt']:
                # 加载提示词文件
                relative_path = item.relative_to(Path(__file__).parent)
                prompt_key = str(relative_path.with_suffix('')).replace(os.sep, '.')
                self._prompts[prompt_key] = item.read_text(encoding='utf-8').strip()
    
    @classmethod
    def get_prompt(cls, key: str) -> str:
        """根据键名获取提示词内容"""
        instance = cls()
        return instance._prompts.get(key, '')
    
    @classmethod
    def list_prompts(cls) -> Dict[str, str]:
        """列出所有提示词"""
        instance = cls()
        return instance._prompts.copy()
    
    @property
    def prompts(self) -> Dict[str, str]:
        """获取所有提示词的属性"""
        return self._prompts.copy()
