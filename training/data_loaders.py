"""
Data loaders for three-phase training curriculum.

Each phase requires different data formats and augmentations:
- Phase 1: Minimal pairs (singular/plural, tense changes, etc.)
- Phase 2: Complex sentences requiring multiple coordinate activations
- Phase 3: Multi-turn dialogues
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import random


class PrimitiveDataset(Dataset):
    """
    Dataset for Phase 1: Primitive Differentiation

    Generates minimal pairs that differ in exactly one linguistic dimension:
    - Singular vs Plural: "The cat runs" / "The cats run"
    - Tense: "I walk" / "I walked"
    - Formality: "Hello" / "Good day, sir"
    """
    def __init__(self, tokenizer, seq_len: int = 512, num_samples: int = 1_000_000):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_samples = num_samples

        # Define primitive contrasts
        self.contrasts = {
            'number': [
                ("The cat runs", "The cats run"),
                ("A dog barks", "Dogs bark"),
                ("The house is big", "The houses are big"),
                ("I see a bird", "I see birds"),
            ],
            'tense': [
                ("I walk to school", "I walked to school"),
                ("She eats lunch", "She ate lunch"),
                ("They play games", "They played games"),
                ("He writes code", "He wrote code"),
            ],
            'formality': [
                ("Hi there", "Good day, sir"),
                ("Yeah, sure", "Yes, certainly"),
                ("Thanks", "I am grateful"),
                ("What's up?", "How do you do?"),
            ],
            'aspect': [
                ("I eat", "I am eating"),
                ("She works", "She is working"),
                ("They sleep", "They are sleeping"),
                ("He studies", "He is studying"),
            ],
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a minimal pair and their contrast type.

        Returns:
            Dictionary with:
            - input_ids1: First sentence tokens
            - input_ids2: Contrasting sentence tokens
            - contrast_type: Which dimension they differ on
            - pair_id: Unique ID for this pair
        """
        # Select random contrast type
        contrast_type = random.choice(list(self.contrasts.keys()))
        pair = random.choice(self.contrasts[contrast_type])

        # Tokenize both sentences
        tokens1 = self.tokenizer.encode(pair[0], add_special_tokens=True)
        tokens2 = self.tokenizer.encode(pair[1], add_special_tokens=True)

        # Pad/truncate to seq_len
        tokens1 = self._pad_or_truncate(tokens1)
        tokens2 = self._pad_or_truncate(tokens2)

        return {
            'input_ids1': torch.tensor(tokens1, dtype=torch.long),
            'input_ids2': torch.tensor(tokens2, dtype=torch.long),
            'contrast_type': hash(contrast_type) % 1024,  # Map to coordinate space
            'pair_id': idx
        }

    def _pad_or_truncate(self, tokens: List[int]) -> List[int]:
        """Pad or truncate to seq_len"""
        if len(tokens) < self.seq_len:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_len - len(tokens))
        else:
            tokens = tokens[:self.seq_len]
        return tokens


class CompositionalDataset(Dataset):
    """
    Dataset for Phase 2: Compositional Blending

    Generates sentences that require multiple coordinate activations:
    - "The doctor formally explained the complex diagnosis"
      → medical domain + formal register + complex syntax
    """
    def __init__(self, tokenizer, seq_len: int = 512, num_samples: int = 5_000_000):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_samples = num_samples

        # Define composition templates
        self.templates = [
            {
                'text': "The {profession} {manner} {action} the {complexity} {topic}",
                'components': ['domain', 'register', 'syntax'],
                'values': {
                    'profession': ['doctor', 'lawyer', 'teacher', 'engineer'],
                    'manner': ['formally', 'casually', 'briefly', 'thoroughly'],
                    'action': ['explained', 'discussed', 'presented', 'analyzed'],
                    'complexity': ['complex', 'simple', 'technical', 'basic'],
                    'topic': ['diagnosis', 'case', 'concept', 'system']
                }
            },
            {
                'text': "In {time}, {subject} {adverb} {verb} {object} {location}",
                'components': ['temporal', 'manner', 'semantic'],
                'values': {
                    'time': ['the morning', 'the evening', 'summer', 'winter'],
                    'subject': ['the team', 'the group', 'the family', 'the class'],
                    'adverb': ['quickly', 'slowly', 'carefully', 'eagerly'],
                    'verb': ['completed', 'started', 'finished', 'organized'],
                    'object': ['the project', 'the work', 'the task', 'the assignment'],
                    'location': ['at home', 'at work', 'outdoors', 'downtown']
                }
            },
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate a compositional sentence.

        Returns:
            Dictionary with:
            - input_ids: Sentence tokens
            - component_types: Which components are active
            - composition_id: Unique ID
        """
        # Select random template
        template = random.choice(self.templates)

        # Fill in template
        text = template['text']
        for key, values in template['values'].items():
            value = random.choice(values)
            text = text.replace(f"{{{key}}}", value)

        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        tokens = self._pad_or_truncate(tokens)

        # Map components to coordinate dimensions
        component_coords = [hash(c) % 1024 for c in template['components']]

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'component_types': torch.tensor(component_coords, dtype=torch.long),
            'composition_id': idx
        }

    def _pad_or_truncate(self, tokens: List[int]) -> List[int]:
        """Pad or truncate to seq_len"""
        if len(tokens) < self.seq_len:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_len - len(tokens))
        else:
            tokens = tokens[:self.seq_len]
        return tokens


class ConversationalDataset(Dataset):
    """
    Dataset for Phase 3: Conversational Dynamics

    Multi-turn dialogues where coordinates should drift across turns
    while maintaining coherence within a conversation.
    """
    def __init__(
        self,
        tokenizer,
        seq_len: int = 512,
        num_samples: int = 10_000_000,
        turns_per_dialogue: int = 4
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.turns_per_dialogue = turns_per_dialogue

        # Define dialogue templates
        self.dialogue_templates = [
            [
                "Hello! How can I help you today?",
                "I'm looking for information about {topic}.",
                "Of course! Let me explain {topic} for you.",
                "Thank you, that was very helpful!"
            ],
            [
                "What brings you here?",
                "I have a question about {topic}.",
                "Sure, what would you like to know about {topic}?",
                "Can you tell me more about {aspect}?"
            ],
        ]

        self.topics = [
            'machine learning', 'cooking', 'travel', 'health',
            'technology', 'sports', 'music', 'art'
        ]

        self.aspects = [
            'the basics', 'advanced techniques', 'common mistakes',
            'best practices', 'recent developments'
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate a multi-turn dialogue.

        Returns:
            Dictionary with:
            - turn_ids: List of token tensors for each turn
            - dialogue_id: Unique dialogue ID
            - turn_count: Number of turns
        """
        # Select random template
        template = random.choice(self.dialogue_templates)

        # Fill in template
        topic = random.choice(self.topics)
        aspect = random.choice(self.aspects)

        turns = []
        for turn_template in template[:self.turns_per_dialogue]:
            turn = turn_template.replace('{topic}', topic)
            turn = turn.replace('{aspect}', aspect)

            # Tokenize
            tokens = self.tokenizer.encode(turn, add_special_tokens=True)
            tokens = self._pad_or_truncate(tokens)
            turns.append(torch.tensor(tokens, dtype=torch.long))

        # Stack turns
        turn_ids = torch.stack(turns)  # (turns_per_dialogue, seq_len)

        return {
            'turn_ids': turn_ids,
            'dialogue_id': idx,
            'turn_count': self.turns_per_dialogue
        }

    def _pad_or_truncate(self, tokens: List[int]) -> List[int]:
        """Pad or truncate to seq_len"""
        if len(tokens) < self.seq_len:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_len - len(tokens))
        else:
            tokens = tokens[:self.seq_len]
        return tokens


class PrimitiveDataLoader:
    """DataLoader wrapper for Phase 1"""
    @staticmethod
    def create(tokenizer, batch_size: int, seq_len: int, num_workers: int = 4):
        dataset = PrimitiveDataset(tokenizer, seq_len)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )


class CompositionalDataLoader:
    """DataLoader wrapper for Phase 2"""
    @staticmethod
    def create(tokenizer, batch_size: int, seq_len: int, num_workers: int = 4):
        dataset = CompositionalDataset(tokenizer, seq_len)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )


class ConversationalDataLoader:
    """DataLoader wrapper for Phase 3"""
    @staticmethod
    def create(tokenizer, batch_size: int, seq_len: int, num_workers: int = 4):
        dataset = ConversationalDataset(tokenizer, seq_len)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
