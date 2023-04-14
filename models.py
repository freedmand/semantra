import torch
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModel
import tiktoken
import openai
from dotenv import load_dotenv
import os

load_dotenv()

if "OPENAI_API_KEY" in os.environ:
    openai.api_key = os.getenv("OPENAI_API_KEY")


minilm_model_name = "sentence-transformers/all-MiniLM-L6-v2"
mpnet_model_name = "sentence-transformers/all-mpnet-base-v2"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class BaseModel(ABC):
    @abstractmethod
    def get_tokens(self, text: str):
        ...

    @abstractmethod
    def get_token_length(self, tokens) -> int:
        ...

    @abstractmethod
    def get_text_chunks(self, text: str, tokens) -> "list[str]":
        ...

    @abstractmethod
    def embed(self, tokens, offsets, is_query: bool = False) -> "list[list[float]]":
        ...

    def embed_query(self, query: str) -> "list[float]":
        tokens = self.get_tokens(query)
        return self.embed(tokens, [(0, self.get_token_length(tokens))], True)[0]


class OpenAIModel(BaseModel):
    def __init__(
        self, model_name="text-embedding-ada-002", tokenizer_name="cl100k_base"
    ):
        self.model_name = model_name
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)

    def get_tokens(self, text: str):
        return self.tokenizer.encode(text)

    def get_token_length(self, tokens) -> int:
        return len(tokens)

    def get_text_chunks(self, _: str, tokens) -> "list[str]":
        return [self.tokenizer.decode([token]) for token in tokens]

    def embed(self, tokens, offsets) -> "list[list[float]]":
        texts = [tokens[i:j] for i, j in offsets]
        response = openai.Embedding.create(model=self.model_name, input=texts)
        return [data["embedding"] for data in response["data"]]


def zero_if_none(x):
    return 0 if x is None else x


class TransformerModel(BaseModel):
    def __init__(
        self,
        model_name,
        doc_token_pre=None,
        doc_token_post=None,
        query_token_pre=None,
        query_token_post=None,
        cuda=None,
    ):
        if cuda is None:
            cuda = torch.cuda.is_available()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Get tokens
        self.doc_token_pre = (
            self.tokenizer.encode(doc_token_pre, add_special_tokens=False)
            if doc_token_pre
            else []
        )
        self.doc_token_post = (
            self.tokenizer.encode(doc_token_post, add_special_tokens=False)
            if doc_token_post
            else []
        )
        self.query_token_pre = (
            self.tokenizer.encode(query_token_pre, add_special_tokens=False)
            if query_token_pre
            else []
        )
        self.query_token_post = (
            self.tokenizer.encode(query_token_post, add_special_tokens=False)
            if query_token_post
            else []
        )

        self.cuda = cuda
        if self.cuda:
            self.model = self.model.cuda()

    def get_tokens(self, text: str):
        return self.tokenizer(
            text, return_offsets_mapping=True, verbose=False, return_tensors="pt"
        )

    def get_token_length(self, tokens) -> int:
        return len(tokens["input_ids"][0])

    def get_text_chunks(self, text: str, tokens) -> "list[str]":
        offsets = tokens["offset_mapping"][0]
        chunks = []
        prev_i = None
        prev_j = None
        for i, j in offsets:
            new_i = prev_j if i == j else i
            if prev_i is not None:
                chunks.append(text[prev_i:new_i])
            if prev_i is None:
                prev_i = 0
            elif new_i > prev_i:
                prev_i = new_i
            if prev_j is None:
                prev_j = j
            elif j > prev_j:
                prev_j = j
        chunks.append(text[0 if prev_i is None else prev_i :])
        return chunks

    def normalize_input_ids(self, input_ids, is_query):
        if is_query:
            return torch.cat(
                [
                    torch.tensor(self.query_token_pre),
                    input_ids,
                    torch.tensor(self.query_token_post),
                ]
            )
        else:
            return torch.cat(
                [
                    torch.tensor(self.doc_token_pre),
                    input_ids,
                    torch.tensor(self.doc_token_post),
                ]
            )

    def normalize_attention_mask(self, attention_mask, is_query):
        if is_query:
            return torch.cat(
                [
                    torch.ones(len(self.query_token_pre)),
                    attention_mask,
                    torch.ones(len(self.query_token_post)),
                ]
            )
        else:
            return torch.cat(
                [
                    torch.ones(len(self.doc_token_pre)),
                    attention_mask,
                    torch.ones(len(self.doc_token_post)),
                ]
            )

    def embed(self, tokens, offsets, is_query=False) -> "list[list[float]]":
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [
                self.normalize_input_ids(
                    tokens["input_ids"][0].index_select(0, torch.tensor(range(i, j))),
                    is_query,
                )
                for i, j in offsets
            ],
            batch_first=True,
            padding_value=zero_if_none(self.tokenizer.pad_token_id),
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [
                self.normalize_attention_mask(
                    tokens["attention_mask"][0].index_select(
                        0, torch.tensor(range(i, j))
                    ),
                    is_query,
                )
                for i, j in offsets
            ],
            batch_first=True,
            padding_value=0,
        )
        if self.cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        with torch.no_grad():
            model_output = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )
        return mean_pooling(model_output, attention_mask).tolist()


models = {
    "openai": {
        "params": {"type": "openai", "model_name": "text-embedding-ada-002"},
        "num_dimensions": 1536,
        "cost_per_token": 0.0004 / 1000,
        "window_token_limit": 7900,  # technically 8192 but sometimes tiktoken gives an inaccurate count
        "pool_size": 50000,
        "pool_count": 2000,
        "get_model": lambda: OpenAIModel(
            model_name="text-embedding-ada-002", tokenizer_name="cl100k_base"
        ),
    },
    "minilm": {
        "params": {"type": "transformers", "model_name": minilm_model_name},
        "num_dimensions": 384,
        "cost_per_token": None,
        "window_token_limit": 128,
        "pool_size": 50000,
        "get_model": lambda: TransformerModel(model_name=minilm_model_name),
    },
    "mpnet": {
        "params": {"type": "transformers", "model_name": mpnet_model_name},
        "num_dimensions": 768,
        "cost_per_token": None,
        "window_token_limit": 128,
        "pool_size": 15000,
        "get_model": lambda: TransformerModel(model_name=mpnet_model_name),
    },
}
