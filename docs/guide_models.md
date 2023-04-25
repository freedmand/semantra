# Using other models with Semantra

Semantra comes with a few preset models along with the ability to run almost any custom [Hugging Face](https://huggingface.co/) [transformers](https://huggingface.co/docs/transformers/index) model. If your computer has a compatible GPU (graphics processing unit, often found in video cards), Semantra will leverage it via [PyTorch](https://pytorch.org/) to dramatically speed up computation.

## Using preset models

The models Semantra comes with out-of-the-box include:

- **openai**: See [the OpenAI guide](guide_openai.md)
- **minilm**: A [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) model that's very quick and lean. It corresponds to the transformers model `sentence-transformers/all-MiniLM-L6-v2`
- **mpnet**: The default model that Semantra uses. It's the [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) that achieves the best accuracy but is still relatively quick. It corresponds to the transformers model `sentence-transformers/all-mpnet-base-v2`
- **sgpt**: A very accurate and decently quick model from [Sentence embeddings for semantic search](https://github.com/Muennighoff/sgpt). The model here is the transformers model `Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit` and is asymmetric, meaning queries and documents are tokenized slightly differently.
- **sgpt-1.3B**: The 1.3 billion parameter version of `sgpt`, corresponding to transformers model `Muennighoff/SGPT-1.3B-weightedmean-msmarco-specb-bitfit`

To use a preset model, specify the `--model` flag with the model name, e.g.

```sh
semantra --model sgpt <documents>
```

## When to use the OpenAI model versus transformers models

If the nature of your work is sensitive in any way, you may not want anything going over the network to external services. The OpenAI model requires sending data to them, which may be a dealbreaker for use cases such as investigating a leaked collection of documents or analyzing documents internal to your company.

OpenAI's model is very powerful and able to encode nuanced semantic meanings but in practice it is not significantly more useful than models that can be run locally in combination with Semantra's advanced features. OpenAI's model is fast at processing large documents and it offloads processing to their servers, so if your computer is slow or you have an enormous amount of documents and want to get through them quickly, OpenAI's model may work well.

Lastly, OpenAI's model costs money to use. The cost is pretty small but adds up if you are processing large collections of documents. See [the OpenAI guide](guide_openai.md) for more information.

## Using custom models

Many models in the [Hugging Face Hub](https://huggingface.co/models) can be used by Semantra with the `--transformer-model` flag followed by the model name.

For instance, any [pretrained Sentence Transformers model](https://huggingface.co/models?library=sentence-transformers) can be used via

```sh
semantra --transformer-model sentence-transformers/all-distilroberta-v1 <filenames>
```

### Specifying custom tokens

Some models require queries and documents to have special tokens prepended before and after. For instance, the **sgpt** models discussed above have queries surrounded square braces (`[]`) and documents surrounded by curly braces (`{}`).

The `--query-token-pre`, `--query-token-post`, `--doc-token-pre`, and `--doc-token-post` control the tokens inserted before and after queries and documents (these parameters default to `None`, which will not add anything).

For example, to use the `sgpt-5.8B` model (which is massive and can only be run on large GPUs), you could run the following:

```sh
semantra --transformer-model Muennighoff/SGPT-1.3B-weightedmean-msmarco-specb-bitfit --query_token_pre='[' --query_token_post=']' --doc_token_pre='{' --doc_token_post='}' <filenames>
```
