python -m kvquant.demo_llm --model distilgpt2 --prompt "What is the capital of France?" --max-new-tokens 20

---- Interactive generation ------------------------------------
  Model  : distilgpt2
  Prompt : 'What is the capital of France?'
  Mode   : Q/A format (base model)

The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
  Unquant: The French government has been very active in this area. It's a great place to live, and

  2-bit  : Thessi. I think in a very short period of time, there are many different types of
  3-bit  : The French government has a very strong and powerful political base. It's not just an economic powerhouse,
  4-bit  : The French government has been very active in this area. It's a great place to live, and
----------------------------------------------------------------

#################################################################################
python -m kvquant.demo_llm --model Qwen/Qwen2.5-1.5B-Instruct --prompt "What is the capital of France?" --max-new-tokens 20

---- Interactive generation ------------------------------------
  Model  : Qwen/Qwen2.5-1.5B-Instruct
  Prompt : 'What is the capital of France?'
  Mode   : chat template

The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
  Unquant: The capital city of France is Paris. Paris has been an important cultural and political center since ancient times

  2-bit  : The ( The Pincerely: : from this time's country..e/ Capitalcap.
  3-bit  : The that is " I amared. U asking for French it's Paris and Capital. is
  4-bit  : The not French largest of france Sur Francaceu in France's answer the cap France Capital France for
----------------------------------------------------------------

#################################################################################
python -m kvquant.demo_llm --model Qwen/Qwen3.5-0.8B --prompt "What is the capital of France?" --max-new-tokens 20

---- Interactive generation ------------------------------------
  Model  : Qwen/Qwen3.5-0.8B
  Prompt : 'What is the capital of France?'
  Mode   : chat template

The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:248044 for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
  Unquant: The capital city of **France** (French: *la capitale de la République française*) is **

  2-bit  : The capitals are: - **London**: The UK's largest city, home to many institutions and
  3-bit  : The **capital** (or seat) of France, where its government and legislature are located, is
  4-bit  : The **Paris** (or *Parys*) city in western Europe and central Asia. It
----------------------------------------------------------------

#################################################################################
python -m kvquant.demo_llm --model Qwen/Qwen2.5-1.5B-Instruct --prompt "What is the capital of France?" --max-new-tokens 20 --correction-rank 4

---- Interactive generation ------------------------------------
  Model  : Qwen/Qwen2.5-1.5B-Instruct
  Prompt : 'What is the capital of France?'
  Mode   : chat template

The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
  Unquant: The capital city of France is Paris. Paris has been an important cultural and political center since ancient times

  2-bit  : The answer to this question would be: Paris, which has been used as a word in various contexts
  3-bit  : The answer to for this question: "France" . Capital of your country as ? state.
  4-bit  : The capital of France is is is: it Capital of France is is is Paris.
----------------------------------------------------------------


#################################################################################
python -m kvquant.demo_llm --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "How to bake pizza?" --max-new-tokens 80 --correction-rank 4

---- Interactive generation ------------------------------------
  Model  : TinyLlama/TinyLlama-1.1B-Chat-v1.0
  Prompt : 'How to bake pizza?'
  Mode   : chat template

  Unquant: Sure, here's a step-by-step guide on how to make homemade pizzas: 1. Preheat your oven and prepare the dough according to package instructions (usually around room temperature). You can use store bought or homemade crust recipes depending upon what you have available in your kitchen. If using premade breadcrumbs

  2-bit  : Sure, here's a step-by-step guide on how you can make delicious and easy homemade pizzas using the following ingredients: 1. Preheat your oven at around temperature of about (205°F) or higher for best results! You want it hot enough that when you touch it with bare hands without gloves - they
  3-bit  : Sure, here's a step-by-step guide on how to make homemade breaded and fried chicken wings: 1. Preheat your oven at the highest temperature setting (usually around or above) for about an hour before starting with this recipe so that you can get it ready in time when making these delicious crispy treats!
  4-bit  : Sure, here's a step-by-step guide on how you can make homemade crust for your favorite Margherita or pepperoni pizzas: 1. Preheat the oven at its highest temperature (usually around about375°F/190C). This will help ensure that both sides of each slice are golden brown and
----------------------------------------------------------------