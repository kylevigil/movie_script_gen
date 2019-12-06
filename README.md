# Movie Script Language Models
## Kyle Vigil
## CSE 256: Statistical Natural Language Processing

### Files:
* continuous_gen.py: Prompts the user for priming text as well as movie genre then continually generates script
* hp_randgen.py: Plays a game where the user has to determine if the section of Harry Potter script is generated and fake or real and from the actual movies.
* run_lm_finetuning.py: Script used for training. This is a modified version of https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py

* Model_Performance.ipynb: Used to evaluate training performance for model selection
* Decompose_LM_Finetune.ipynb: Used to modify the huggingface run_lm_finetune.py script
* webscrape.ipynb: Used to webscrape movie script dataset from imsdb.com

### Training Commands:
General movie script by genre: python run_lm_finetuning.py --output_dir=genre_output --model_type=gpt2 --model_name_or_path=gpt2 --per_gpu_train_batch_size=2 --logging_steps=5000 --save_steps=5000 --do_train --train_data_file=../data/train.pkl --do_eval --eval_data_file=../data/test.pkl --eval_all_checkpoints

Harry Potter specific: python run_lm_finetuning.py --output_dir=hp_output --model_type=gpt2 --model_name_or_path=gpt2 --per_gpu_train_batch_size=2 --logging_steps=5000 --save_steps=5000 --do_train --train_data_file=../data/train_hp.pkl --do_eval --eval_data_file=../data/test_hp.pkl --eval_all_checkpoints

### Other Commands:
Generate a script: python continuous_gen.py

Test your Harry Potter knowledge: python hp_randgen.py

