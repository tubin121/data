import nltk
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.rouge import Rouge

# 示例文本
golden_sense = "An enclosure to restrain cattle, horses, etc."
answer = "A sailor."
definitions = "A lascar is a man who is stateless and not very good in a."

# 分词
golden_tokens = nltk.word_tokenize(golden_sense)
answer_tokens = nltk.word_tokenize(answer)
definitions_tokens = nltk.word_tokenize(definitions)

# 计算BLEU分数
bleu_score_answer = sentence_bleu([golden_tokens], answer_tokens)
bleu_score_definitions = sentence_bleu([golden_tokens], definitions_tokens)

# 计算ROUGE分数
rouge = Rouge()
rouge_score_answer = rouge.get_scores(answer, golden_sense)[0]['rouge-1']['f']
rouge_score_definitions = rouge.get_scores(definitions, golden_sense)[0]['rouge-1']['f']

# 计算BERT-Score
bert_score_answer = score([answer], [golden_sense], lang="en")[2].item()
bert_score_definitions = score([definitions], [golden_sense], lang="en")[2].item()

# 打印分数
print("BLEU Score (answer):", bleu_score_answer)
print("BLEU Score (definitions):", bleu_score_definitions)
print("ROUGE Score (answer):", rouge_score_answer)
print("ROUGE Score (definitions):", rouge_score_definitions)
print("BERT-Score (answer):", bert_score_answer)
print("BERT-Score (definitions):", bert_score_definitions)
