import json
import numpy as np

def few_shot_evaluation(data):

	results = {}
	overall_f1_scores = {}

	for model in data:
		all_f1_scores = []
		results[model] = {}
		for n_way_k_shot_results in data[model]:
			for key, results_list in n_way_k_shot_results.items():
				f1_scores = []
				for result in results_list:
					f1_scores.append(result['f1'])
				
				f1_mean = np.mean(f1_scores)
				f1_std = np.std(f1_scores)
				all_f1_scores.extend(f1_scores)

				results[model][key] = {'mean': f1_mean, 'std': f1_std}
		
		overall_f1_score = np.mean(all_f1_scores)
		overall_f1_scores[model] = overall_f1_score

	return results, overall_f1_scores

if __name__ == '__main__':
	with open('few_shot_results.json') as f:
		data = json.load(f)

	results, overall_f1_scores = few_shot_evaluation(data)
	print(results)
	print(overall_f1_scores)