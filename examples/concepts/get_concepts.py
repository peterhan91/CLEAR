import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json
from vllm import LLM
from tqdm import tqdm
from vllm.sampling_params import SamplingParams

from data_process import get_cxr_paths_list

if __name__ == "__main__":
    cxr_paths = get_cxr_paths_list('/home/than/DeepLearning/CheXzero/data/cxr_paths.csv')
    txt_folder = '/home/than/physionet.org/files/mimic-cxr-jpg/2.0.0/reports/files/'

    model_name = "mistralai/Ministral-8B-Instruct-2410"
    sampling_params = SamplingParams(max_tokens=8192, temperature=0, top_k=-1)
    llm = LLM(model=model_name, tokenizer_mode="mistral", config_format="mistral", load_format="mistral")

    results = []
    for cxr_path in tqdm(cxr_paths):
        tokens = cxr_path.split('/')
        study_num = tokens[-2]
        patient_num = tokens[-3]
        patient_group = tokens[-4]
        filename = study_num + '.txt'
        txt_report = txt_folder + patient_group + '/' + patient_num + '/' + study_num + '.txt'

        f = open(txt_report, 'r')
        s = f.read()
        s_split = s.split()
        s = ' '.join(s_split)

        prompt = f"""
You are a helpful assistant. Please respond in valid JSON only.
Question: What are the descriptive observations in the report? FINAL REPORT EXAMINATION: CHEST (PORTABLE AP)CHEST (PORTABLE AP)i INDICATION: ___ year old woman with SOB // eval for sign of PNA, effusion, pulm vascular congestion COMPARISON: Chest radiographs ___ IMPRESSION: Mild pulmonary edema and small to moderate bilateral pleural effusions all improved since ___ following extubation. Heart size normal. No pneumothorax. Left subclavian line ends in the SVC 
Respond with a JSON object with this structure:
{{
  "observations": ["Mild pulmonary edema", "Small to moderate bilateral pleural effusions", "Heart size normal", "No pneumothorax", "Left subclavian line ends in the SVC"],    
}}
Question: What are the descriptive observations in the report? FINAL REPORT EXAMINATION: CHEST (AP AND LAT) INDICATION: History: ___M with cough, fever TECHNIQUE: Upright AP and lateral views of the chest COMPARISON: Chest CT ___ FINDINGS: Cardiac silhouette size is normal. Mediastinal and hilar contours are unremarkable. Lungs are hyperinflated. No pulmonary edema is seen. Ill-defined patchy opacities are noted in the left lung base, concerning for pneumonia. Blunting of the costophrenic angles bilaterally suggests trace bilateral pleural effusions, more pronounced on the left. No pneumothorax is present. No acute osseous abnormalities detected. Multiple clips are again noted at the gastroesophageal junction and in the right upper quadrant of the abdomen. IMPRESSION: Patchy ill-defined left basilar opacity concerning for pneumonia. Small bilateral pleural effusions.
Respond with a JSON object with this structure:
{{
  "observations": ["Cardiac silhouette size is normal", "Mediastinal and hilar contours are unremarkable", "Lungs are hyperinflated", "No pulmonary edema is seen", "Ill-defined patchy opacities in the left lung base", "Blunting of the costophrenic angles bilaterally", "No pneumothorax", "No acute osseous abnormalities", "Small bilateral pleural effusions"],    
}}
Question: What are the descriptive observations in the report? {s}
Respond with a JSON object with this structure:
"""
        messages = [
            {
                "role": "user",
                "content": prompt
            },
        ]

        outputs = llm.chat(messages, 
                        sampling_params=sampling_params)

        result = {
            "id": txt_report,
            "model_output": outputs[0].outputs[0].text,
        }
        results.append(result)
        f.close()
        
        with open(os.path.join('data', 'mimic_concepts.json'), 'w') as f:
            json.dump(results, f, indent=4)
        