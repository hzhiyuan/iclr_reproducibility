import re
import json
import numpy as np


class ActionEvaluation():

    split_info_path = {
        "domain": "xxx",
        "task": "xxx",
        "website": "xxx"
    }

    def __init__(self):
        self.split_domin_images = self.get_split_info(self.split_info_path["domain"])
        self.split_task_images = self.get_split_info(self.split_info_path["task"])
        self.split_website_images = self.get_split_info(self.split_info_path["website"])

    def get_split_info(self, split_info_path):
        image_path_set = set()
        with open(split_info_path, 'r') as file:
            for line in file:
                image_path_set.add(json.loads(line)["image"])
        return image_path_set

    def parse_output(self, str_) -> list:
        str_ = str_.strip()
        match = re.match(r'^(.*?)\(\[(\d{3}), (\d{3}), (\d{3}), (\d{3})\], \"(.*?)\"\)$', str_)
        if not match:
            return {
                'action': "",
                'location': [-1, -1, -1, -1],
                'text': ""
            }
        
        group = match.groups()
        action, cx, cy, w, h, text = group

        cx = int(cx)
        cy = int(cx)
        w = int(w)
        h = int(h)

        return {
            'action': action,
            'location': [cx, cy, w, h],
            'text': text
        }
    
    # Edit from Mind2Web
    def calculate_f1(self, predictions, annotations):
        f1_list = []
        for prediction, annotation in zip(predictions, annotations):
            pred = prediction['action'].strip()
            ann = annotation['action'].strip()

            if pred != 'CLICK':
                pred += ' ' + prediction['text'].strip()
                ann += ' ' + annotation['text'].strip()

            pred = set(pred.strip().split())
            ann = set(ann.strip().split())

            if len(pred) == 0 and len(ann) == 0:
                f1_list.append(1)
                continue
            if len(pred) == 0 or len(ann) == 0:
                f1_list.append(0)
                continue

            tp = len(pred & ann)
            fp = len(pred - ann)
            fn = len(ann - pred)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

            if precision == 0 or recall == 0:
                f1_list.append(0)
                continue
            
            f1_list.append(2 * precision * recall / (precision + recall))

        return np.mean(f1_list), np.array(f1_list)
    
    def get_metrix(self, predictions, annotations):
        action_f1, f1_list = self.calculate_f1(predictions, annotations)

        pred_act = np.array([_['action'] for _ in predictions])
        pred_loc = np.array([_['location'] for _ in predictions])

        ann_act = np.array([_['action'] for _ in annotations])
        ann_loc = np.array([_['location'] for _ in annotations])

        # ann_loc[:, 2: 4] 有极少量标注错误的数据，长或宽为0，警告可以忽略
        is_inside_gt = (np.abs(pred_loc[:, 0: 2] - ann_loc[:, 0: 2]) / ann_loc[:, 2: 4]) < 0.5
        loc_right = is_inside_gt[:, 0] * is_inside_gt[:, 1]
        action_right = pred_act == ann_act

        loc_acc = np.mean(loc_right)
        action_acc = np.mean(action_right)

        step_right = loc_right * (f1_list == 1.)
        step_rate = np.mean(step_right)

        return {
            "loc_acc": loc_acc,
            "action_f1": action_f1,
            "action_acc": action_acc,
            "step_rate": step_rate
        }

    def get_domain_metrix(self, predictions, annotations, domain_list):
        predictions_task = [_ for _, domain in zip(predictions, domain_list) if domain == 'task']
        annotations_task = [_ for _, domain in zip(annotations, domain_list) if domain == 'task']
        res_task = self.get_metrix(predictions_task, annotations_task)

        predictions_website = [_ for _, domain in zip(predictions, domain_list) if domain == 'website']
        annotations_website = [_ for _, domain in zip(annotations, domain_list) if domain == 'website']
        res_website = self.get_metrix(predictions_website, annotations_website)

        predictions_domain = [_ for _, domain in zip(predictions, domain_list) if domain == 'domain']
        annotations_domain = [_ for _, domain in zip(annotations, domain_list) if domain == 'domain']
        res_domain = self.get_metrix(predictions_domain, annotations_domain)

        return {
            "task": res_task,
            "website": res_website,
            "domain": res_domain
        }

    def eval_pred_list(self, output_list):
        predictions = []
        annotations = []
        domain_list = []

        for output_str in output_list:
            output = json.loads(output_str.strip())

            image_path = output["image_path"]
            prediction = output['prediction']
            annotation = output['annotation']

            if image_path in self.split_domin_images:
                domain_list.append('domain')
            elif image_path in self.split_task_images:
                domain_list.append('task')
            elif image_path in self.split_website_images:
                domain_list.append('website')
            else:
                raise ValueError(f"Unknown domain for this image: {image_path}")

            prediction_parsed = self.parse_output(prediction)
            annotation_parsed = self.parse_output(annotation)

            predictions.append(prediction_parsed)
            annotations.append(annotation_parsed)

        results_all = self.get_metrix(predictions, annotations)
        results_domain = self.get_domain_metrix(predictions, annotations, domain_list)

        return {
            "result_all": results_all,
            "result_domain": results_domain
        }
    
    def eval_during_training(self, outputs):
        predictions = []
        annotations = []

        for prediction, annotation in zip(outputs):
            prediction_parsed = self.parse_output(prediction)
            annotation_parsed = self.parse_output(annotation)

            predictions.append(prediction_parsed)
            annotations.append(annotation_parsed)

        results = self.get_metrix(predictions, annotations)

        return results


class ActionEvaluationBlock(ActionEvaluation):

    # abstract object
    split_info_path = {}
        
    def __init__(self, block_num_w: int, block_num_h: int, block_w=1000, block_h=1000):
        super().__init__()
        self.block_num_w = block_num_w
        self.block_num_h = block_num_h
        self.block_w = block_w
        self.block_h = block_h

    def parse_output(self, str_) -> list:
        str_ = str_.strip()
        match = re.match(r'^(.*?)\(\[(\d{1}), (\d{3}), (\d{3}), (\d{3}), (\d{3})\], \"(.*?)\"\)$', str_)
        if not match:
            return {
                'action': "",
                'location': [-1, -1, -1, -1],
                'text': ""
            }
        
        group = match.groups()
        action, block_idx, cx, cy, w, h, text = group

        block_idx = int(block_idx)
        cx = int(cx)
        cy = int(cy)
        w = int(w)
        h = int(h)

        cx += (block_idx % self.block_num_w) * self.block_w
        cy += (block_idx // self.block_num_w) * self.block_h
        cx /= self.block_num_w
        cy /= self.block_num_h

        # Mind2Web 训练集的 wh 被 block_w/h 归一化，但测试集没有，因此不需要反变换
        # w /= self.block_num_w
        # h /= self.block_num_h

        return {
            'action': action,
            'location': [cx, cy, w, h],
            'text': text
        }


class ActionEvaluatorBlock32(ActionEvaluationBlock):

    split_info_path = {
        "domain": "xxx",
        "task": "xxx",
        "website": "xxx"
    }

    def __init__(self):
        super().__init__(block_num_w=3, block_num_h=2)


if __name__ == '__main__':
    res_path = "res_path"

    evaluator = ActionEvaluatorBlock32()

    results = open(res_path, encoding='utf-8').readlines()
    accuracy = evaluator.eval_pred_list(results)
    print(json.dumps(accuracy, indent=4))
