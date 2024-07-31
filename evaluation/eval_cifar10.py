import os
import argparse
import pandas as pd

from src.grounddino_evaluator import Cifar10GroundDINOEval


CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def evaluate_cifar10(folder_path, prompts_path, eval_result_path, gdino_result_path,
                     erased_concept=None, save_detections=False, device='cuda'):
    # get evaluator
    evaluator = Cifar10GroundDINOEval(device, save_detections=save_detections)

    # collect image paths
    names = [
        name for name in os.listdir(folder_path)
        if name.endswith('.png') or name.endswith('.jpg')
    ]
    image_paths = [os.path.join(folder_path, name) for name in names]

    # classify and save results
    df = pd.read_csv(prompts_path)
    df['case_number'] = df['case_number'].astype('int')
    casenum2class = {df['case_number'][i]: df['class'][i] for i in range(len(df))}

    # split names '{case_number}_{image_index}.png/jpg' into two lists
    image_cases, image_indices = zip(*[map(int, name.split('.')[0].split('_')) for name in names])
    class_labels = [casenum2class[case] for case in image_cases]

    # run evaluation
    results, addi_info = evaluator.eval(image_paths, class_labels)

    df_results = pd.DataFrame({
        'case_number': image_cases,
        'img_index': image_indices,
        'class_detected': results,
        **addi_info,
    })

    merged_df = pd.merge(df, df_results)
    # move 'img_index' to the second column
    columns = merged_df.columns.tolist()
    columns.remove('img_index')
    columns.insert(1, 'img_index')
    merged_df = merged_df[columns]
    # save csv
    merged_df.to_csv(gdino_result_path)

    # calculate erasing accuracy
    df_results = pd.DataFrame({
        'concept': class_labels,
        'class_detected': results
    })

    # Calculate accuracy for each concept
    accuracy_df = df_results.groupby('concept')['class_detected'].agg(['sum', 'count'])
    accuracy_df['accuracy'] = (accuracy_df['sum'] / accuracy_df['count'] * 100).round(3)
    accuracy_df = accuracy_df.reset_index()

    # Save per-concept accuracy
    accuracy_df[['concept', 'accuracy']].to_csv(eval_result_path, index=False)

    # Calculate in-class and other classes accuracy if erased_concept is provided
    if erased_concept is not None:
        in_class = accuracy_df[accuracy_df['concept'] == erased_concept]
        other_classes = accuracy_df[accuracy_df['concept'] != erased_concept]

        in_cls_acc = in_class['accuracy'].values[0] if not in_class.empty else float('inf')
        other_cls_acc = other_classes['accuracy'].mean() if not other_classes.empty else float('inf')

        with open(eval_result_path, 'a') as f:
            f.write(f'\nIn class accuracy ({erased_concept}): {in_cls_acc:.3f}\n')
            f.write(f'Other classes averaged accuracy: {other_cls_acc:.3f}\n')

    print("===============================")
    print(f"G.DINO's result saved in {gdino_result_path}")
    print(f"Evaluation result saved in {eval_result_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ImageClassification',
        description='Takes the path of images and generates classification results'
    )

    parser.add_argument('--folder_path', help='path to the folder of generated images', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to the prompts file used to generate images', type=str, required=True)
    parser.add_argument('--erased_concept', help='the erased concept of the model', type=str, required=True, choices=CIFAR10_CLASSES)

    parser.add_argument('--save_detections', help='Whether to save the detection results', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    
    # path to save results
    eval_result_path = os.path.join(args.folder_path, 'cifar10_eval_result.csv')
    gdino_result_path = os.path.join(args.folder_path, 'groundingdino_result.csv')

    evaluate_cifar10(
        args.folder_path,
        args.prompts_path,
        erased_concept=args.erased_concept,
        save_detections=args.save_detections,
        eval_result_path=eval_result_path,
        gdino_result_path=gdino_result_path,
        device=args.device,
    )
