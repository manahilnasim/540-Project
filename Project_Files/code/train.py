import argparse
import os
import shutil
import torch
from ultralytics import YOLO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data.yaml')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt')
    parser.add_argument('--saved-model-weights', type=str, default='model.pt')
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    args, _ = parser.parse_known_args()

    model = YOLO(args.yolo_model)
    results = model.train(data=args.data, epochs=args.epochs, batch=args.batch)

    path_best_model_source = f"{settings['runs_dir']}/detect/train/weights/best.pt"
    path_best_model_dest = os.path.join(args.model_dir, args.saved_model_weights)
    shutil.copy(path_best_model_source, path_best_model_dest)

    print('EVALUATING MODEL ON TEST DATASET...')
    model_val = YOLO(path_best_model_dest)
    metrics = model_val.val(data=args.data, split='test')
    
    print('-------------')
    print('MODEL EVALUATION METRIC:')
    print('mAP50:', round(metrics.box.map50, 4))
    print('-------------')
