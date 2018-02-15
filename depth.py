import argparse
import os
from tensorflowFiles.GeneratePointCloud import generate
from tensorflowFiles.predict import predict



def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    pred = predict(args.model_path, args.image_paths)
    generate(pred)


    os._exit(0)


if __name__ == '__main__':
    main()
