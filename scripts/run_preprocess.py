import argparse
import os
import pandas as pd
from pathlib import Path
from data_process import get_cxr_paths_list, img_to_hdf5, get_cxr_path_csv, write_report_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_out_path', type=str, default='data/cxr_paths.csv', help="Directory to save paths to all chest x-ray images in dataset.")
    parser.add_argument('--cxr_out_path', type=str, default='data/cxr.h5', help="Directory to save processed chest x-ray image data.")
    parser.add_argument('--dataset_type', type=str, default='mimic', choices=['mimic', 'chexpert-test', 
                                                                              'chexpert-plus', 'rexgradient', 
                                                                              'chexpert-valid', 'padchest-test',
                                                                              'vindrcxr-test', 'vindrpcxr-test',
                                                                              'indiana-test', 'vindrcxr-train',
                                                                              'brax'], 
                                                                              help="Type of dataset to pre-process")
    parser.add_argument('--mimic_impressions_path', default='data/mimic_impressions.csv', help="Directory to save extracted impressions from radiology reports.")
    parser.add_argument('--chest_x_ray_path', default='/deep/group/data/mimic-cxr/mimic-cxr-jpg/2.0.0/files', help="Directory where chest x-ray image data is stored. This should point to the files folder from the MIMIC chest x-ray dataset.")
    parser.add_argument('--radiology_reports_path', default='/deep/group/data/med-data/files/', help="Directory radiology reports are stored. This should point to the files folder from the MIMIC radiology reports dataset.")
    parser.add_argument('--resolution', type=int, default=448, help="Resolution of chest x-ray images.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.dataset_type == "mimic":
        # Write Chest X-ray Image HDF5 File
        get_cxr_path_csv(args.csv_out_path, args.chest_x_ray_path)
        cxr_paths = get_cxr_paths_list(args.csv_out_path)
        img_to_hdf5(cxr_paths, args.cxr_out_path, resolution=args.resolution)

        #Write CSV File Containing Impressions for each Chest X-ray
        write_report_csv(cxr_paths, args.radiology_reports_path, args.mimic_impressions_path)
    
    elif args.dataset_type == "chexpert-plus":
        df = pd.read_csv("data/chexpert_train.csv")
        df['path_to_image'] = df['path_to_image'].apply(lambda x: os.path.join(args.chest_x_ray_path, x))
        cxr_paths = df['path_to_image'].tolist()
        impressions = df['impression'].tolist()
        
        df_paths = pd.DataFrame({'Path': cxr_paths})
        df_paths.to_csv(args.csv_out_path, index=False)
        
        img_to_hdf5(cxr_paths, args.cxr_out_path, resolution=args.resolution)
        
        df_impressions = pd.DataFrame({'filename': [Path(p).name for p in cxr_paths], 'impression': impressions})
        df_impressions.to_csv(args.mimic_impressions_path, index=False)
    
    elif args.dataset_type == "rexgradient":
        df = pd.read_csv("data/rexgradient_all.csv")
        cxr_paths = df['path_to_image'].tolist()
        impressions = df['Impression'].tolist()
        
        df_paths = pd.DataFrame({'Path': cxr_paths})
        df_paths.to_csv(args.csv_out_path, index=False)
        
        img_to_hdf5(cxr_paths, args.cxr_out_path, resolution=args.resolution)
        
        df_impressions = pd.DataFrame({'filename': [Path(p).name for p in cxr_paths], 'impression': impressions})
        df_impressions.to_csv(args.mimic_impressions_path, index=False)
    
    elif args.dataset_type == "chexpert-valid":
        df_chexpert = pd.read_csv("data/chexpert_valid.csv")
        cxr_paths = df_chexpert['Path'].tolist()
        cxr_paths = [os.path.join("/home/than/Datasets/stanford_mit_chest/", p) for p in cxr_paths]
        assert(len(cxr_paths) == 200)
       
        img_to_hdf5(cxr_paths, args.cxr_out_path, resolution=args.resolution)

    elif args.dataset_type == "chexpert-test": 
        # Get all test paths based on cxr dir
        # cxr_dir = Path(args.chest_x_ray_path)
        # cxr_paths = list(cxr_dir.rglob("*.jpg"))
        # cxr_paths = list(filter(lambda x: "view1" in str(x), cxr_paths)) # filter only first frontal views 
        # cxr_paths = sorted(cxr_paths) # sort to align with groundtruth
        df_chexpert = pd.read_csv("data/chexpert_test.csv")
        cxr_paths = df_chexpert['Path'].tolist()
        cxr_paths = [os.path.join("/home/than/Datasets/stanford_mit_chest/", p) for p in cxr_paths]
        assert(len(cxr_paths) == 500)
       
        img_to_hdf5(cxr_paths, args.cxr_out_path, resolution=args.resolution)


    elif args.dataset_type == "padchest-test":
        df_padchest = pd.read_csv("data/padchest_test.csv")
        df_padchest = df_padchest[df_padchest['is_test'] == True]
        cxr_paths = df_padchest['ImageID'].tolist() # sort to align with groundtruth
        cxr_paths = [os.path.join("/home/than/padchest/images/", p) for p in cxr_paths]
        assert len(cxr_paths) == 7943, f"Expected 7943 images, but got {len(cxr_paths)}"
       
        img_to_hdf5(cxr_paths, args.cxr_out_path, resolution=args.resolution)     

    elif args.dataset_type == "vindrcxr-train":
        df_vindrcxr = pd.read_csv("data/vindrcxr_train.csv")
        cxr_paths = df_vindrcxr['image_id'].tolist()
        cxr_paths = [os.path.join("/home/than/Datasets/CXR_VQA/OD/physionet.org/files/vindr-cxr/1.0.0/train_png/", p + '.png') for p in cxr_paths]
        assert(len(cxr_paths) == 15000)

        img_to_hdf5(cxr_paths, args.cxr_out_path, resolution=args.resolution)

    elif args.dataset_type == "vindrcxr-test":
        df_vindrcxr = pd.read_csv("data/vindrcxr_test.csv")
        cxr_paths = df_vindrcxr['image_id'].tolist()
        cxr_paths = [os.path.join("/home/than/Datasets/CXR_VQA/OD/physionet.org/files/vindr-cxr/1.0.0/test_png/", p + '.png') for p in cxr_paths]
        assert(len(cxr_paths) == 3000)

        img_to_hdf5(cxr_paths, args.cxr_out_path, resolution=args.resolution)

    elif args.dataset_type == "vindrpcxr-test":
        df_vindrpcxr = pd.read_csv("data/vindrpcxr_test.csv")
        cxr_paths = df_vindrpcxr['image_id'].tolist()
        cxr_paths = [os.path.join("/home/than/Datasets/CXR_VQA/OD/physionet.org/files/vindr-pcxr/1.0.0/test_png/", p + '.png') for p in cxr_paths]
        assert(len(cxr_paths) == 1397)

        img_to_hdf5(cxr_paths, args.cxr_out_path, resolution=args.resolution)

    elif args.dataset_type == "indiana-test":
        df_indiana = pd.read_csv("data/indiana_test.csv")
        cxr_paths = df_indiana['filename'].tolist()
        cxr_paths = [os.path.join("/home/than/Datasets/IU_XRay/images/images_normalized/", p) for p in cxr_paths]
        assert(len(cxr_paths) == 7466)

        img_to_hdf5(cxr_paths, args.cxr_out_path, resolution=args.resolution)

    elif args.dataset_type == "brax":
        df_brax = pd.read_csv("data/brax.csv")
        cxr_paths = df_brax['PngPath'].tolist()
        cxr_paths = [os.path.join("/home/than/Datasets/BRAX", p) for p in cxr_paths]
        assert(len(cxr_paths) == 40922) 

        img_to_hdf5(cxr_paths, args.cxr_out_path, resolution=args.resolution)   
