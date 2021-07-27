from rpe_prediction.config import SubjectDataIterator, StereoAzureLoader, RPELoader, FusedAzureLoader
from rpe_prediction.processing import segment_1d_joint_on_example, compute_statistics_for_subjects
from rpe_prediction.stereo_cam import StereoAzure
from rpe_prediction.plot import PDFWriter
from os.path import join, isdir

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import argparse
import os

plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.weight"] = 'bold'
plt.rcParams["font.size"] = 22

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="data/raw")
parser.add_argument('--dst_path', type=str, dest='dst_path', default="data/processed")
parser.add_argument('--log_path', type=str, dest='log_path', default="results/segmentation")
parser.add_argument('--show_plots', type=bool, dest='show_plots', default=False)
args = parser.parse_args()

example = np.loadtxt("data/example.np")


def fuse_kinect_data(pdf_file):
    """
    Fuse Kinect data from sub and master camera and calculate features
    @param pdf_file: output name of current pdf file
    """
    pdf_writer = PDFWriter(pdf_file)
    file_iterator = SubjectDataIterator(args.src_path).add_loader(StereoAzureLoader).add_loader(RPELoader)
    sum_repetitions = 0

    for set_data in file_iterator.iterate_over_specific_subjects("4AD6F3"):
        # Create output folder to save averaged skeleton
        dst_path = join(args.dst_path, set_data['subject_name'])
        log_path = join(args.log_path, set_data['subject_name'])
        for cur_path in [dst_path, log_path]:
            if not os.path.exists(cur_path):
                os.makedirs(cur_path)

        sub_path, master_path = set_data['azure']
        azure = StereoAzure(master_path=master_path, sub_path=sub_path)

        # Segment data based on pelvis joint
        repetitions = segment_1d_joint_on_example(joint_data=azure.sub_position['PELVIS (y)'],
                                                  exemplar=example, std_dev_percentage=0.5,
                                                  show=False)
        sum_repetitions += len(repetitions)
        print(f"reps: {len(repetitions)}")

        # Cut Kinect data before first and right after last repetition
        azure.cut_skeleton_data(repetitions[0][0], repetitions[-1][1])
        azure.calculate_affine_transform_based_on_data(show=False)
        print(f"Agreement internal {sub_path}: {azure.check_agreement_of_both_cameras(show=False)}")
        avg_df = azure.fuse_cameras(show=True, pp=pdf_writer)
        avg_df.to_csv(f"{os.path.join(dst_path, str(set_data['nr_set']))}_azure.csv", sep=';', index=False)

        # Save individual repetitions
        for count, (r1, r2) in enumerate(repetitions):
            csv_dir = join(log_path, "csv")
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
            avg_df.loc[r1:r2].to_csv(join(csv_dir, f"{set_data['nr_set']}_{count}_{set_data['rpe']}.csv"),
                                     sep=';',
                                     index=False)

        # Create output folder to save averaged skeleton
        dst_path = join(args.dst_path, set_data['subject_name'])
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        pdf_writer.add_booklet(set_data['subject_name'], set_data['nr_set'], avg_df.columns)

    pdf_writer.close_file()
    print(f'Found in total {sum_repetitions} repetitions.')


def plot_repetition_data(pdf_file):
    """
    Plot the resulting joints into a single
    @param pdf_file: the name of the output PDF file
    @return: None
    """
    pdf_render = PDFWriter(pdf_file)
    subjects = list(filter(lambda x: isdir(x), map(lambda x: join(args.log_path, x), os.listdir(args.log_path))))
    file_iterator = SubjectDataIterator(args.dst_path).add_loader(FusedAzureLoader)
    means, std = compute_statistics_for_subjects(file_iterator.iterate_over_all_subjects())

    # Utils for colorbar
    cmap = plt.cm.get_cmap("jet")
    norm = matplotlib.colors.Normalize(vmin=10, vmax=20)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    def parse_rpe_file(segment_file):
        return int(segment_file.split('_')[2][:2])

    for subject in subjects:
        subject_name = str(subject.split("/")[-1])
        sub_path = join(subject, "csv")
        files = list(filter(lambda f: f.endswith('.csv'), os.listdir(sub_path)))
        files = list(map(lambda x: join(sub_path, x), files))
        files = list(map(lambda f: (pd.read_csv(f, sep=';', index_col=False), parse_rpe_file(f)), files))

        plt.close()
        plt.figure()
        cols = files[0][0].columns
        for col in cols:
            for df, rpe in files:
                df = (df - means[subject_name]) / (std[subject_name] * 3)
                df = df.clip(-1, 1)
                plt.plot(df[col], color=cmap(norm(rpe)))

            plt.xlabel("Frames [1/30s]")
            plt.ylabel("Distance [mm]")
            plt.title(f"{subject_name} {col}")
            plt.colorbar(sm)
            plt.tight_layout()
            pdf_render.save_figure()
            plt.clf()

        pdf_render.add_booklet(subject_name, 0, cols)

    pdf_render.close_file()


def normalize_data_plot(pdf_file):
    """
    Normalize the data by subtracting mean and dividing by std deviation
    @param pdf_file: the current output pdf file
    @return: None
    """
    file_iterator = SubjectDataIterator(args.dst_path).add_loader(FusedAzureLoader)
    means, std = compute_statistics_for_subjects(file_iterator.iterate_over_all_subjects())
    pdf_writer = PDFWriter(pdf_file)

    for set_data in file_iterator.iterate_over_all_subjects():
        subject = set_data['subject_name']
        path = set_data['azure']
        df = pd.read_csv(path, delimiter=';')
        df = (df - means[subject]) / (std[subject] * 3)
        df = df.clip(-1, 1)

        plt.close()
        plt.figure()
        for joint in df.columns:
            plt.plot(df[joint], color="red", label="Sensor")
            plt.xlabel("Frames (30Hz)")
            plt.ylabel("Distance (mm)")
            plt.title(f"{joint.title().replace('_', ' ')}")
            plt.legend()
            plt.tight_layout()
            pdf_writer.save_figure()
            plt.clf()

        pdf_writer.add_booklet(subject, set_data['nr_set'], df.columns)

    pdf_writer.close_file()


if __name__ == '__main__':
    # if args.show_plots:  # Quick fix for running script on server
    import matplotlib
    matplotlib.use("TkAgg")

    fuse_kinect_data(pdf_file='raw_fusion.pdf')
    normalize_data_plot(pdf_file="fusion_norm.pdf")
    plot_repetition_data(pdf_file='fusion_segmented.pdf')
