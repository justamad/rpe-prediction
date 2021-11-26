from src.camera import AzureKinect, StereoAzure
from src.plot import PDFWriter
from os.path import join, isdir
from functools import reduce

from src.config import (
    SubjectDataIterator,
    StereoAzureSubjectLoader,
    RPESubjectLoader,
    DataFrameLoader,
)

from src.processing import (
    segment_1d_joint_on_example,
    compute_mean_and_std_of_joint_for_subjects,
    align_skeleton_parallel_to_x_axis,
)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import argparse
import os

# matplotlib.use("TkAgg")

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


def fuse_both_kinect_cameras(pdf_file: str):
    pdf_writer = PDFWriter(pdf_file)
    file_iterator = SubjectDataIterator(
        base_path=args.src_path,
        log_path=args.dst_path,
        loaders=[StereoAzureSubjectLoader, RPESubjectLoader],
    )

    sum_repetitions = 0

    for set_data in file_iterator.iterate_over_all_subjects():
        dst_path = join(args.dst_path, set_data['subject_name'])
        log_path = join(args.log_path, set_data['subject_name'])
        for cur_path in [dst_path, log_path]:
            if not os.path.exists(cur_path):
                os.makedirs(cur_path)

        azure = StereoAzure(
            master_path=set_data['azure']['master'],
            sub_path=set_data['azure']['sub'],
        )

        repetitions = segment_1d_joint_on_example(
            joint_data=azure.sub_position['PELVIS (y)'],
            exemplar=example,
            std_dev_p=0.5,
            show=False,
        )

        sum_repetitions += len(repetitions)

        # Cut Kinect data before first and right after last repetition
        azure.cut_skeleton_data(repetitions[0][0], repetitions[-1][1])
        avg_df = azure.fuse_cameras(show=True, pp=pdf_writer)
        avg_df = align_skeleton_parallel_to_x_axis(avg_df)
        avg_df.to_csv(f"{os.path.join(dst_path, str(set_data['nr_set']))}_azure.csv", sep=';', index=True)

        # Save individual repetitions
        for count, (r1, r2) in enumerate(repetitions):
            csv_dir = join(log_path, "csv")
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
            avg_df.loc[r1:r2].to_csv(join(csv_dir, f"{set_data['nr_set']}_{count}_{set_data['rpe']}.csv"),
                                     sep=';',
                                     index=False)

        dst_path = join(args.dst_path, set_data['subject_name'])
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        pdf_writer.add_booklet(set_data['subject_name'], set_data['nr_set'], avg_df.columns)

    pdf_writer.close_and_save_file()
    print(f'Found in total {sum_repetitions} repetitions.')
    conf = AzureKinect.conf_values
    subs = reduce(lambda df1, df2: df1 + df2, ([value for key, value in conf.items() if "sub" in key]))
    master = reduce(lambda df1, df2: df1 + df2, ([value for key, value in conf.items() if "master" in key]))
    subs.to_csv(join("results", "conf_sub.csv"), sep=';', index=False)
    master.to_csv(join("results", "conf_master.csv"), sep=';', index=False)

    os.system('cd data/raw;find . -type f -name "*.json" -exec install -v {} ../processed/{} \\;')


def plot_repetition_data(pdf_file: str):
    pdf_render = PDFWriter(pdf_file)
    subjects = list(filter(lambda x: isdir(x), map(lambda x: join(args.log_path, x), os.listdir(args.log_path))))

    cmap = plt.cm.get_cmap("jet")
    norm = matplotlib.colors.Normalize(vmin=10, vmax=20)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    def parse_rpe_file(segment_file: str):
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
                plt.plot(df[col], color=cmap(norm(rpe)))

            plt.xlabel("Frames [1/30s]")
            plt.ylabel("Distance [mm]")
            plt.title(f"{subject_name} {col}")
            plt.colorbar(sm)
            plt.tight_layout()
            pdf_render.save_figure()
            plt.clf()

        pdf_render.add_booklet(subject_name, 0, cols)

    pdf_render.close_and_save_file()


def normalize_data_plot(pdf_file: str):
    file_iterator = SubjectDataIterator(
        base_path=args.dst_path,
        loaders=[DataFrameLoader]
    )
    means, std = compute_mean_and_std_of_joint_for_subjects(file_iterator.iterate_over_all_subjects())
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

    pdf_writer.close_and_save_file()


if __name__ == '__main__':
    fuse_both_kinect_cameras(pdf_file='results/raw_fusion.pdf')
    # plot_repetition_data(pdf_file='results/fusion_segmented.pdf')
