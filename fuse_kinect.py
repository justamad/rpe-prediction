from rpe_prediction.config import SubjectDataIterator, StereoAzureLoader, RPELoader
from rpe_prediction.processing import segment_1d_joint_on_example, get_hsv_color_interpolation
from rpe_prediction.stereo_cam import StereoAzure
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfFileWriter, PdfFileReader
from os.path import join, isdir

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
import io

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


def plot_repetition_data(output_path, file_name):
    """
    Plot the resulting joints into a single
    :param output_path:
    :param file_name:
    :return:
    """
    subjects = list(filter(lambda x: isdir(x), map(lambda x: join(args.log_path, x), os.listdir(args.log_path))))
    figures = []

    for subject in subjects:
        sub_path = join(subject, "csv")
        files = list(filter(lambda f: f.endswith('.csv'), os.listdir(sub_path)))
        files = list(map(lambda x: join(sub_path, x), files))
        files = list(map(lambda f: (pd.read_csv(f, sep=';', index_col=False), int(f.split('_')[2][:2])), files))

        fig = plt.figure()
        for df, rpe in files:
            plt.plot(-df['pelvis (y)'], color=get_hsv_color_interpolation(rpe, 20))

        plt.title(subject)
        plt.xlabel("Frames [1/30s]")
        plt.ylabel("Distance [mm]")
        figures.append(fig)

    pp = PdfPages(file_name)
    for fig in figures:
        pp.savefig(fig)
    pp.close()


def fuse_kinect_data(iterator, pdf_file, show=False):
    """
    Fuse Kinect data from sub and master camera and calculate features
    @param iterator: iterator that crawls through raw data
    @param pdf_file: output name of current pdf file
    @param show: A flag whether to show the plots or not
    """
    pdf_buffer = io.BytesIO()
    pp = PdfPages(pdf_buffer)

    bookmarks = []

    for set_data in iterator:
        # Create output folder to save averaged skeleton
        dst_path = join(args.dst_path, set_data['subject_name'])
        log_path = join(args.log_path, set_data['subject_name'])
        for cur_path in [dst_path, log_path]:
            if not os.path.exists(cur_path):
                os.makedirs(cur_path)

        sub_path, master_path = set_data['azure']
        azure = StereoAzure(master_path=master_path, sub_path=sub_path)
        azure.reduce_skeleton_joints()

        # Segment data based on pelvis joint
        repetitions = segment_1d_joint_on_example(joint_data=azure.sub_position['pelvis (y)'],
                                                  exemplar=example, std_dev_percentage=0.5,
                                                  show=False)  # , path=join(log_path, f"{set_data['nr_set']}_segment.png"))

        # Cut Kinect data
        azure.cut_skeleton_data(repetitions[0][0], repetitions[-1][1])
        azure.calculate_affine_transform_based_on_data(show=show)
        print(f"Agreement internal: {azure.check_agreement_of_both_cameras(show=show)}")
        avg_df = azure.fuse_cameras(show=True, pp=pp)
        avg_df.to_csv(f"{os.path.join(dst_path, str(set_data['nr_set']))}_azure.csv", sep=';', index=True)

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

        # Add data for bookmarks
        for joint in avg_df.columns:
            bookmarks.append((set_data['subject_name'], set_data['nr_set'], joint))

    # Write the final PDF file
    pp.close()
    output_file = PdfFileWriter()
    input_file = PdfFileReader(pdf_buffer)

    bookmark_cache = {}
    for nr_page in range(input_file.getNumPages()):
        output_file.addPage(input_file.getPage(nr_page))

        # Determine the parent bookmark
        subject_name, nr_set, joint = bookmarks[nr_page]
        if subject_name not in bookmark_cache:
            bookmark_cache[subject_name] = output_file.addBookmark(subject_name, nr_page, parent=None)
        subject_bookmark = bookmark_cache[subject_name]

        # Determine set bookmark
        key = subject_name + str(nr_set)
        if key not in bookmark_cache:
            bookmark_cache[key] = output_file.addBookmark(f"Set {nr_set}", nr_page, parent=subject_bookmark)
        set_bookmark = bookmark_cache[key]

        output_file.addBookmark(joint, nr_page, parent=set_bookmark)

    outputStream = open(pdf_file, 'wb')
    output_file.write(outputStream)
    outputStream.close()


if __name__ == '__main__':
    # if args.show_plots:  # Quick fix for running script on server
    import matplotlib
    matplotlib.use("TkAgg")

    file_iterator = SubjectDataIterator(args.src_path).add_loader(StereoAzureLoader).add_loader(RPELoader)
    iterator = file_iterator.iterate_over_specific_subjects("C47EFC")
    fuse_kinect_data(iterator, pdf_file='raw_output.pdf', show=False)
    # plot_repetition_data(args.log_path, 'report.pdf')
