
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

# %%

# input data
data_file = 'analysis/data/output_2020-04-07_09h04m54s.csv'

# show plots
display_plots = True

# output plot size
plot_size = {
    'width': 7.5,
    'height': 5,
    'dpi': 200
}

# table output size
table_size = {
    'width': 5,
    'height': 9,
    'dpi': 200
}

# %%
# Load data
data_type = {'names': ('case', 'image',
                       'detector', 'descriptor',
                       'matcher', 'selector',
                       'detector_duration', 'descriptor_duration', 'matching_duration',
                       'keypoints', 'keypoint_size_mean', 'keypoint_size_stddev',
                       'matches'),
             'formats': ('i', 'i',
                         'U10', 'U10',
                         'U10', 'U10',
                         'd', 'd', 'd',
                         'd', 'd', 'd',
                         'i')}
raw_data = np.loadtxt(data_file, skiprows=1, delimiter=',', dtype=data_type)


# %%
# Get average keypoint size for each detector
keypoint_stddev = {}
keypoint_mean_size = {}
keypoints = {}
keypoints_images = {}
num_images = np.max(raw_data['image'])

# number of keypoints for each detector
for row in raw_data:
    detector = row['detector']
    if detector not in keypoints:
        keypoints[detector] = 0
        keypoints_images[detector] = np.zeros((num_images,))
    keypoints[detector] += row['keypoints']
    keypoints_images[detector][row['image'] - 1] = row['keypoints']

# get mean size
for row in raw_data:
    detector = row['detector']
    if detector not in keypoint_mean_size:
        keypoint_mean_size[detector] = 0.0
    keypoint_mean_size[detector] += row['keypoint_size_mean'] * \
        row['keypoints'] / keypoints[detector]

# get size stddev
for row in raw_data:
    detector = row['detector']
    if detector not in keypoint_stddev:
        keypoint_stddev[detector] = 0.0
    keypoint_stddev[detector] += (np.power(row['keypoint_size_stddev'], 2) + np.power(
        row['keypoint_size_mean'] - keypoint_mean_size[detector], 2)) * row['keypoints'] / keypoints[detector]

# standard deviation
for detector in keypoint_stddev:
    keypoint_stddev[detector] = np.sqrt(keypoint_stddev[detector])


# %%
# keypoint count plot
# plot labels
plot_title = "Detector vs. Keypoint Count"
plot_x_label = "Detector"
plot_y_label = "Number of Keypoints"

# plot data
plot_keypoint_count_mean = []
plot_keypoint_count_errmin = []
plot_keypoint_count_errmax = []
plot_keypoint_labels = []

for detector in keypoints_images:
    plot_keypoint_labels.append(detector)
    count_mean = np.mean(keypoints_images[detector])
    count_min = np.min(keypoints_images[detector])
    count_max = np.max(keypoints_images[detector])
    plot_keypoint_count_mean.append(count_mean)
    plot_keypoint_count_errmin.append(count_mean - count_min)
    plot_keypoint_count_errmax.append(count_max - count_mean)

fig_keypoint_count, ax = plt.subplots()

x = np.arange(len(plot_keypoint_count_mean))

ax.bar(x, plot_keypoint_count_mean, yerr=(
    plot_keypoint_count_errmin, plot_keypoint_count_errmax), capsize=3)
ax.set_xticks(x)
ax.set_xticklabels(plot_keypoint_labels)

ax.set_ylabel(plot_y_label)
ax.set_xlabel(plot_x_label)
ax.set_title(plot_title)

# %%
# keypoint size plot
# plot labels
plot_title = "Detector vs. Mean Keypoint Size"
plot_x_label = "Detector"
plot_y_label = "Keypoint Size (pixels)"

# plot data
plot_keypoint_size = []
plot_keypoint_stddev = []
plot_keypoint_labels = []
for detector in keypoint_mean_size:
    plot_keypoint_labels.append(detector)
    plot_keypoint_size.append(keypoint_mean_size[detector])
    plot_keypoint_stddev.append(keypoint_stddev[detector])

# bar plot
x = np.arange(len(plot_keypoint_labels))
fig_keypoint_size, ax = plt.subplots()

ax.bar(x, plot_keypoint_size, yerr=plot_keypoint_stddev, capsize=3)
ax.set_xticks(x)
ax.set_xticklabels(plot_keypoint_labels)

ax.set_ylabel(plot_y_label)
ax.set_xlabel(plot_x_label)
ax.set_title(plot_title)

# %%
# Get timing info for each case
detector_duration = {}
descriptor_duration = {}
case_labels = {}
case_images = {}
case_durations = {}
case_matches = {}
case_matches_min = {}
case_matches_max = {}

# number of data points for each case
for row in raw_data:
    case = row['case']
    if case not in case_images:
        case_images[case] = 0
        case_labels[case] = (row['detector'], row['descriptor'])
    case_images[case] += 1

# get mean timing and matches
for row in raw_data:
    case = row['case']
    # initialize
    if case not in detector_duration:
        detector_duration[case] = 0.0
    if case not in descriptor_duration:
        descriptor_duration[case] = 0.0
    if case not in case_matches:
        case_matches[case] = 0
        case_matches_min[case] = row['matches']
        case_matches_max[case] = row['matches']
    # mean time
    detector_duration[case] += row['detector_duration'] / case_images[case]
    descriptor_duration[case] += row['descriptor_duration'] / case_images[case]
    # mean matches
    case_matches[case] += row['matches'] / case_images[case]
    # min/max matches
    case_matches_min[case] = np.min((case_matches_min[case], row['matches']))
    case_matches_max[case] = np.max((case_matches_max[case], row['matches']))

# get total mean durations and matches
for case in case_labels:
    detector, descriptor = case_labels[case]
    case_durations[case] = detector_duration[case] + descriptor_duration[case]

# %%
# matches plot
# plot labels
plot_title = "Detector vs. Matches Count"
plot_x_label = "Case Number (refer to table)"
plot_y_label = "Number of Matches"

# plot data
plot_matches_count = []
plot_case_labels = []
plot_yerr_min = []
plot_yerr_max = []
for case in case_matches:
    count_mean = case_matches[case]
    count_min = case_matches_min[case]
    count_max = case_matches_max[case]

    plot_case_labels.append(case)
    plot_matches_count.append(count_mean)
    plot_yerr_min.append(count_mean - count_min)
    plot_yerr_max.append(count_max - count_mean)

# bar plot
x = np.arange(len(plot_case_labels))
fig_matches_count, ax = plt.subplots()

ax.bar(x, plot_matches_count, yerr=(plot_yerr_min, plot_yerr_max), capsize=3)
ax.set_xticks(x)
ax.set_xticklabels(plot_case_labels)

ax.set_ylabel(plot_y_label)
ax.set_xlabel(plot_x_label)
ax.set_title(plot_title)

# %%
# timing plot
# plot labels
plot_title = "Detector vs. Descriptor Durations"
plot_x_label = "Detector Duration (ms)"
plot_y_label = "Descriptor Duration (ms)"

# plot data
plot_duration_detector = []
plot_duration_descriptor = []
plot_duration_annotations = []
for case in detector_duration:
    plot_duration_annotations.append(case)
    plot_duration_detector.append(detector_duration[case])
    plot_duration_descriptor.append(descriptor_duration[case])

# scatter plot
fig_duration_scatter, ax = plt.subplots()

pline = ax.plot(plot_duration_detector, plot_duration_descriptor)[0]
pline.set_linestyle('none')
pline.set_marker('o')

for k, annotation in enumerate(plot_duration_annotations):
    ax.annotate(annotation,
                (plot_duration_detector[k], plot_duration_descriptor[k]))

ax.set_ylabel(plot_y_label)
ax.set_xlabel(plot_x_label)
ax.set_title(plot_title)

# %%
# timing plot - descriptor vs detector histogram
# plot labels
plot_title = "Detector vs. Descriptor Durations"
plot_x_label = "Detector"
plot_y_label = "Descriptor"

# detector and descriptor labels
plot_detector_labels = []
plot_descriptor_labels = []
plot_detector_index = {}
plot_descriptor_index = {}
for case in case_labels:
    detector, descriptor = case_labels[case]
    if detector not in plot_detector_index:
        plot_detector_index[detector] = len(plot_detector_labels)
        plot_detector_labels.append(detector)
    if descriptor not in plot_descriptor_index:
        plot_descriptor_index[descriptor] = len(plot_descriptor_labels)
        plot_descriptor_labels.append(descriptor)
    case_durations[case] = detector_duration[case] + descriptor_duration[case]

# 2d mesh plot data
x = np.arange(-0.5, len(plot_detector_labels) + 0.5, 1.0)
y = np.arange(-0.5, len(plot_descriptor_labels) + 0.5, 1.0)

xx, yy = np.meshgrid(x, y)
zz = np.zeros(xx.shape)
zz = zz[:-1, :-1]

minmax = None
for case in case_labels:
    detector, descriptor = case_labels[case]
    xind = plot_detector_index[detector]
    yind = plot_descriptor_index[descriptor]
    zz[yind, xind] = case_durations[case]
    if minmax is None:
        minmax = [case_durations[case], case_durations[case]]
    else:
        minmax = [np.fmin(minmax[0], case_durations[case]),
                  np.fmax(minmax[1], case_durations[case])]
zz[zz < minmax[0]] = minmax[1] + 0.5

# plot
fig_duration_mesh, ax = plt.subplots()
im = ax.pcolor(xx, yy, zz, vmin=minmax[0], vmax=minmax[1])
fig_duration_mesh.colorbar(im, ax=ax)

ax.set_xticks(x[:-1] + 0.5)
ax.set_xticklabels(plot_detector_labels)

ax.set_yticks(y[:-1] + 0.5)
ax.set_yticklabels(plot_descriptor_labels)

for case in case_labels:
    detector, descriptor = case_labels[case]
    xind = plot_detector_index[detector]
    yind = plot_descriptor_index[descriptor]
    ax.annotate("{:.0f}".format(case_durations[case]), (xind - 0.2, yind))

ax.set_ylabel(plot_y_label)
ax.set_xlabel(plot_x_label)
ax.set_title(plot_title)
# %%
# Duration table

table_data = []
rows = []
columns = ('Detector', 'Descriptor', 'Duration (ms)', 'Matches')

durations = []
for case in case_labels:
    detector, descriptor = case_labels[case]

    durations.append(case_durations[case])
    duration_string = "{:.1f}".format(case_durations[case])
    matches_string = "{:.0f}".format(case_matches[case])

    table_data.append([detector, descriptor, duration_string, matches_string])
    rows.append("Case {}".format(case))

columns = np.array(columns)
rows = np.array(rows)
table_data = np.array(table_data)

fig_table, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
ax.table(cellText=table_data,
                     rowLabels=rows,
                     colLabels=columns, loc='center')

sort_indices = np.argsort(durations)
rows = rows[sort_indices]
table_data = table_data[sort_indices, :]

fig_table_sorted, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
ax.table(cellText=table_data,
                     rowLabels=rows,
                     colLabels=columns, loc='center')


# %%
# save plots

fprefix, _ = os.path.splitext(data_file)

fig_keypoint_count.set_size_inches(plot_size['width'], plot_size['height'])
fig_keypoint_size.set_size_inches(plot_size['width'], plot_size['height'])
fig_matches_count.set_size_inches(plot_size['width'] + 1, plot_size['height'])
fig_duration_scatter.set_size_inches(plot_size['width'], plot_size['height'])
fig_duration_mesh.set_size_inches(plot_size['width'], plot_size['height'])
fig_table.set_size_inches(table_size['width'], table_size['height'])
fig_table_sorted.set_size_inches(table_size['width'], table_size['height'])

fig_keypoint_count.tight_layout()
fig_keypoint_size.tight_layout()
fig_matches_count.tight_layout()
fig_duration_scatter.tight_layout()
fig_duration_mesh.tight_layout()
fig_table.tight_layout()
fig_table_sorted.tight_layout()

print("Saving plots...")

fig_keypoint_count.savefig(fprefix + "_kpnt_count.png",
                           bbox_inches='tight', dpi=plot_size['dpi'])
fig_keypoint_size.savefig(fprefix + "_kpnt_size.png",
                          bbox_inches='tight', dpi=plot_size['dpi'])
fig_matches_count.savefig(fprefix + "_matches.png",
                          bbox_inches='tight', dpi=plot_size['dpi'])
fig_duration_scatter.savefig(
    fprefix + "_duration_scatter.png", bbox_inches='tight', dpi=plot_size['dpi'])
fig_duration_mesh.savefig(fprefix + "_duration_mesh.png",
                          bbox_inches='tight', dpi=plot_size['dpi'])
fig_table.savefig(fprefix + "_table.png",
                  bbox_inches='tight', dpi=table_size['dpi'])
fig_table_sorted.savefig(fprefix + "_table_sorted.png",
                  bbox_inches='tight', dpi=table_size['dpi'])

print("Done")

# show plots
if display_plots:
    plt.show()
