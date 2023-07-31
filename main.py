import matplotlib.pyplot as plt
import numpy as np


subjects_hash = {}  # global variable


def read_from_file(path):
    data = []
    f = open(path, 'r')
    headers = f.readline()
    headers = headers.strip('\n').split(',')
    # remove \n from string and split the header to a list by ',' as a separator

    while True:
        line = (f.readline())
        if not line:
            break
        else:
            data.append(line[:-1].split(','))  # remove the last element in line: '\n'
    return headers, np.array(data, dtype='str')


def shallow_analysis(arr):
    path = "./shallow analysis output.txt"
    f = open(path, 'w')
    devices = len(arr)
    print("factory test data:\n", arr, '\n')
    f.write("factory test data:\n" + str(arr) + '\n')
    print("\nNumber of devices: " + str(devices) + '\n')
    f.write("\nNumber of devices: " + str(devices) + '\n')
    subjects = arr[:, 0]
    data = arr[:, range(2, 9, 2)]
    unique_values = np.unique(subjects)
    bad_samples = data == '[]'
    f.write("\nThe subjects: \n" + str(unique_values) + '\n')
    # print("Data only: (Y axis) \n", data, '\n')
    f.write("\nNumber of malfunctioning devices: " + str(int(bad_samples.sum() / 4)) + '\n')  # divide by 4 due to 4 samples
    f.write("\nNumber of bad samples: " + str(bad_samples.sum()) + '\n')
    print("The subjects: \n", unique_values, '\n')
    # print("Data only: (Y axis) \n", data, '\n')
    print("Number of malfunctioning devices: ", int(bad_samples.sum() / 4), '\n')  # divide by 4 due to 4 samples
    print("Number of bad samples: ", bad_samples.sum(), '\n')


def Data_loading_and_Explain():
    path = "./factory_test_data.csv"
    headers_load, np_data_load = read_from_file(path)
    shallow_analysis(np_data_load)
    forming_hash(np_data_load)
    return headers_load, np_data_load


def plot_XY(arr):
    """ Plot all subjects. Each subject's graph will contain all of its devices samples """
    x_axis = [int(np.unique(arr[:, 1])[0]), int(np.unique(arr[:, 3])[0]), int(np.unique(arr[:, 5])[0]), int(np.unique(arr[:, 7])[0])]
    for subject in subjects_hash:
        subject_samples = subjects_hash[subject]
        fig, ax = plt.subplots()
        for devices in subject_samples:
            if (devices[1] == '[]').any():
                ax.plot(x_axis, np.zeros(4), label="device " + str(devices[0]))
                continue
            ax.plot(x_axis, devices[1].astype(float), label="device " + str(devices[0]), marker='o', markersize=8)
            ax.legend(loc='upper right')
        ax.set_title("subject " + str(subject) + " X-Y Graph")
        ax.grid(True)
        plt.show()


def plot_correation_distribution(sub_data, correl_arr):
    pass


def Data_Visualization(arr):
    plot_XY(arr)


def find_correlation(subject_arr):
    """ Will find the cross-correlation between different devices on the same subject

                                            -- NOTICE --
        Bad samples recognized by '[]'. every vector with '[]' will have an array of 0's correlation!


        :param subject_arr: list of subjects devices and their tests
        :return The cross-correlation array, contains tuples of which devices where crossed and their result """

    cross_correlation = []
    for deviceX_idx in range(0, (len(subject_arr))):
        deviceX = subject_arr[deviceX_idx][0]
        deviceX_data = subject_arr[deviceX_idx][1]
        bad_scan_x = False
        if (deviceX_data == '[]').any():
            bad_scan_x = True
        for deviceY_idx in range(deviceX_idx + 1, len(subject_arr)):
            deviceY = subject_arr[deviceY_idx][0]
            deviceY_data = subject_arr[deviceY_idx][1]
            if bad_scan_x:
                # print('BAD SCAN - deviceX\n')
                cross_correlation.append([(deviceX, deviceY), np.zeros(4)])
                continue
            if (deviceY_data == '[]').any():
                # print('BAD SCAN - deviceY\n')
                cross_correlation.append([(deviceX, deviceY), np.zeros(4)])
                continue
            correlated = np.correlate(deviceX_data.astype(float), deviceY_data.astype(float), mode='same')
            normalized_correlation = correlated / np.max(correlated)
            cross_correlation.append([(deviceX, deviceY), normalized_correlation.astype(float)])
    return cross_correlation



# --------------------------- DATA STRUCTURE ------------------------------

def forming_hash(arr):
    """ keys will be the subject number (as a string). every key will have a list containing lists.
        every inner list will have the device number in index 0 and it's tests results """
    unique_values = np.unique(arr[:, 0])
    for value in unique_values:
        subjects_hash[value] = []
        find_subject_tests(value, arr)


def find_subject_tests(sub, arr):
    """ Find for a specific subject all of it tests by the devices
        and map them to the subject's hash table

        :param      sub: Subject number (as a string)
        :param      arr: NumPy array contain all of test """

    global subjects_hash
    for device, tests in enumerate(arr):
        if tests[0] != sub:
            continue
        else:
            data = tests[2::2]
            subjects_hash[sub].extend([[device + 1, data]])

# --------------------------- END OF DATA STRUCTURE ------------------------------


headers, np_data = Data_loading_and_Explain()
print(subjects_hash, '\n')
plot_XY(np_data)


