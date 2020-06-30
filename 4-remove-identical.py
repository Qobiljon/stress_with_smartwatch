# !pip install hrv-analysis
import os, csv, datetime, math
import numpy as np
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from hrvanalysis import get_time_domain_features, get_frequency_domain_features, get_csi_cvi_features, get_geometrical_features, get_poincare_plot_features, get_sampen
print('-- libraries imported --')




# paths
root_dir = os.getcwd()
raw_dir = '{0}/{1}'.format(root_dir, '0. combined-dataset')
sep_dir = '{0}/{1}'.format(root_dir, '1. separated-dataset')
no_filter_dir = '{0}/{1}'.format(root_dir, '2. identical-removed')
# id->value maps
user_id_email_map = {'19': 'salman@nsl.inha.ac.kr', '20': 'jumabek4044@gmail.com', '18': 'jskim@nsl.inha.ac.kr', '11': 'aliceblackwood123@gmail.com', '10': 'laurentkalpers3@gmail.com', '7': 'nazarov7mu@gmail.com', '1': 'nslabinha@gmail.com', '8': 'azizsambo58@gmail.com', '3': 'nnarziev@gmail.com', '6': 'mr.khikmatillo@gmail.com'}
data_source_id_name_map = {'34': 'HR', '37': 'ACCELEROMETER', '35': 'RR_INTERVAL', '33': 'HAD_ACTIVITY', '38': 'AMBIENT_LIGHT', '31': 'LOCATION', '36': 'HAD_SLEEP_PATTERN', '32': 'LIGHT_INTENSITY'}
# participants
emails = [user_id_email_map[id] for id in user_id_email_map]
# features
feature_names = [
  'mean_nni', 'sdnn', 'sdsd', 'rmssd', 'median_nni', 'nni_50', 'pnni_50', 'nni_20', 'pnni_20', 'range_nni', 'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr', 'std_hr',  # time domain features
  'total_power', 'vlf', 'lf', 'hf', 'lf_hf_ratio', 'lfnu', 'hfnu',  # frequency domain features
  'csi', 'cvi', 'Modified_csi',  # csi / cvi features
  'triangular_index', 'tinn',  # geometrical features
  'sd1', 'sd2', 'ratio_sd2_sd1',  # pointcare plot features
  'sampen'  # sample entropy as a feature
]
sensing_window_size = 1800000 # 30 mins
feature_window_size = 60000 # 1 min
acc_window_size = 10000 # 10 sec
activity_threshold = 3
print('-- global variables initialized --')




# converts string time into timestamp (i.e., 2020-04-23T11:00+0900 --> 1587607200000)
def str2ts(_str):
    if _str == '':
        return None
    elif _str[-3] == ':':
        _str = _str[:-3] + _str[-2:]
    return int(datetime.datetime.strptime(_str, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp()) * 1000


# converts a numeric string into a number
def str2num(_str):
    if _str == '':
        return None
    else:
        return int(_str)


# loads ESM responses, calculates scores, and adds a label (i.e., "stressed" / "not-stressed")
def load_ground_truths(esm_file, participant_email):
    res = []
    with open(esm_file, 'r') as r:
        csv_reader = csv.reader(r, delimiter=',', quotechar='"')
        for csv_row in csv_reader:
            if csv_row[0] != participant_email:
                continue
            rt = str2ts(csv_row[11])
            st = str2ts(csv_row[12])
            control = str2num(csv_row[16])
            difficulty = str2num(csv_row[17])
            confident = str2num(csv_row[18])
            yourway = str2num(csv_row[19])
            row = (
                st is None,  # is self report
                rt if st is None else st,  # timestamp
                control,  # (-)PSS:Control
                difficulty,  # (-)PSS:Difficult
                confident,  # (+)PSS:Confident
                yourway,  # (+)PSS:YourWay
                str2num(csv_row[20]),  # LikeRt:StressLevel,
            )
            if None in row:
                continue
            score = (control + difficulty + (6 - confident) + (6 - yourway)) / 4
            res += [(row) + (score,)]
        res.sort(key=lambda e: e[1])
        mean = sum(row[7] for row in res) / len(res)
        for i in range(len(res)):
            res[i] += (res[i][7] > mean,)
    return res


# loads participant's IBI readings
def load_rr_data(parent_dir, participant_email):
    res = []
    file_path = os.path.join(parent_dir, participant_email, 'RR_INTERVAL.csv')
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as r:
        for line in r:
            ts, rr = line[:-1].split(',')
            try:
                ts, rr = int(ts), int(rr)
            except ValueError:
                continue
            res += [(ts, rr)]
        res.sort(key=lambda e: e[0])
    return res


# loads participant's ACC readings
def load_acc_data(parent_dir, participant_email):
    res = []
    file_path = os.path.join(parent_dir, participant_email, 'ACCELEROMETER.csv')
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as r:
        for line in r:
            cells = line[:-1].split(',')
            if len(cells) == 2:
                continue
            try:
                ts, x, y, z = int(cells[0]), float(cells[1]), float(cells[2]), float(cells[3])
            except ValueError:
                continue
            res += [(ts, x, y, z)]
        res.sort(key=lambda e: e[0])
    return res


# selects data within the specified timespan
def select_data(dataset, from_ts, till_ts):
    res = []
    read = {}
    for row in dataset:
        if row not in read:
            if from_ts <= row[0] < till_ts:
                res += [row[1]]
                read[row] = True
        elif row[0] >= till_ts:
            break
    return res if len(res) >= 60 else None


# selects the closest data point (w/ timestamp)
def find_closest(start_ts, ts, dataset):
	while ts not in dataset and ts != start_ts:
		ts -= 1
	if ts == start_ts:
		# print('error occurred, reached the start_ts!')
		# exit(1)
		return None
	else:
		return dataset[ts]


# downsamples data
def select_downsample_data(dataset, from_ts, till_ts):
  selected_data = {}
  for row in dataset:
    if from_ts <= row[0] < till_ts:
      selected_data[row[0]] = row[1]
    elif row[0] >= till_ts:
      break
  timestamps = [ts for ts in range(from_ts, till_ts, 1000)]
  res = []
  for ts in timestamps:
    closest = find_closest(start_ts=from_ts, ts=ts, dataset=selected_data)
    if closest is not None:
    	res += [closest]
  return res if len(res) >= 20 else None

def is_acc_window_active(acc_values):  
    activeness_scores = []
    magnitudes = []
    end = acc_values[0][0] + acc_window_size
    last_ts = acc_values[-1][0]
    for ts, x, y, z in acc_values:
        if ts >= end or ts == last_ts:
            activeness_scores += [np.mean(magnitudes)]
            end += acc_window_size
            magnitudes = []
        magnitudes += [math.sqrt(x**2 + y**2 + z**2)]
    # low_limit = np.percentile(stdevs, 1)
    # high_limit = np.percentile(stdevs, 99)
    # activeness_range = high_limit - low_limit
    active_count = 0
    for activeness_score in activeness_scores:
        #if activeness_score > (low_limit + activity_threshold * activeness_range):
        if activeness_score > activity_threshold:
            active_count += 1
    if active_count > len(activeness_scores) / 2:
        return True
    else:
        return False


# calculates stress features from provided IBI readings
def calculate_features(rr_intervals):
    # process the RR-intervals
    rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_intervals, low_rri=300, high_rri=2000)
    interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers, interpolation_method='linear')
    nn_intervals = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method='malik')
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals)

    # extract the features
    time_domain_features = get_time_domain_features(nn_intervals=interpolated_nn_intervals)
    frequency_domain_features = get_frequency_domain_features(nn_intervals=interpolated_nn_intervals)
    csi_cvi_features = get_csi_cvi_features(nn_intervals=interpolated_nn_intervals)
    geometrical_features = get_geometrical_features(nn_intervals=interpolated_nn_intervals)
    poincare_plot_features = get_poincare_plot_features(nn_intervals=interpolated_nn_intervals)
    sample_entropy = get_sampen(nn_intervals=interpolated_nn_intervals)

    return [
        time_domain_features['mean_nni'],  # The mean of RR-intervals
        time_domain_features['sdnn'],  # The standard deviation of the time interval between successive normal heart beats (i.e. the RR-intervals)
        time_domain_features['sdsd'],  # The standard deviation of differences between adjacent RR-intervals
        time_domain_features['rmssd'],  # The square root of the mean of the sum of the squares of differences between adjacent NN-intervals. Reflects high frequency (fast or parasympathetic) influences on hrV (i.e., those influencing larger changes from one beat to the next)
        time_domain_features['median_nni'],  # Median Absolute values of the successive differences between the RR-intervals
        time_domain_features['nni_50'],  # Number of interval differences of successive RR-intervals greater than 50 ms
        time_domain_features['pnni_50'],  # The proportion derived by dividing nni_50 (The number of interval differences of successive RR-intervals greater than 50 ms) by the total number of RR-intervals
        time_domain_features['nni_20'],  # Number of interval differences of successive RR-intervals greater than 20 ms
        time_domain_features['pnni_20'],  # The proportion derived by dividing nni_20 (The number of interval differences of successive RR-intervals greater than 20 ms) by the total number of RR-intervals
        time_domain_features['range_nni'],  # Difference between the maximum and minimum nn_interval
        time_domain_features['cvsd'],  # Coefficient of variation of successive differences equal to the rmssd divided by mean_nni
        time_domain_features['cvnni'],  # Coefficient of variation equal to the ratio of sdnn divided by mean_nni
        time_domain_features['mean_hr'],  # Mean heart rate value
        time_domain_features['max_hr'],  # Maximum heart rate value
        time_domain_features['min_hr'],  # Minimum heart rate value
        time_domain_features['std_hr'],  # Standard deviation of heart rate values

        frequency_domain_features['total_power'],  # Total power density spectral
        frequency_domain_features['vlf'],  # variance (=power) in HRV in the Very low Frequency (.003 to .04 Hz by default). Reflect an intrinsic rhythm produced by the heart which is modulated primarily by sympathetic activity
        frequency_domain_features['lf'],  # variance (=power) in HRV in the low Frequency (.04 to .15 Hz). Reflects a mixture of sympathetic and parasympathetic activity, but in long-term recordings, it reflects sympathetic activity and can be reduced by the beta-adrenergic antagonist propanolol
        frequency_domain_features['hf'],  # variance (=power) in HRV in the High Frequency (.15 to .40 Hz by default). Reflects fast changes in beat-to-beat variability due to parasympathetic (vagal) activity. Sometimes called the respiratory band because it corresponds to HRV changes related to the respiratory cycle and can be increased by slow, deep breathing (about 6 or 7 breaths per minute) and decreased by anticholinergic drugs or vagal blockade
        frequency_domain_features['lf_hf_ratio'],  # lf/hf ratio is sometimes used by some investigators as a quantitative mirror of the sympatho/vagal balance
        frequency_domain_features['lfnu'],  # normalized lf power
        frequency_domain_features['hfnu'],  # normalized hf power

        csi_cvi_features['csi'],  # Cardiac Sympathetic Index
        csi_cvi_features['cvi'],  # Cardiac Vagal Index
        csi_cvi_features['Modified_csi'],  # Modified CSI is an alternative measure in research of seizure detection

        geometrical_features['triangular_index'],  # The HRV triangular index measurement is the integral of the density distribution (= the number of all NN-intervals) divided by the maximum of the density distribution
        geometrical_features['tinn'],  # The triangular interpolation of NN-interval histogram (TINN) is the baseline width of the distribution measured as a base of a triangle, approximating the NN-interval distribution

        poincare_plot_features['sd1'],  # The standard deviation of projection of the Poincaré plot on the line perpendicular to the line of identity
        poincare_plot_features['sd2'],  # SD2 is defined as the standard deviation of the projection of the Poincaré plot on the line of identity (y=x)
        poincare_plot_features['ratio_sd2_sd1'],  # Ratio between SD2 and SD1

        sample_entropy['sampen'],  # The sample entropy of the Normal to Normal Intervals
    ]
print('-- utility functions defined --')




for email in emails:
    print('calculating stress-features of "', email, "'s IBI readings")
    with open('{0}/{1}.csv'.format(no_filter_dir, email), 'w+') as w:
        w.write('timestamp,{0},gt_self_report,gt_timestamp,gt_pss_control,gt_pss_difficult,gt_pss_confident,gt_pss_yourway,gt_likert_stresslevel,gt_score,gt_label\n'.format(','.join(feature_names)))
        participants_rr_dataset = load_rr_data(parent_dir=sep_dir, participant_email=email)
        ground_truths = load_ground_truths(esm_file='{0}/all_ESMs.csv'.format(raw_dir), participant_email=email)
        for ground_truth in ground_truths:
            is_self_report = ground_truth[0]
            timestamp = ground_truth[1]
            till_timestamp = timestamp - 1800000 if is_self_report else timestamp - 3600000
            while timestamp > till_timestamp:
                selected_rr_intervals = select_data(
                    dataset=participants_rr_dataset,
                    from_ts=timestamp-feature_window_size,
                    till_ts=timestamp
                )
                timestamp -= feature_window_size
                if selected_rr_intervals is not None:
                    try:
                        print('processing :', timestamp, email)
                        features = calculate_features(selected_rr_intervals)
                        w.write('{0},{1},{2}\n'.format(
                            timestamp,
                            ','.join([str(value) for value in features]),
                            ','.join([str(value) for value in ground_truth])
                        ))
                    except ValueError:
                        print('erroneous case met :', email)
                        pass

# print stats for features - no filter
print('participant-email\t\tsamples\t\tstressed\tnot-stressed')
for email in emails:
    sti = len(feature_names)
    with open('{0}/{1}.csv'.format(no_filter_dir, email), 'r') as r:
        lines = r.readlines()[1:]
        print('{0}\t\t{1}\t\t{2}\t\t{3}'.format(
            email[:20],
            len(lines),
            len([1 for line in lines if line[:-1].split(',')[sti+9].lower() == 'true']),
            len([1 for line in lines if not line[:-1].split(',')[sti+9].lower() == 'true'])
        ))