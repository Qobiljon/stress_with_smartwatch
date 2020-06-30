import os

mob_dir = 'C:/Users/Kevin/Desktop/data-processing-v2/9. mobility-filter-v2'
ppg_dir = 'C:/Users/Kevin/Desktop/data-processing-v2/10. ppg-filter-v2'

for file in [file for file in os.listdir(mob_dir) if file.endswith('.csv')]:
	print(file[:-4])
	with open('{0}/{1}'.format(mob_dir, file), 'r') as mob, open('{0}/{1}'.format(ppg_dir, file), 'r') as ppg:
		ppgs = [tuple(line[:-1].split(',')) for line in ppg.readlines()[1:]]
		mobs = [tuple(line[:-1].split(',')) for line in mob.readlines()[1:]]
		ovr_amount = max(len(ppgs), len(mobs))
		
		ppgs = [int(ts) for ts,ar in ppgs if ar.lower() == 'true']
		mobs = [int(ts) for ts,ar in mobs if ar.lower() == 'true']
		
		ppg_artifacts_count = len(ppgs)
		mob_artifacts_count = len(mobs)
		combined_artifacts_count = len(set(ppgs) & set(mobs))
		
		print('\tPPG artifacts\t', ppg_artifacts_count - combined_artifacts_count, '\t-%.1f%%' % (100 * ppg_artifacts_count / ovr_amount))
		print('\tBTH artifacts\t', combined_artifacts_count, '\t-%.1f%%' % (100 * combined_artifacts_count / ovr_amount))
		print('\tMOB artifacts\t', mob_artifacts_count - combined_artifacts_count, '\t-%.1f%%' % (100 * mob_artifacts_count / ovr_amount))
		print('\tCLN data\t', ovr_amount - ppg_artifacts_count - mob_artifacts_count + combined_artifacts_count)
