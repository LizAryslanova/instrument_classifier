import os




source = '/Users/cookie/dev/instrumant_classifier/audio_files/nsynth/audio/'
destination = '/Users/cookie/dev/instrumant_classifier/audio_files/nsynth/Training_audios/'

allfiles = os.listdir(source)

# iterate on all files to move them to destination folder
i = 31
print('Processing: ', i)
counter = 0
for f in allfiles:

    counter += 1


    if counter < 10000:
        src_path = os.path.join(source, f)
        dst_path = os.path.join(destination  + str(i) + '/', f)
        os.rename(src_path, dst_path)

