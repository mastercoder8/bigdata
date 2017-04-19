#unzipper
import os
import zipfile
import tarfile
def checkpaths(paths):
    for path in paths:
        if not os.path.exists(path):
            print(path+" doesn't exist")
            return False
    return True
video_source_path = '../../../../EagleEye/Videos/'
align_source_path = '../../../../EagleEye/Align/'

video_dest_path = './resources/Videos/'#'./resources/Videos/'
align_dest_path = './resources/Align/'

def main():
    if not checkpaths([video_source_path,align_dest_path,video_dest_path,align_dest_path]): return
    speaker_list = next(os.walk(video_source_path))[2]
    align_list = next(os.walk(align_source_path))[2]
    print("Total Speakers:",len(speaker_list))
    if not len(align_list)==len(speaker_list):
        print("Speaker count & align count doesnt match",len(align_list),len(speaker_list))
    for i in range(1,len(speaker_list)+1):
        if os.path.exists(video_source_path+'s'+str(i)+'.mpg_vcd.zip'):
            print("Unzip:"+video_source_path+'s'+str(i)+'.mpg_vcd.zip'+" -> "+video_dest_path,end='')
            # zip_ref = zipfile.ZipFile(video_source_path+'s'+str(i)+'.mpg_vcd.zip', 'r')
            # zip_ref.extractall(video_dest_path)
            print("Renaming:"+video_dest_path+'s'+str(i)+" -> "+video_dest_path+str(i))
            # os.rename(video_dest_path+'s'+str(i),video_dest_path+str(i))
            # zip_ref.close()
        if os.path.exists(align_source_path+'s'+str(i)+'.tar'):
            print("Untar:", align_source_path+'s'+str(i)+'.tar'+ " -> "+ align_dest_path,end='\t')
            # tar_ref = tarfile.TarFile(align_source_path+'s'+str(i)+'.tar','r')
            # tar_ref.extractall(align_dest_path)
            print("Renaming:",align_dest_path+'align'," -> ",align_dest_path+str(i))
            # os.rename(align_dest_path+'align',align_dest_path+str(i))
            # tar_ref.close()


if __name__ == '__main__':
    main()