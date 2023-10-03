import nrrd
import random
import SimpleITK as sitk
from lama import common
import os
import pandas as pd
from pathlib import Path


def randomise_file_list(_dir):

    o_dir = Path(_dir.parent / "output")
    os.mkdir(o_dir)

    file_list = [file_name for file_name in common.get_file_paths(_dir, )]
    print(file_list)
    # randomise list
    random.shuffle(file_list)

    file_df = pd.DataFrame({"name": file_list, "num": range(len(file_list))})

    print(file_df)

    file_df.to_csv(o_dir/"results.csv")
    for i, file_name in enumerate(file_list):
        i_name = o_dir / (str(i) + ".nrrd")
        os.rename(file_name,i_name)

def randomise_drishti_file_list(_dir):

    o_dir = Path(_dir/ "output")
    os.mkdir(o_dir)

    file_list = [file_name for file_name in common.get_file_paths(_dir, extension_tuple='001')]
    print(file_list)
    # randomise list
    random.shuffle(file_list)

    file_df = pd.DataFrame({"name": file_list, "num": range(100, 100+len(file_list))})

    print(file_df)



    file_df.to_csv(o_dir/"results.csv")
    for i, file_name in enumerate(file_list):
        i_name = o_dir / (str(i) + ".pvl.nc.001")
        os.rename(file_name,i_name)


        i_name_2 =  o_dir / (str(i) + ".pvl.nc")


        file_name_2 = o_dir.parent / os.path.basename(Path(file_name).stem)
        print(file_name, file_name_2)
        print(i_name, i_name_2)

        os.rename(file_name_2, i_name_2)






def main():
    _dir = Path("E:/230815_all_emb_renders/renders_grp_3/")
    randomise_drishti_file_list(_dir)

if __name__ == '__main__':
    main()