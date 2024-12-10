import os


# -s data/horse_blender --eval -m output/horse_blender -w --brdf_dim 1 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512
# datasets = ["angel_blender","bell_blender","cat_blender","luyu_blender","potion_blender","tbell_blender","teapot_blender"]


datasets =os.listdir("datasets")
# "horse_blender",
common_args = " --eval -w --brdf_dim 1 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512"
for dataset in datasets:
    os.system("python super_train.py -s " + "datasets/"+dataset +" -m " + "output/"+dataset +  common_args)