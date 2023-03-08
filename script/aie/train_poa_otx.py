import os
import subprocess

if __name__ == "__main__":
    # command_build = (
    #     "otx build Custom_Image_Classification_EfficinetNet-B0 "
    #     "--backbone mmcls.VisionTransformer "
    #     "--work-dir otx-workspace-AIE/POA_attribution")

    # subprocess.run(command_build, shell=True)

    GPU = 0
    os.makedirs("otx-workspace-AIE/POA_attribution/otx", exist_ok=True)
    command_train = (
        f"CUDA_VISIBLE_DEVICES={GPU} otx train "
        f"--save-model-to /local/sungchul/logs/aie/poa/otx "
        f"--work-dir otx-workspace-AIE/POA_attribution/otx "
        f"2>&1 | tee otx-workspace-AIE/POA_attribution/otx/output.log")

    subprocess.run(command_train, shell=True)
