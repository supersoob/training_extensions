import os
import subprocess

if __name__ == "__main__":
    # command_build = (
    #     "otx build Custom_Image_Classification_EfficinetNet-B0 "
    #     "--backbone mmcls.VisionTransformer "
    #     "--work-dir otx-workspace-AIE/POA_attribution")

    # subprocess.run(command_build, shell=True)

    GPU = 2
    # os.makedirs("otx-workspace-AIE/POA_attribution/clip_vit_freeze", exist_ok=True)
    # command_train = (
    #     f"CUDA_VISIBLE_DEVICES={GPU} otx train "
    #     f"--save-model-to /local/sungchul/logs/aie/poa/clip_vit_freeze "
    #     f"--work-dir otx-workspace-AIE/POA_attribution/clip_vit_freeze "
    #     f"2>&1 | tee otx-workspace-AIE/POA_attribution/clip_vit_freeze/output.log")

    # subprocess.run(command_train, shell=True)

    os.makedirs("otx-workspace-AIE/POA_attribution/otx_clip_pretrained_freeze", exist_ok=True)
    command_train = (
        f"CUDA_VISIBLE_DEVICES={GPU} otx train "
        f"--save-model-to /local/sungchul/logs/aie/poa/otx_clip_pretrained_freeze "
        f"--work-dir otx-workspace-AIE/POA_attribution/otx_clip_pretrained_freeze "
        f"2>&1 | tee otx-workspace-AIE/POA_attribution/otx_clip_pretrained_freeze/output.log")

    subprocess.run(command_train, shell=True)

    # os.makedirs("otx-workspace-AIE/POA_attribution/otx_pretrained_freeze", exist_ok=True)
    # command_train = (
    #     f"CUDA_VISIBLE_DEVICES={GPU} otx train "
    #     f"--save-model-to /local/sungchul/logs/aie/poa/otx_pretrained_freeze "
    #     f"--work-dir otx-workspace-AIE/POA_attribution/otx_pretrained_freeze "
    #     f"2>&1 | tee otx-workspace-AIE/POA_attribution/otx_pretrained_freeze/output.log")

    # subprocess.run(command_train, shell=True)
