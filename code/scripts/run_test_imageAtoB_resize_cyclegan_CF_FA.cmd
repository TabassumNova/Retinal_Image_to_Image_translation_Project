@set "model=CycleGAN_CF2FAinv_2021_12_14"
@set "phase=train"
@set "size=720"

python test_load_size.py ^
	--name %model% ^
	--dataroot C:\Users\Sindel\Project\Data\datasets\Multi-Modal_Retina_Fundus_Color_Fluorescein_Angiogram\Images ^
	--imglist_path C:\Users\Sindel\Project\Data\datasets\Multi-Modal_Retina_Fundus_Color_Fluorescein_Angiogram\Images ^
	--checkpoints_dir E:\CycleGAN\Checkpoints\ ^
	--model test_AtoB ^
	--phase %phase% ^
	--direction AtoB ^
	--no_dropout ^
	--fineSize %size% ^
	--loadSize %size% ^
	--input_nc 3 ^
	--output_nc 3 ^
	--dataset_mode test_dir_image ^
	--netG resnet_9blocks ^
	--netD global_np ^
	--nameA VIS ^
	--nameB FA ^
	--invertB ^
	--results_dir E:\CycleGAN\Results\%model%\%phase%\fakeB ^
	