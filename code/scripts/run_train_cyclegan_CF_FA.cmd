@set "dataDir=F:\Nova\RetinaImageSynthesisProject\Code\pytorch-CycleGAN-and-pix2pix"

python F:\Nova\RetinaImageSynthesisProject\Code\pytorch-CycleGAN-and-pix2pix\train.py ^
	--name CycleGAN_CF2FAinv_2021_12_14 ^
	--dataroot F:\Nova\RetinaImageSynthesisProject\Data ^
	--checkpoints_dir %dataDir%\Checkpoints\ ^
	--nameA VIS ^
	--nameB FA ^
	--model cycle_gan ^
	--norm instance ^
	--pool_size 50^
	--no_dropout ^
	--loadSize 720 ^
	--fineSize 448 ^
	--batch_size 1 ^
	--input_nc 3 ^
	--output_nc 3 ^
	--invertB ^
	--dataset_mode unaligned ^
	--netG resnet_6blocks ^
	--netD global_np ^
	--niter 800 ^
	--niter_decay 400 ^
	--display_freq 100 ^
	--display_id 0 ^
		