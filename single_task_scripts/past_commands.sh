./render_single_directory.sh ../../BootstrappedOutputs_v4/faustarap/hessian/faust10_hi/samples/ 512 /home/samk/glass_project/matcaps/glass.png 1>../render_logs/samples.out 2>../render_logs/samples.err &
./render_single_directory.sh ../../BootstrappedOutputs_v4/faustarap/hessian/faust10_hi/recon/ 512 /home/samk/glass_project/matcaps/reconstruction.png 1>../render_logs/recon.out 2>../render_logs/recon.err &
./render_single_directory.sh ../../BootstrappedOutputs_v4/faustarap/hessian/faust10_hi/lse/ 512 /home/samk/glass_project/matcaps/lse.png 1>../render_logs/lse.out 2>../render_logs/lse.err &
./render_single_directory.sh ../../Bootstrapped/Data/faust10_hi/ 512 /home/samk/glass_project/matcaps/dataset.png 1>../render_logs/data.out 2>../render_logs/data.err &
