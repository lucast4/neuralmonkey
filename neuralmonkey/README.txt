# Steps, if installing in new machine.

Make directories in neuralmonkey:
/logs
/logs_checks


Paths - set symbolic links, to paths in SN.Paths:
1. Path to where neural_preprocess (e.g., if it is in /lemur2/lucas/neural_preprocess):
sudo ln -s /lemur2/lucas /gorilla1

2. Path to server should be /mnt/Freiwald (both of these are hard coded in self.Paths).
sudo ln -s /home/lucas/mnt/Freiwald /mnt/Freiwald
sudo ln -s /home/lucas/mnt/Freiwald /mnt/hopfield_data01
