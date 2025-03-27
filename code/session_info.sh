#!/bin/bash

# Script to display information about the used environment and hardware.

# Display NVIDIA System Management Interface and store it in the output log file.
echo '========================================================================'
echo 'Output of: nvidia-smi'
echo '========================================================================'
nvidia-smi
printf '\n'


echo '========================================================================'
echo 'Output of: cat /etc/os-release'
echo '========================================================================'
cat /etc/os-release
printf '\n'

echo '========================================================================'
echo 'Linux Standard Base: lsb_release -a'
echo '========================================================================'
lsb_release -a
printf '\n'

echo '========================================================================'
echo 'Linux host: hostnamectl'
echo '========================================================================'
hostnamectl
printf '\n'

echo '========================================================================'
echo 'Information about the current locale'
echo '========================================================================'
locale
printf '\n'

echo '========================================================================'
echo 'List of Linux modules currently loaded (in alphabetical order)'
echo '========================================================================'
lsmod | sort
printf '\n'

# Print loaded Python version
echo '========================================================================'
echo 'Python version'
echo '========================================================================'
python3 -c 'import sys; print(sys.version)'
printf '\n'

echo '========================================================================'
echo 'Paths used by the Python interpreter'
echo '========================================================================'
python3 -c "import sys; print('\n'.join(sys.path))"
printf '\n'

echo '========================================================================'
echo 'List of Python packages currently installed in the environment (in alphabetical order)'
echo '========================================================================'
pip list # pip displays the packages and their version in alphabetical order by default.
printf '\n'
