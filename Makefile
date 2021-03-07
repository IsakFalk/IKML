pull_air_quality_data:
	curl -o data/exp_raw/beijing_air_quality/data.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip
	unzip data/exp_raw/beijing_air_quality/data.zip -d data/exp_raw/beijing_air_quality

pull_gas_sensor_data:
	curl -o data/exp_raw/gas_sensor/data.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00487/gas-sensor-array-temperature-modulation.zip
	unzip data/exp_raw/gas_sensor/data.zip -d data/exp_raw/gas_sensor

create_air_quality_dataset: pull_air_quality_data
	python3 scripts/data/beijing_air_quality/create_dataset.py

create_gas_sensor_dataset: pull_gas_sensor_data
	python3 scripts/data/gas_sensor/create_dataset.py

create_datasets: create_air_quality_dataset create_gas_sensor_dataset

# Format all files using autoflake / isort / black
remove_unused_imports:
	autoflake implicit_kernel_meta_learning --recursive --in-place --remove-unused-variables --remove-all-unused-imports --ignore-init-module-imports
	autoflake scripts --recursive --in-place --remove-unused-variables --remove-all-unused-imports --ignore-init-module-imports

sort_imports:
	isort implicit_kernel_meta_learning --atomic
	isort scripts --atomic

format_pyfiles:
	black implicit_kernel_meta_learning
	black scripts

format_package: remove_unused_imports sort_imports format_pyfiles

# Output conda spec file
freeze_conda_environment:
	conda list --explicit > spec-file.txt

# Process project for update
update_project: format_package freeze_conda_environment 
