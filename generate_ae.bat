@echo OFF
@REM How to run a Python script in a given conda environment from a batch file.

@REM It doesn't require:
@REM - conda to be in the PATH
@REM - cmd.exe to be initialized with conda init

@REM Define here the path to your conda installation
set CONDAPATH=C:\Users\Miniconda3
@REM Define here the name of the environment
set ENVNAME=xai

@REM The following command activates the base environment.
if %ENVNAME%==base (set ENVPATH=%CONDAPATH%) else (set ENVPATH=%CONDAPATH%\envs\%ENVNAME%)

@REM Activate the conda environment
@REM Using call is required here, see: https://stackoverflow.com/questions/24678144/conda-environments-and-bat-files
call %CONDAPATH%\Scripts\activate.bat %ENVPATH%

@REM Run a python script in that environment

@REM RUN DeepFool Attack for 10 times

@REM python 2_generate_ae.py --deepfool True --times 0
@REM python 2_generate_ae.py --deepfool True --times 1
@REM python 2_generate_ae.py --deepfool True --times 2
@REM python 2_generate_ae.py --deepfool True --times 3
@REM python 2_generate_ae.py --deepfool True --times 4
@REM python 2_generate_ae.py --deepfool True --times 5
@REM python 2_generate_ae.py --deepfool True --times 6
@REM python 2_generate_ae.py --deepfool True --times 7
@REM python 2_generate_ae.py --deepfool True --times 8
@REM python 2_generate_ae.py --deepfool True --times 9

@REM RUN Carlini Attack for 10 times

@REM python 2_generate_ae.py --carlini True --times 0
@REM python 2_generate_ae.py --carlini True --times 1
@REM python 2_generate_ae.py --carlini True --times 2
@REM python 2_generate_ae.py --carlini True --times 3
@REM python 2_generate_ae.py --carlini True --times 4
@REM python 2_generate_ae.py --carlini True --times 5
@REM python 2_generate_ae.py --carlini True --times 6
@REM python 2_generate_ae.py --carlini True --times 7
@REM python 2_generate_ae.py --carlini True --times 8
@REM python 2_generate_ae.py --carlini True --times 9

@REM RUN LowProFool Attack for 10 times

@REM python 2_generate_ae.py --lowprofool True --times 0
@REM python 2_generate_ae.py --lowprofool True --times 1
@REM python 2_generate_ae.py --lowprofool True --times 2
@REM python 2_generate_ae.py --lowprofool True --times 3
@REM python 2_generate_ae.py --lowprofool True --times 4
@REM python 2_generate_ae.py --lowprofool True --times 5
@REM python 2_generate_ae.py --lowprofool True --times 6
@REM python 2_generate_ae.py --lowprofool True --times 7
@REM python 2_generate_ae.py --lowprofool True --times 8
@REM python 2_generate_ae.py --lowprofool True --times 9

@REM RUN FGSM Attack for 10 times

@REM python 2_generate_ae.py --fgsm True --times 0
@REM python 2_generate_ae.py --fgsm True --times 1
@REM python 2_generate_ae.py --fgsm True --times 2
@REM python 2_generate_ae.py --fgsm True --times 3
@REM python 2_generate_ae.py --fgsm True --times 4
@REM python 2_generate_ae.py --fgsm True --times 5
@REM python 2_generate_ae.py --fgsm True --times 6
@REM python 2_generate_ae.py --fgsm True --times 7
@REM python 2_generate_ae.py --fgsm True --times 8
@REM python 2_generate_ae.py --fgsm True --times 9

@REM RUN BIM Attack for 10 times

@REM python 2_generate_ae.py --bim True --times 0
@REM python 2_generate_ae.py --bim True --times 1
@REM python 2_generate_ae.py --bim True --times 2
@REM python 2_generate_ae.py --bim True --times 3
@REM python 2_generate_ae.py --bim True --times 4
@REM python 2_generate_ae.py --bim True --times 5
@REM python 2_generate_ae.py --bim True --times 6
@REM python 2_generate_ae.py --bim True --times 7
@REM python 2_generate_ae.py --bim True --times 8
@REM python 2_generate_ae.py --bim True --times 9

@REM RUN MIM Attack for 10 times

@REM python 2_generate_ae.py --mim True --times 0
@REM python 2_generate_ae.py --mim True --times 1
@REM python 2_generate_ae.py --mim True --times 2
@REM python 2_generate_ae.py --mim True --times 3
@REM python 2_generate_ae.py --mim True --times 4
@REM python 2_generate_ae.py --mim True --times 5
@REM python 2_generate_ae.py --mim True --times 6
@REM python 2_generate_ae.py --mim True --times 7
@REM python 2_generate_ae.py --mim True --times 8
@REM python 2_generate_ae.py --mim True --times 9

@REM RUN PGD Attack for 10 times

@REM python 2_generate_ae.py --pgd True --times 0
@REM python 2_generate_ae.py --pgd True --times 1
@REM python 2_generate_ae.py --pgd True --times 2
@REM python 2_generate_ae.py --pgd True --times 3
@REM python 2_generate_ae.py --pgd True --times 4
@REM python 2_generate_ae.py --pgd True --times 5
@REM python 2_generate_ae.py --pgd True --times 6
@REM python 2_generate_ae.py --pgd True --times 7
@REM python 2_generate_ae.py --pgd True --times 8
@REM python 2_generate_ae.py --pgd True --times 9


@REM RUN Boundary Attack for 10 times

@REM python 2_generate_ae.py --boundary True --times 0
@REM python 2_generate_ae.py --boundary True --times 1
@REM python 2_generate_ae.py --boundary True --times 2
@REM python 2_generate_ae.py --boundary True --times 3
@REM python 2_generate_ae.py --boundary True --times 4
@REM python 2_generate_ae.py --boundary True --times 5
@REM python 2_generate_ae.py --boundary True --times 6
@REM python 2_generate_ae.py --boundary True --times 7
@REM python 2_generate_ae.py --boundary True --times 8
@REM python 2_generate_ae.py --boundary True --times 9

@REM RUN Boundary Attack for 10 times

@REM python 2_generate_ae.py --hopskipjump True --times 0
@REM python 2_generate_ae.py --hopskipjump True --times 1
@REM python 2_generate_ae.py --hopskipjump True --times 2
@REM python 2_generate_ae.py --hopskipjump True --times 3
@REM python 2_generate_ae.py --hopskipjump True --times 4
@REM python 2_generate_ae.py --hopskipjump True --times 5
@REM python 2_generate_ae.py --hopskipjump True --times 6
@REM python 2_generate_ae.py --hopskipjump True --times 7
@REM python 2_generate_ae.py --hopskipjump True --times 8
@REM python 2_generate_ae.py --hopskipjump True --times 9

@REM Deactivate the environment
call conda deactivate

@REM If conda is directly available from the command line then the following code works.
@REM call activate xai
@REM python script.py
@REM conda deactivate

@REM One could also use the conda run command
@REM conda run -n someenv python script.py