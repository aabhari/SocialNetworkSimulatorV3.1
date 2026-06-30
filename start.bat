@echo off
setlocal

if not exist classes (
    mkdir classes
)
if not exist classes (
    echo Could not create the classes directory.
    pause
    exit /b 1
)

echo Compiling Java sources...
javac -nowarn -cp "lib/*" -d classes TwitterGatherDataFollowers\userRyersonU\*.java
if errorlevel 1 (
    echo Java compilation failed. ControllerAgent.class was not created.
    pause
    exit /b 1
)

if not exist classes\TwitterGatherDataFollowers\userRyersonU\ControllerAgent.class (
    echo ControllerAgent.class was not created under classes\TwitterGatherDataFollowers\userRyersonU.
    pause
    exit /b 1
)

java -Xms256m -Xmx14240m -XX:-UseGCOverheadLimit -cp "lib/*;classes" jade.Boot -jade_domain_df_maxresult 1500 -jade_core_messaging_MessageManager_poolsize 10 -jade_core_messaging_MessageManager_maxqueuesize 2000000000 -jade_core_messaging_MessageManager_deliverytimethreshold 10000 -jade_domain_df_autocleanup true -local-port 35240 controller:TwitterGatherDataFollowers.userRyersonU.ControllerAgent
