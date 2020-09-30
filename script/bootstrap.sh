
#!/bin/sh

# script/bootstrap: 
# Resolve all dependencies that the application requires to run.
# This can mean packages, software language versions, Git submodules, etc.
# The goal is to make sure all required dependencies are installed.

# Install based on OS
# https://stackoverflow.com/a/3466183/363829
case "$(uname -s)" in
   Darwin)
      # Do something under Mac OS X platform
    echo "Kernal = Mac OS X, using brew install wget"
    brew install wget
     ;;

   Linux)
       # Do something under GNU/Linux platform
    echo "Kernal = Linux, using apt install unzip"
    apt install unzip
     ;;
esac

## Check for $FLOOD_ANALYSIS_CORE
# https://stackoverflow.com/q/3601515
if [ -z "${FLOOD_ANALYSIS_CORE:-}" ]; then echo "FLOOD_ANALYSIS_CORE is unset"; else echo "FLOOD_ANALYSIS_CORE is set to '$FLOOD_ANALYSIS_CORE'"; fi
