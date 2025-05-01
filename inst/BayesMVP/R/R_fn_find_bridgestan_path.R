




#' bridgestan_path
#' @export
bridgestan_path <- function() {
  
  suppressMessages({
    # Check if the BRIDGESTAN environment variable is already set
    if (!(Sys.getenv("BRIDGESTAN") %in% c("", " ", "  "))) {
      message(paste(("Bridgestan path found at:"), Sys.getenv("BRIDGESTAN")))
      return(Sys.getenv("BRIDGESTAN")) # Use the value from the environment variable
    }
    
    # Get the user's home directory
    home_dir <- Sys.getenv(if (.Platform$OS.type == "windows") "USERPROFILE" else "HOME")
    
    # Define the default paths for BridgeStan v2.5.0
    default_path <- file.path(home_dir, ".bridgestan", "bridgestan-2.5.0")
    
    # Check if the default path exists
    if (dir.exists(default_path)) {
      Sys.setenv(BRIDGESTAN=default_path)
      message(paste(("Bridgestan path found at:"), default_path))
      return(default_path)
    }
    
    # If v2.5.0 not available, then fallback to finding the most recent bridgestan directory
    search_pattern <- file.path(home_dir, ".bridgestan", "bridgestan-*")
    available_dirs <- Sys.glob(search_pattern)
    
    # Filter for valid version directories and sort by version
    if (length(available_dirs) > 0) {
      recent_dir <- available_dirs[order(available_dirs, decreasing = TRUE)][1]
      Sys.setenv(BRIDGESTAN=recent_dir)
      message(paste(("Bridgestan path found at:"), recent_dir))
      return(recent_dir)
    }
    
    ## Check for plain "bridgestan" dir (i.e. w/o a ".")
    bridgestan_dir <- file.path(home_dir, "bridgestan")
    if (dir.exists(bridgestan_dir)) {
      Sys.setenv(BRIDGESTAN=bridgestan_dir)
      message(paste(("Bridgestan path found at:"), bridgestan_dir))
      return(bridgestan_dir)
    }
    
    # If no directory found, just return the BRIDGESTAN environment variable anyway
    # This will allow the Makevars file to show the correct error
    bridgestan_env <- Sys.getenv("BRIDGESTAN", "")
    message("BridgeStan directory not found. Using BRIDGESTAN environment variable: ", bridgestan_env)
    return(bridgestan_env)
  })
  
}
