



#' convert_Stan_data_list_to_JSON
#' @keywords internal
#' @export
convert_Stan_data_list_to_JSON <- function(Stan_data_list, 
                                           data_dir = NULL) {
      
      if (is.null(data_dir)) { 
        # Get package directory paths
        pkg_dir <- system.file(package = "BayesMVP")
        data_dir <- file.path(pkg_dir, "stan_data")  # directory to store data inc. JSON data files
      }
  
      # Create data directory if it doesn't exist
      if (!dir.exists(data_dir)) {
        dir.create(data_dir, recursive = TRUE)
      }
      
      ## make persistent (non-temp) JSON data file path with unique identifier:
      data_hash <- digest::digest(Stan_data_list)  # Hash the data to create unique identifier
      json_filename <- paste0("data_", data_hash, ".json")
      json_file_path <- file.path(data_dir, json_filename)
      ##
      ## write JSON data using cmdstanr:
      cmdstanr::write_stan_json(data = Stan_data_list, file = json_file_path)
      ##
      ## json_file_path <- normalizePath(json_file_path)
      
      # ## Clean the JSON
      # ## json_file_path <- "/home/enzo/R/x86_64-pc-linux-gnu-library/4.4/BayesMVP/stan_data/data_19e7904abe32c558dffad0f43d88da59.json"
      # ##
      # json_str <- base::readLines(json_file_path)
      # json_clean <- gsub('^\\["', '', json_str)
      # json_clean <- gsub('"\\]$', '', json_clean)
      # json_clean <- gsub('\\"', '"', json_clean)
      # ##
      # ## Write back to the file:
      # base::writeLines(json_clean, json_file_path)
      # ##
      # ## Read the file
      # json_str <- base::readLines(json_file_path)
      # ##
      # ## Remove escaped quotes 
      # json_clean <- gsub('\\"', '"', json_str)
      # ##
      # ## Write it back
      # base::writeLines(json_clean, json_file_path)
      # ##
      # ## Read the file
      # json_str <- base::readLines(json_file_path)
      # ##
      # ## Remove escaped quotes and any extra whitespace
      # json_clean <- json_str
      # json_clean <- gsub('\\"', '"', json_clean)  # Replace escaped quotes
      # json_clean <- gsub('\\\\', '', json_clean)  # Remove any backslashes
      # ## json_clean <- gsub('^\s+', '', json_clean)  # Remove leading whitespace
      # ## json_clean <- gsub('\s+$', '', json_clean)  # Remove trailing whitespace
      # 
      # # # Read the JSON string
      # # json_str <- base::readLines(json_file_path)
      # 
      # ##
      # json_data <- jsonlite::fromJSON(json_clean)
      # ##
      # jsonlite::write_json(json_data, json_file_path, auto_unbox = TRUE, pretty = TRUE)
      
      return(json_file_path)
  
}
  





#' convert_JSON_string_to_R_vector
#' @keywords internal
#' @export
convert_JSON_string_to_R_vector <- function(json_string) {
  
  require(jsonlite)
  
  # Helper function to flatten a matrix/array row by row
  flatten_matrix <- function(m) {
    if(is.null(dim(m))) {
      return(as.numeric(m))
    }
    return(as.numeric(t(matrix(unlist(m), ncol=ncol(m), byrow=TRUE))))
  }
  
  # Helper function to recursively process nested structures
  process_element <- function(x) {
    if(is.numeric(x) && length(x) == 1) {
      return(x)
    }
    if(is.list(x) || is.matrix(x) || is.array(x)) {
      if(is.matrix(x) || (is.list(x) && all(sapply(x, length) == length(x[[1]])))) {
        return(flatten_matrix(x))
      }
      # Recursively process nested elements
      return(unlist(lapply(x, process_element)))
    }
    return(as.numeric(x))
  }
  
  # Parse JSON
  data <- jsonlite::fromJSON(json_string)
  
  # Process each top-level element in order of appearance
  result <- c()
  for(name in names(data)) {
    result <- c(result, process_element(data[[name]]))
  }
  
  return(result)
  
}








