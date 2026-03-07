// Mehul: Created this file for Socket Interface
#include "socket_interface.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <cmath>
#include <iomanip>
#include <git_info.h>

namespace dftfe
{

  SocketDriver::SocketDriver(const std::string &host, int port, MPI_Comm comm)
    : host(host)
    , port(port)
    , comm(comm)
    , sockfd(-1)
  {
    MPI_Comm_rank(comm, &rank);
  }

  SocketDriver::~SocketDriver()
  {
    close_socket();
  }

  void
  SocketDriver::connect_socket()
  {
    if (rank == 0)
      {
        struct sockaddr_in serv_addr;
        struct hostent    *server;

        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0)
          {
            std::cerr << "ERROR opening socket" << std::endl;
            exit(1);
          }

        server = gethostbyname(host.c_str());
        if (server == NULL)
          {
            std::cerr << "ERROR, no such host" << std::endl;
            exit(1);
          }

        bzero((char *)&serv_addr, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;
        bcopy((char *)server->h_addr,
              (char *)&serv_addr.sin_addr.s_addr,
              server->h_length);
        serv_addr.sin_port = htons(port);

        // Retry connection loop
        int retries = 0;
        while (
          connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
          {
            if (retries > 10)
              {
                std::cerr << "ERROR connecting" << std::endl;
                exit(1);
              }
            sleep(1);
            retries++;
          }

        // Send handshake
        std::string handshake = "READY\n";
        write(sockfd, handshake.c_str(), handshake.length());
      }
  }

  void
  SocketDriver::close_socket()
  {
    if (rank == 0 && sockfd >= 0)
      {
        close(sockfd);
        sockfd = -1;
      }
  }

  void
  SocketDriver::send_data(const std::string &data)
  {
    if (rank == 0)
      {
        std::string msg = data + "\n";
        int         n   = write(sockfd, msg.c_str(), msg.length());
        if (n < 0)
          std::cerr << "ERROR writing to socket" << std::endl;
      }
  }

  std::string
  SocketDriver::receive_data()
  {
    std::string data;
    if (rank == 0)
      {
        char buffer[4096];
        bool done = false;
        while (!done)
          {
            bzero(buffer, 4096);
            int n = read(sockfd, buffer, 4095);
            if (n <= 0)
              {
                // Socket closed or error
                data = "EXIT";
                done = true;
              }
            else
              {
                data += buffer;
                if (data.find('\n') != std::string::npos)
                  {
                    data = data.substr(0, data.find('\n')); // Take first line
                    done = true;
                  }
              }
          }
      }

    // Broadcast data to all ranks
    int len = data.length();
    MPI_Bcast(&len, 1, MPI_INT, 0, comm);
    if (rank != 0)
      data.resize(len);
    MPI_Bcast((void *)data.data(), len, MPI_CHAR, 0, comm);

    return data;
  }

  // Helper to parse [[x,y,z], ...]
  std::vector<std::vector<double>>
  parse_matrix(const std::string &json, const std::string &key)
  {
    std::vector<std::vector<double>> mat;
    size_t                           pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos)
      return mat;

    pos = json.find("[", pos); // Start of outer array
    if (pos == std::string::npos)
      return mat;

    size_t end = json.find("]]", pos);
    if (end == std::string::npos)
      return mat;

    std::string content =
      json.substr(pos + 1, end - pos); // Inside outer brackets

    std::stringstream ss(content);
    char              c;
    while (ss >> c)
      {
        if (c == '[')
          {
            std::vector<double> row;
            double              val;
            while (ss >> val)
              {
                row.push_back(val);
                ss >> c; // comma or ]
                if (c == ']')
                  break;
              }
            mat.push_back(row);
          }
      }
    return mat;
  }

  // Helper to parse [1, 2, 3]
  std::vector<dftfe::uInt>
  parse_int_array(const std::string &json, const std::string &key)
  {
    std::vector<dftfe::uInt> vec;
    size_t                   pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos)
      return vec;

    pos = json.find("[", pos);
    if (pos == std::string::npos)
      return vec;

    size_t end = json.find("]", pos);
    if (end == std::string::npos)
      return vec;

    std::string       content = json.substr(pos + 1, end - pos - 1);
    std::stringstream ss(content);
    int               val;
    char              c;
    while (ss >> val)
      {
        vec.push_back((dftfe::uInt)val);
        ss >> c; // comma
      }
    return vec;
  }

  // Helper to parse [true, false]
  std::vector<bool>
  parse_bool_array(const std::string &json, const std::string &key)
  {
    std::vector<bool> vec;
    size_t            pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos)
      return vec;

    pos = json.find("[", pos);
    if (pos == std::string::npos)
      return vec;

    size_t end = json.find("]", pos);
    if (end == std::string::npos)
      return vec;

    std::string       content = json.substr(pos + 1, end - pos - 1);
    std::stringstream ss(content);
    std::string       val;
    while (std::getline(ss, val, ','))
      {
        // Trim whitespace
        val.erase(0, val.find_first_not_of(" \t\n\r"));
        val.erase(val.find_last_not_of(" \t\n\r") + 1);
        if (val == "true")
          vec.push_back(true);
        else
          vec.push_back(false);
      }
    return vec;
  }

  // Helper to parse scalar values
  template <typename T>
  T
  parse_scalar(const std::string &json, const std::string &key, T default_val)
  {
    size_t pos = json.find("\"" + key + "\":");
    if (pos == std::string::npos)
      return default_val;

    pos        = json.find(":", pos) + 1;
    size_t end = json.find_first_of(",}", pos);
    if (end == std::string::npos)
      return default_val;

    std::string val_str = json.substr(pos, end - pos);
    // Trim
    val_str.erase(0, val_str.find_first_not_of(" \t\n\r"));
    val_str.erase(val_str.find_last_not_of(" \t\n\r") + 1);

    std::stringstream ss(val_str);
    T                 val;
    if (val_str == "true")
      return (T) true;
    if (val_str == "false")
      return (T) false;
    ss >> val;
    return val;
    return val;
  }

  // Helper to parse string values
  std::string
  parse_string(const std::string &json,
               const std::string &key,
               std::string        default_val = "")
  {
    size_t pos = json.find("\"" + key + "\":");
    if (pos == std::string::npos)
      return default_val;

    pos                = json.find(":", pos) + 1;
    size_t start_quote = json.find("\"", pos);
    if (start_quote == std::string::npos)
      return default_val;

    size_t end_quote = json.find("\"", start_quote + 1);
    if (end_quote == std::string::npos)
      return default_val;

    return json.substr(start_quote + 1, end_quote - start_quote - 1);
  }

  void
  SocketDriver::parse_request(const std::string                &json,
                              std::vector<std::vector<double>> &coords,
                              std::vector<std::vector<double>> &cell,
                              std::vector<dftfe::uInt>         &numbers,
                              std::vector<bool>                &pbc,
                              std::vector<dftfe::uInt>         &mp_grid,
                              std::vector<dftfe::uInt>         &mp_grid_shift,
                              dftfe::Int                       &spin_polarized,
                              double      &start_magnetization,
                              double      &fermi_temp,
                              dftfe::uInt &npkpt,
                              double      &mesh_size,
                              double      &scf_mixing,
                              std::string &mixing_scheme, // New parameter
                              dftfe::Int  &polynomial_order,
                              double      &tolerance,
                              std::string &xc,
                              double      &atom_ball_radius,
                              dftfe::uInt &num_kohn_sham,
                              std::string &orthogonalization_type,
                              // New parameters
                              dftfe::uInt &wfc_block_size,
                              dftfe::uInt &cheby_wfc_block_size,
                              dftfe::Int  &smeared_nuclear_charges,
                              dftfe::Int  &use_group_symmetry,
                              dftfe::Int  &use_time_reversal_symmetry,
                              dftfe::Int  &mixing_history,
                              dftfe::Int  &max_scf_iterations,
                              dftfe::Int  &dispersion_correction_type,
                              dftfe::Int  &pseudopotential_calculation,
                              dftfe::Int  &verbosity,
                              bool        &use_device,
                              bool        &keep_scratch,   // New parameter
                              bool        &compute_forces, // New parameter
                              bool        &compute_stress, // New parameter
                              std::string &cmd)
  {
    // Simple parsing
    if (json.find("\"cmd\": \"exit\"") != std::string::npos || json == "EXIT")
      {
        cmd = "exit";
        return;
      }
    cmd = "run";

    coords  = parse_matrix(json, "coords");
    cell    = parse_matrix(json, "cell");
    numbers = parse_int_array(json, "numbers");
    pbc     = parse_bool_array(json, "pbc");

    // Parse parameters with defaults (though Python should send them)
    mp_grid = parse_int_array(json, "mp_grid");

    mp_grid_shift = parse_int_array(json, "mp_grid_shift");

    spin_polarized = parse_scalar<dftfe::Int>(json, "spin_polarized", -1);
    start_magnetization =
      parse_scalar<double>(json, "start_magnetization", -1.0);
    fermi_temp       = parse_scalar<double>(json, "fermi_temp", -1.0);
    npkpt            = parse_scalar<dftfe::uInt>(json, "npkpt", 999999);
    mesh_size        = parse_scalar<double>(json, "mesh_size", -1.0);
    scf_mixing       = parse_scalar<double>(json, "scf_mixing", -1.0);
    mixing_scheme    = parse_string(json, "mixing_scheme", "Unprovided");
    polynomial_order = parse_scalar<dftfe::Int>(json, "polynomial_order", -1);
    tolerance        = parse_scalar<double>(json, "tolerance", -1.0);
    xc               = parse_string(json, "xc", "Unprovided");
    atom_ball_radius = parse_scalar<double>(json, "atom_ball_radius", -1.0);
    num_kohn_sham    = parse_scalar<dftfe::uInt>(json, "num_kohn_sham", 999999);
    orthogonalization_type =
      parse_string(json, "orthogonalization_type", "Unprovided");

    // New parameters parsing
    wfc_block_size = parse_scalar<dftfe::uInt>(json, "wfc_block_size", 999999);
    cheby_wfc_block_size =
      parse_scalar<dftfe::uInt>(json, "cheby_wfc_block_size", 999999);

    smeared_nuclear_charges =
      parse_scalar<dftfe::Int>(json, "smeared_nuclear_charges", -1);
    use_group_symmetry = parse_scalar<dftfe::Int>(json, "use_group_symmetry", -1);
    use_time_reversal_symmetry =
      parse_scalar<dftfe::Int>(json, "use_time_reversal_symmetry", -1);
    mixing_history = parse_scalar<dftfe::Int>(json, "mixing_history", -1);
    max_scf_iterations =
      parse_scalar<dftfe::Int>(json, "max_scf_iterations", -1);
    // New optional parameter for debugging
    // New optional parameter for debugging
    keep_scratch = parse_scalar<bool>(json, "keep_scratch", false);
    dispersion_correction_type =
      parse_scalar<dftfe::Int>(json, "dispersion_correction_type", -1);
    pseudopotential_calculation =
      parse_scalar<dftfe::Int>(json, "pseudopotential_calculation", -1);

    verbosity  = parse_scalar<dftfe::Int>(json, "verbosity", -1);
    use_device = parse_scalar<bool>(json, "use_device", false);
    compute_forces =
      parse_scalar<bool>(json, "compute_forces", true); // Default True
    compute_stress = parse_scalar<bool>(json, "compute_stress", false);
  }

  std::string
  SocketDriver::format_response(double                                  energy,
                                const std::vector<std::vector<double>> &forces,
                                const std::vector<std::vector<double>> &stress)
  {
    std::stringstream ss;
    ss << std::scientific << std::setprecision(16);
    ss << "{\"energy\": " << energy << ", \"forces\": [";
    for (size_t i = 0; i < forces.size(); ++i)
      {
        ss << "[" << forces[i][0] << "," << forces[i][1] << "," << forces[i][2]
           << "]";
        if (i < forces.size() - 1)
          ss << ",";
      }
    ss << "], \"stress\": [";
    for (size_t i = 0; i < stress.size(); ++i)
      {
        ss << "[" << stress[i][0] << "," << stress[i][1] << "," << stress[i][2]
           << "]"; // Assuming 3x3 or similar
        if (i < stress.size() - 1)
          ss << ",";
      }
    ss << "]}";
    return ss.str();
  }

  // Helper for 3x3 matrix inverse
  std::vector<std::vector<double>>
  invert_3x3(const std::vector<std::vector<double>> &m)
  {
    double det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
                 m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                 m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    double invDet = 1.0 / det;

    std::vector<std::vector<double>> minv(3, std::vector<double>(3));

    minv[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * invDet;
    minv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invDet;
    minv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invDet;

    minv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invDet;
    minv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invDet;
    minv[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * invDet;

    minv[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * invDet;
    minv[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * invDet;
    minv[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * invDet;

    return minv;
  }

  // Helper for matrix multiplication
  std::vector<std::vector<double>>
  mat_mul(const std::vector<std::vector<double>> &A,
          const std::vector<std::vector<double>> &B)
  {
    std::vector<std::vector<double>> C(3, std::vector<double>(3, 0.0));
    for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
          {
            for (int k = 0; k < 3; ++k)
              {
                C[i][j] += A[i][k] * B[k][j];
              }
          }
      }
    return C;
  }

  void
  SocketDriver::run()
  {
    // We do NOT initialize dftfeWrapper with paramFile here anymore.
    // We wait for the first socket message to get the configuration.

    if (rank == 0)
      std::cout << "SocketDriver: Starting server..." << std::endl;

    connect_socket();
    if (rank == 0)
      std::cout << "SocketDriver: Connected to socket." << std::endl;

    // dftfeWrapper instance (default constructed, empty)
    dftfeWrapper dft;

    bool initialized = false;

    while (true)
      {
        if (rank == 0)
          std::cout << "SocketDriver: Waiting for data..." << std::endl;
        std::string json = receive_data();
        if (rank == 0)
          std::cout << "SocketDriver: Received data." << std::endl;

        std::vector<std::vector<double>> new_coords;
        std::vector<std::vector<double>> new_cell;
        std::vector<dftfe::uInt>         numbers;
        std::vector<bool>                pbc;
        std::vector<dftfe::uInt>         mp_grid;
        std::vector<dftfe::uInt>         mp_grid_shift;
        dftfe::Int                       spin_polarized;
        double                           start_magnetization;
        double                           fermi_temp;
        dftfe::uInt                      npkpt;
        double                           mesh_size;
        double                           scf_mixing;
        std::string                      mixing_scheme;
        dftfe::Int                       polynomial_order;
        double                           tolerance;
        std::string                      xc;
        double                           atom_ball_radius;
        dftfe::uInt                      num_kohn_sham;
        std::string                      orthogonalization_type;
        dftfe::Int                       verbosity;
        bool                             use_device;
        std::string                      cmd;

        // New parameters
        dftfe::uInt wfc_block_size;
        dftfe::uInt cheby_wfc_block_size;
        dftfe::Int  smeared_nuclear_charges;
        dftfe::Int  use_group_symmetry;
        dftfe::Int  use_time_reversal_symmetry;
        dftfe::Int  mixing_history;
        dftfe::Int  max_scf_iterations;
        dftfe::Int  dispersion_correction_type;
        dftfe::Int  pseudopotential_calculation;
        bool        keep_scratch;   // New variable
        bool        compute_forces; // New variable
        bool        compute_stress; // New variable

        parse_request(json,
                      new_coords,
                      new_cell,
                      numbers,
                      pbc,
                      mp_grid,
                      mp_grid_shift,
                      spin_polarized,
                      start_magnetization,
                      fermi_temp,
                      npkpt,
                      mesh_size,
                      scf_mixing,
                      mixing_scheme,
                      polynomial_order,
                      tolerance,
                      xc,
                      atom_ball_radius,
                      num_kohn_sham,
                      orthogonalization_type,
                      wfc_block_size,
                      cheby_wfc_block_size,
                      smeared_nuclear_charges,
                      use_group_symmetry,
                      use_time_reversal_symmetry,
                      mixing_history,
                      max_scf_iterations,
                      dispersion_correction_type,
                      pseudopotential_calculation,
                      verbosity,
                      use_device,
                      keep_scratch,
                      compute_forces, // Pass new param
                      compute_stress,
                      cmd);

        if (cmd == "exit")
          {
            if (rank == 0)
              std::cout << "SocketDriver: Received exit command." << std::endl;
            break;
          }

        if (!initialized)
          {
            // First run: Initialize DFT-FE with explicit data
            if (rank == 0 && verbosity >= 1)
              {
                  std::cout
                    << "=========================================================================================================="
                    << std::endl;
                  std::cout
                    << "=========================================================================================================="
                    << std::endl;
                  std::cout
                    << "			Welcome to the ASE interface of Open Source program DFT-FE version	1.1.0-pre		        "
                    << std::endl;
                  std::cout
                    << "This is a python interface for C++ code for materials modeling from first principles using Kohn-Sham density functional theory."
                    << std::endl;
                  std::cout
                    << "DFT-FE is a real-space code for periodic, semi-periodic and non-periodic pseudopotential"
                    << std::endl;
                  std::cout
                    << "and all-electron calculations, and is based on adaptive finite-element discretization."
                    << std::endl;
                  std::cout
                    << "For further details, and citing, please refer to our website: https://sites.google.com/umich.edu/dftfe"
                    << std::endl;
                  std::cout
                    << "=========================================================================================================="
                    << std::endl;
                  std::cout
                    << " DFT-FE Mentors and Development leads (alphabetically) :									"
                    << std::endl;
                  std::cout << "														" << std::endl;
                  std::cout << " Sambit Das               - University of Michigan, USA"
                            << std::endl;
                  std::cout << " Vikram Gavini            - University of Michigan, USA"
                            << std::endl;
                  std::cout
                    << " Phani Motamarri          - Indian Institute of Science, India"
                    << std::endl;
                  std::cout
                    << " (A complete list of the many authors that have contributed to DFT-FE can be found in the authors file)"
                    << std::endl;
                  std::cout
// Adding Mehul as the lead for ASE interface
                    << "=========================================================================================================="
                    << std::endl;
                  std::cout
                    << " ASE interface lead:                                                                                      "
                    << std::endl;
                  std::cout
                    << " Mehul Darak            - Indian Institute of Science, Bangalore"
                    << std::endl;
                  std::cout
                    << "=========================================================================================================="
                    << std::endl;
                  std::cout
                    << " 	     Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE authors         "
                    << std::endl;
                  std::cout
                    << " 			DFT-FE is published under [LGPL v2.1 or newer] 				"
                    << std::endl;
                  std::cout
                    << "=========================================================================================================="
                    << std::endl;
                  std::cout
                    << "=========================================================================================================="
                    << std::endl;
                  std::cout << " DFT-FE branch: " << GIT_BRANCH
                            << ", commit: " << GIT_COMMIT << std::endl;
                  std::cout << " compiled ";
            #  ifdef DFTFE_WITH_DEVICE
                  std::cout << "with GPU support, ";
            #    ifdef DFTFE_WITH_DEVICE_LANG_CUDA
                  std::cout << "using CUDA, ";
            #    elif DFTFE_WITH_DEVICE_LANG_HIP
                  std::cout << "using HIP, ";
            #    endif
            #    if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
                  std::cout << "with device-aware MPI support, ";
            #    endif
            #    if defined(DFTFE_WITH_CUDA_NCCL)
                  std::cout << "with NCCL support, ";
            #    endif
            #    if defined(DFTFE_WITH_HIP_RCCL)
                  std::cout << "with RCCL support, ";
            #    endif
            #  else
                  std::cout << "without GPU support, ";
            #  endif
            #  ifdef _OPENMP
                  std::cout << "with OpenMP support, ";
            #  endif
            #  ifdef DFTFE_WITH_64BIT_INT
                  std::cout << "with 64 bit integers, ";
            #  else
                  std::cout << "with 32 bit integers, ";
            #  endif
            #  ifdef DFTFE_WITH_HIGHERQUAD_PSP
                  std::cout << "and with HIGHERQUAD_PSP" << std::endl;
            #  else
                  std::cout << "and without HIGHERQUAD_PSP" << std::endl;
            #  endif
                  std::cout
                    << "=========================================================================================================="
                    << std::endl;
              }

            if (rank == 0)
              std::cout
                << "SocketDriver: Initializing dftfeWrapper with explicit data..."
                << std::endl;

            dft.reinit(comm,
                       use_device,
                       new_coords,
                       numbers,
                       new_cell,
                       pbc,
                       mp_grid,
                       mp_grid_shift,
                       spin_polarized,
                       start_magnetization,
                       fermi_temp,
                       npkpt,
                       mesh_size,
                       scf_mixing,
                       mixing_scheme,
                       polynomial_order,
                       tolerance,
                       xc,
                       atom_ball_radius,
                       num_kohn_sham,
                       orthogonalization_type,
                       wfc_block_size,
                       cheby_wfc_block_size,
                       smeared_nuclear_charges,
                       use_group_symmetry,
                       use_time_reversal_symmetry,
                       mixing_history,
                       max_scf_iterations,
                       dispersion_correction_type,
                       pseudopotential_calculation,
                       verbosity,
                       use_device,   // setDeviceToMPITaskBindingInternally
                       keep_scratch, // keepScratch
                       compute_forces,
                       compute_stress);

            initialized = true;
            if (rank == 0)
              std::cout << "SocketDriver: dftfeWrapper initialized."
                        << std::endl;

            // For the first run, we just compute.
            // The atoms are already at new_coords.
          }
        else
          {
            // Subsequent runs: Update positions
            auto current_coords = dft.getAtomPositionsCart();
            std::vector<std::vector<double>> displacements;

            if (new_coords.size() != current_coords.size())
              {
                if (rank == 0)
                  std::cerr << "Atom count mismatch!" << std::endl;
                break;
              }

            double max_disp = 0.0;
            for (size_t i = 0; i < new_coords.size(); ++i)
              {
                std::vector<double> disp(3);
                for (int j = 0; j < 3; ++j)
                  {
                    disp[j]  = new_coords[i][j] - current_coords[i][j];
                    max_disp = std::max(max_disp, std::abs(disp[j]));
                  }
                displacements.push_back(disp);
              }

            if (max_disp > 1e-6)
              {
                if (rank == 0)
                  std::cout
                    << "SocketDriver: Updating atom positions... Max disp: "
                    << max_disp << std::endl;
                dft.updateAtomPositions(displacements);
              }
            else
              {
                if (rank == 0)
                  std::cout
                    << "SocketDriver: Displacements small, skipping update."
                    << std::endl;
              }

            // Handle cell deformation (simplified for now, similar to before)
            // ... (omitting complex cell deformation logic for this step to
            // focus on basic integration as requested) If needed, we can re-add
            // the deformation logic.
          }

        // Compute
        if (rank == 0)
          std::cout << "SocketDriver: Computing free energy... (Forces: "
                    << (compute_forces ? "ON" : "OFF")
                    << ", Stress: " << (compute_stress ? "ON" : "OFF") << ")"
                    << std::endl;
        auto result = dft.computeDFTFreeEnergy(compute_forces, compute_stress);
        if (rank == 0)
          std::cout << "SocketDriver: Computation complete." << std::endl;

        double energy = std::get<0>(result);
        auto   forces = dft.getForcesAtoms();
        auto   stress = dft.getCellStress();

        std::string response = format_response(energy, forces, stress);
        send_data(response);
        if (rank == 0)
          std::cout << "SocketDriver: Response sent." << std::endl;
      }

    close_socket();
  }

} // namespace dftfe
