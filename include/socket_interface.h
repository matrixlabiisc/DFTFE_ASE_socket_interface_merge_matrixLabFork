#ifndef SOCKET_INTERFACE_H
#define SOCKET_INTERFACE_H

// Mehul: Created this header for Socket Interface
#include <string>
#include <vector>
#include <mpi.h>
#include "dftfeWrapper.h"

namespace dftfe
{

  class SocketDriver
  {
  public:
    SocketDriver(const std::string &host, int port, MPI_Comm comm);
    ~SocketDriver();
    void
    run();

  private:
    std::string host;
    int         port;
    MPI_Comm    comm;
    int         rank;
    int         sockfd;

    void
    connect_socket();
    void
    close_socket();
    void
    send_data(const std::string &data);
    std::string
    receive_data();

    // Simple JSON helpers
    void
    parse_request(
      const std::string                &json,
      std::vector<std::vector<double>> &coords,
      std::vector<std::vector<double>> &cell,
      std::vector<dftfe::uInt>         &numbers,
      std::vector<bool>                &pbc,
      std::vector<dftfe::uInt>         &mp_grid,
      std::vector<dftfe::uInt>         &mp_grid_shift,
      bool                             &spin_polarized,
      double                           &start_magnetization,
      double                           &fermi_temp,
      dftfe::uInt                      &npkpt,
      double                           &mesh_size,
      double                           &scf_mixing,
      std::string                      &mixing_scheme, // New parameter
      dftfe::Int                       &polynomial_order,
      double                           &tolerance,
      std::string                      &xc,
      double                           &atom_ball_radius,
      dftfe::uInt                      &num_kohn_sham,
      std::string                      &orthogonalization_type,
      // New parameters
      dftfe::uInt &wfc_block_size,
      dftfe::uInt &cheby_wfc_block_size,
      bool        &smeared_nuclear_charges,
      bool        &use_group_symmetry,
      bool &use_time_reversal_symmetry, // Mehul -> added time reversal symmetry
      dftfe::Int  &mixing_history,
      dftfe::Int  &max_scf_iterations,
      dftfe::Int  &dispersion_correction_type,
      bool        &pseudopotential_calculation,
      dftfe::Int  &verbosity,
      bool        &use_device,
      bool        &keep_scratch,   // New parameter
      bool        &compute_forces, // New parameter: ION FORCE control
      bool        &compute_stress, // New parameter
      std::string &cmd);

    std::string
    format_response(double                                  energy,
                    const std::vector<std::vector<double>> &forces,
                    const std::vector<std::vector<double>> &stress);
  };

} // namespace dftfe
#endif
