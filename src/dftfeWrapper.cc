// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// @author Sambit Das
//


// deal.II header
//
#include <dftfe/config.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/multithread_info.h>
#include <p4est_bits.h>

#ifdef USE_PETSC
#  include <petscsys.h>
#  include <slepcsys.h>
#endif

//
// C++ headers
//
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <chrono>
#include <unistd.h> // For sleep
#include <sys/time.h>
#include <ctime>

#include <dftfe/dft.h>
#include <dftfe/dftParameters.h>
#include <dftfe/deviceKernelsGeneric.h>
#include <dftfe/dftUtils.h>
#include <dftfe/dftfeWrapper.h>
#include <dftfe/fileReaders.h>
#include <dftfe/PeriodicTable.h>
#include <dftfe/MemorySpaceType.h>

namespace dftfe
{
  // Defined in src/dft/moveAtoms.cc. Declared here the same way restart.cc
  // declares it, so reinit() folds out-of-cell atoms with DFT-FE's own
  // periodic wrap rather than a caller-side reimplementation.
  namespace internal
  {
    std::vector<double>
    wrapAtomsAcrossPeriodicBc(const dealii::Point<3>    &cellCenteredCoord,
                              const dealii::Point<3>    &corner,
                              const std::vector<double> &latticeVectors,
                              const std::vector<bool>   &periodicBc);
  } // namespace internal

  namespace internalWrapper
  {
    dftfe::Int
    divisor_closest(dftfe::Int totalSize, dftfe::Int desiredDivisor)
    {
      dftfe::Int i;
      for (i = desiredDivisor; i >= 1; --i)
        {
          if (totalSize % i == 0 && i <= desiredDivisor)
            return i;
        }
      return 1;
    }


    /**
     * @brief Minimal in-memory editor for deal.II .prm files (Mehul).
     *
     * .prm files are line oriented and *nested*: "subsection <name>" / "end"
     * open and close scopes, "set <key> = <value>" assigns within the current
     * scope. Editing them with `sed` cannot see that structure, which produced
     * two silent-corruption modes in socket mode:
     *
     *   - a key declared in more than one subsection could not be targeted --
     *     "TOLERANCE" lives in both "SCF parameters" (1e-4) and "Poisson problem
     *     parameters" (1e-8), so writing one deleted the other;
     *   - a substring pattern matched longer keys -- "set MAXIMUM ITERATIONS "
     *     also matched "set MAXIMUM ITERATIONS HELMHOLTZ".
     *
     * In both cases the lost entry fell back to its built-in default with no
     * diagnostic, because dftParameters::parse_parameters runs deal.II's parser
     * with skip_undefined=true. This class tracks the subsection stack so every
     * assignment resolves to exactly one entry, and reports when it does not.
     */
    class PrmFile
    {
    public:
      explicit PrmFile(const std::string &path)
      {
        std::ifstream in(path);
        std::string   line;
        while (std::getline(in, line))
          d_lines.push_back(line);
      }

      /**
       * Assign @p key inside @p section, where @p section is the *innermost*
       * subsection name ("" for a top-level entry). Returns false if the entry
       * is not present in the template, in which case nothing is written.
       */
      bool
      set(const std::string &section,
          const std::string &key,
          const std::string &value)
      {
        std::vector<std::string> stack;
        for (auto &line : d_lines)
          {
            const std::string s = trim(line);
            if (s.rfind("subsection", 0) == 0)
              {
                stack.push_back(trim(s.substr(std::string("subsection").size())));
                continue;
              }
            if (s == "end")
              {
                if (!stack.empty())
                  stack.pop_back();
                continue;
              }
            if (s.rfind("set", 0) != 0)
              continue;

            const std::size_t eq = s.find('=');
            if (eq == std::string::npos)
              continue;
            if (trim(s.substr(3, eq - 3)) != key)
              continue;

            const std::string here = stack.empty() ? "" : stack.back();
            if (here != section)
              continue;

            line = line.substr(0, line.find_first_not_of(" \t")) + "set " + key +
                   "=" + value;
            return true;
          }
        return false;
      }

      /** Convenience overload for top-level entries. */
      bool
      set(const std::string &key, const std::string &value)
      {
        return set("", key, value);
      }

      void
      write(const std::string &path) const
      {
        std::ofstream out(path);
        for (const auto &line : d_lines)
          out << line << "\n";
      }

    private:
      static std::string
      trim(const std::string &s)
      {
        const std::size_t b = s.find_first_not_of(" \t\r\n");
        if (b == std::string::npos)
          return "";
        return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
      }

      std::vector<std::string> d_lines;
    };


    template <dftfe::utils::MemorySpace memory>
    void
    create_dftfe(const MPI_Comm       &mpi_comm_parent,
                 const MPI_Comm       &mpi_comm_domain,
                 const MPI_Comm       &interpoolcomm,
                 const MPI_Comm       &interBandGroupComm,
                 const std::string    &scratchFolderName,
                 dftfe::dftParameters &dftParams,
                 dftBase             **dftfeBaseDoublePtr)
    {
      *dftfeBaseDoublePtr = new dftfe::dftClass<memory>(mpi_comm_parent,
                                                        mpi_comm_domain,
                                                        interpoolcomm,
                                                        interBandGroupComm,
                                                        scratchFolderName,
                                                        dftParams);
    }
  } // namespace internalWrapper

  void
  dftfeWrapper::globalHandlesInitialize(const MPI_Comm &mpi_comm_world)
  {
    sc_init(mpi_comm_world, 0, 0, nullptr, SC_LP_SILENT);
    p4est_init(nullptr, SC_LP_SILENT);

#ifdef USE_PETSC
    SlepcInitializeNoArguments();
    PetscPopSignalHandler();
#endif

    if (elpa_init(ELPA_API_VERSION) != ELPA_OK)
      {
        fprintf(stderr, "Error: ELPA API version not supported.");
        exit(1);
      }
    dealii::MultithreadInfo::set_thread_limit(1);
  }

  void
  dftfeWrapper::globalHandlesFinalize()
  {
    sc_finalize();

#ifdef USE_PETSC
    SlepcFinalize();
#endif

    int error;
    elpa_uninit(&error);
    AssertThrow(error == ELPA_OK,
                dealii::ExcMessage("DFT-FE Error: elpa error."));
  }


  //
  // constructor
  //
  dftfeWrapper::dftfeWrapper()
    : d_dftfeBasePtr(nullptr)
    , d_dftfeParamsPtr(nullptr)
    , d_mpi_comm_parent(MPI_COMM_NULL)
    , d_isDeviceToMPITaskBindingSetInternally(false)
  {}

  //
  // constructor
  //
  dftfeWrapper::dftfeWrapper(const std::string parameter_file,
                             const MPI_Comm   &mpi_comm_parent,
                             const bool        printParams,
                             const bool setDeviceToMPITaskBindingInternally,
                             const std::string mode,
                             const std::string restartFilesPath,
                             const dftfe::Int  _verbosity,
                             const bool        useDevice)
    : d_dftfeBasePtr(nullptr)
    , d_dftfeParamsPtr(nullptr)
    , d_mpi_comm_parent(MPI_COMM_NULL)
    , d_isDeviceToMPITaskBindingSetInternally(false)
  {
    reinit(parameter_file,
           mpi_comm_parent,
           printParams,
           setDeviceToMPITaskBindingInternally,
           mode,
           restartFilesPath,
           _verbosity,
           useDevice);
  }


  //
  // constructor
  //
  dftfeWrapper::dftfeWrapper(const std::string parameter_file,
                             const std::string restartCoordsFile,
                             const std::string restartDomainVectorsFile,
                             const MPI_Comm   &mpi_comm_parent,
                             const bool        printParams,
                             const bool setDeviceToMPITaskBindingInternally,
                             const std::string mode,
                             const std::string restartFilesPath,
                             const dftfe::Int  _verbosity,
                             const bool        useDevice,
                             const bool        isScfRestart)
    : d_dftfeBasePtr(nullptr)
    , d_dftfeParamsPtr(nullptr)
    , d_mpi_comm_parent(MPI_COMM_NULL)
    , d_isDeviceToMPITaskBindingSetInternally(false)
  {
    reinit(parameter_file,
           restartCoordsFile,
           restartDomainVectorsFile,
           mpi_comm_parent,
           printParams,
           setDeviceToMPITaskBindingInternally,
           mode,
           restartFilesPath,
           _verbosity,
           useDevice,
           isScfRestart);
  }



  //
  // constructor
  //
  dftfeWrapper::dftfeWrapper(
    const MPI_Comm                        &mpi_comm_parent,
    const bool                             useDevice,
    const std::vector<std::vector<double>> atomicPositionsCart,
    const std::vector<dftfe::uInt>         atomicNumbers,
    const std::vector<std::vector<double>> cell,
    const std::vector<bool>                pbc,
    const std::vector<dftfe::uInt>         mpGrid,
    const std::vector<dftfe::uInt>         mpGridShift,
    const dftfe::Int                       spinPolarizedDFT,
    const double                           startMagnetization,
    const double                           fermiDiracSmearingTemp,
    const dftfe::uInt                      npkpt,
    const double                           meshSize,
    const double                           scfMixingParameter,
    const dftfe::Int                       polynomialOrder,
    const double                           tolerance,
    const std::string                      xc,
    const dftfe::Int                       verbosity,
    const bool                             setDeviceToMPITaskBindingInternally)
    : d_dftfeBasePtr(nullptr)
    , d_dftfeParamsPtr(nullptr)
    , d_mpi_comm_parent(MPI_COMM_NULL)
    , d_isDeviceToMPITaskBindingSetInternally(false)
  {
    reinit(mpi_comm_parent,
           useDevice,
           atomicPositionsCart,
           atomicNumbers,
           cell,
           pbc,
           mpGrid,
           mpGridShift,
           spinPolarizedDFT,
           startMagnetization,
           fermiDiracSmearingTemp,
           npkpt,
           meshSize,
           scfMixingParameter,
           "Anderson", // Default mixingScheme
           polynomialOrder,
           tolerance,
           xc,
           0.0,    // Default atomBallRadius
           0,      // Default numKohnSham
           "Auto", // Default orthogonalizationType
           verbosity,
           setDeviceToMPITaskBindingInternally);
  }


  dftfeWrapper::~dftfeWrapper()
  {
    clear();
  }

  void
  dftfeWrapper::reinit(const std::string parameter_file,
                       const MPI_Comm   &mpi_comm_parent,
                       const bool        printParams,
                       const bool        setDeviceToMPITaskBindingInternally,
                       const std::string mode,
                       const std::string restartFilesPath,
                       const dftfe::Int  _verbosity,
                       const bool        useDevice)
  {
    clear();
    if (mpi_comm_parent != MPI_COMM_NULL)
      MPI_Comm_dup(mpi_comm_parent, &d_mpi_comm_parent);
    createScratchFolder();

    if (d_mpi_comm_parent != MPI_COMM_NULL)
      {
        d_dftfeParamsPtr = new dftfe::dftParameters;
        d_dftfeParamsPtr->parse_parameters(parameter_file,
                                           d_mpi_comm_parent,
                                           printParams,
                                           mode,
                                           restartFilesPath,
                                           _verbosity,
                                           useDevice);
      }
    initialize(setDeviceToMPITaskBindingInternally, useDevice);
  }


  void
  dftfeWrapper::reinit(const std::string parameter_file,
                       const std::string restartCoordsFile,
                       const std::string restartDomainVectorsFile,
                       const MPI_Comm   &mpi_comm_parent,
                       const bool        printParams,
                       const bool        setDeviceToMPITaskBindingInternally,
                       const std::string mode,
                       const std::string restartFilesPath,
                       const dftfe::Int  _verbosity,
                       const bool        useDevice,
                       const bool        isScfRestart)
  {
    clear();
    if (mpi_comm_parent != MPI_COMM_NULL)
      MPI_Comm_dup(mpi_comm_parent, &d_mpi_comm_parent);

    createScratchFolder();

    if (d_mpi_comm_parent != MPI_COMM_NULL)
      {
        d_dftfeParamsPtr = new dftfe::dftParameters;
        d_dftfeParamsPtr->parse_parameters(parameter_file,
                                           d_mpi_comm_parent,
                                           printParams,
                                           mode,
                                           restartFilesPath,
                                           _verbosity,
                                           useDevice);
        d_dftfeParamsPtr->coordinatesFile           = restartCoordsFile;
        d_dftfeParamsPtr->domainBoundingVectorsFile = restartDomainVectorsFile;
        d_dftfeParamsPtr->loadQuadData =
          d_dftfeParamsPtr->loadQuadData && isScfRestart;
      }
    initialize(setDeviceToMPITaskBindingInternally, useDevice);
  }


  void
  // Mehul: Modified reinit to accept new parameters (ASE)
  dftfeWrapper::reinit(
    const MPI_Comm                        &mpi_comm_parent,
    const bool                             useDevice,
    const std::vector<std::vector<double>> atomicPositionsCart,
    const std::vector<dftfe::uInt>         atomicNumbers,
    const std::vector<std::vector<double>> cell,
    const std::vector<bool>                pbc,
    const std::vector<dftfe::uInt>         mpGrid,
    const std::vector<dftfe::uInt>         mpGridShift,
    const dftfe::Int                       spinPolarizedDFT,
    const double                           startMagnetization,
    const double                           fermiDiracSmearingTemp,
    const dftfe::uInt                      npkpt,
    const double                           meshSize,
    const double                           scfMixingParameter,
    const std::string                      mixingScheme,
    const dftfe::Int                       polynomialOrder,
    const double                           tolerance,
    const std::string                      xc,
    const double                           atomBallRadius,
    const dftfe::uInt                      numBands,
    const std::string                      orthogonalizationType,
    // New parameters
    const dftfe::uInt wfcBlockSize,
    const dftfe::uInt chebyWfcBlockSize,
    const dftfe::Int  densityQuadratureRule,
    const dftfe::Int  useSinglePrecCheby,
    const dftfe::Int  smearedNuclearCharges,
    const dftfe::Int  useGroupSymmetry,
    const dftfe::Int  useTimeReversalSymmetry,
    const dftfe::Int  mixingHistory,
    const dftfe::Int  maxSCFIterations,
    const dftfe::Int  dispersionCorrectionType,
    const dftfe::Int  pseudopotentialCalculation,
    const std::string pseudopotentialFilename,
    const dftfe::Int  verbosity,
    const bool        setDeviceToMPITaskBindingInternally,
    const bool        keepScratch,
    const bool        computeIonForces,
    const bool        computeStress)
  {
    clear();
    if (mpi_comm_parent != MPI_COMM_NULL)
      {
        dftfe::Int ierr = MPI_Comm_dup(mpi_comm_parent, &d_mpi_comm_parent);
        int        rank;
        MPI_Comm_rank(d_mpi_comm_parent, &rank);
        if (ierr != 0)
          {
            throw std::runtime_error("MPI_Comm_dup failed.");
          }
      }

    createScratchFolder();

    if (d_mpi_comm_parent != MPI_COMM_NULL)
      {
        const dftfe::Int totalMPIProcesses =
          dealii::Utilities::MPI::n_mpi_processes(d_mpi_comm_parent);

        std::string parameter_file_path =
          d_scratchFolderName + "/parameterFile.prm";

        if (dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent) == 0)
          {
            AssertThrow(
              atomicPositionsCart.size() == atomicNumbers.size(),
              dealii::ExcMessage(
                "DFT-FE Error:  Mismatch in sizes of atomicPositionsCart and atomicNumbers."));
            //
            // write pseudo.inp
            //
            std::set<dftfe::uInt> atomicNumbersSet;
            for (dftfe::uInt i = 0; i < atomicNumbers.size(); i++)
              atomicNumbersSet.insert(atomicNumbers[i]);

            std::vector<dftfe::uInt> atomicNumbersUniqueVec(
              atomicNumbersSet.size());
            std::copy(atomicNumbersSet.begin(),
                      atomicNumbersSet.end(),
                      atomicNumbersUniqueVec.begin());


            // Parse dictionary format strings for multi-element pseudo paths
            std::map<std::string, std::string> exactFileMap;
            std::string dftfePspPath = pseudopotentialFilename;
            if (dftfePspPath.find("DICT|") == 0)
              {
                std::string dictStr = dftfePspPath.substr(5);
                std::stringstream ss(dictStr);
                std::string pair;
                while (std::getline(ss, pair, '|'))
                  {
                    size_t colonPos = pair.find(':');
                    if (colonPos != std::string::npos)
                      {
                        std::string sym = pair.substr(0, colonPos);
                        std::string path = pair.substr(colonPos + 1);
                        exactFileMap[sym] = path;
                      }
                  }
              }
            else if (dftfePspPath == "Unprovided" || dftfePspPath.empty())
              {
                const char* envPath = getenv("DFTFE_PSP_PATH");
                if (envPath) dftfePspPath = std::string(envPath);
              }

            pseudoUtils::PeriodicTable periodicTable;
            const std::string          dftfePseudoFileName =
              d_scratchFolderName + "/pseudo.inp";
            std::ofstream dftfePseudoFile(dftfePseudoFileName);
            if (dftfePseudoFile.is_open())
              {
                for (dftfe::uInt irow = 0; irow < atomicNumbersUniqueVec.size();
                     ++irow)
                  {
                    std::string sym = periodicTable.symbol(atomicNumbersUniqueVec[irow]);
                    std::string upffilePath;
                    
                    if (exactFileMap.find(sym) != exactFileMap.end())
                      {
                        upffilePath = exactFileMap[sym];
                      }
                    else
                      {
                        upffilePath = dftfePspPath;
                        if (dftfePspPath.find(".upf") == std::string::npos && 
                            dftfePspPath.find(".psp8") == std::string::npos &&
                            dftfePspPath.find(".UPF") == std::string::npos)
                          {
                            upffilePath = dftfePspPath + "/" + sym + ".upf";
                          }
                      }

                    dftfePseudoFile
                      << std::to_string(atomicNumbersUniqueVec[irow]);
                    dftfePseudoFile << " ";
                    dftfePseudoFile << upffilePath;
                    dftfePseudoFile << "\n";
                  }

                dftfePseudoFile.close();
              }

            //
            // write coordinates.inp
            //
            std::map<dftfe::uInt, dftfe::uInt> atomicNumberToValenceNumberMap;

            for (dftfe::uInt i = 0; i < atomicNumbersUniqueVec.size(); i++)
              {
                std::string sym = periodicTable.symbol(atomicNumbersUniqueVec[i]);
                std::string upffilePath;
                if (exactFileMap.find(sym) != exactFileMap.end())
                  {
                    upffilePath = exactFileMap[sym];
                  }
                else
                  {
                    upffilePath = dftfePspPath;
                    if (dftfePspPath.find(".upf") == std::string::npos && 
                        dftfePspPath.find(".psp8") == std::string::npos &&
                        dftfePspPath.find(".UPF") == std::string::npos)
                      {
                        upffilePath = dftfePspPath + "/" + sym + ".upf";
                      }
                  }
                std::ifstream upffile(upffilePath);
                double        valenceNumber = 0;
                std::string   line;
                while (getline(upffile, line))
                  {
                    if (line.find("z_valence=") == std::string::npos)
                      continue;
                    std::istringstream ss(line);
                    std::string        dummy1;
                    std::string        dummy2;
                    ss >> dummy1 >> valenceNumber >> dummy2;
                    break;
                  }
                atomicNumberToValenceNumberMap[atomicNumbersUniqueVec[i]] =
                  std::round(valenceNumber);
              }

            std::vector<std::vector<double>> dftfeCoordinates(
              atomicPositionsCart.size(), std::vector<double>(5, 0));

            std::vector<double> cellVectorsFlattened(9, 0.0);
            for (dftfe::uInt idim = 0; idim < 3; idim++)
              for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
                cellVectorsFlattened[3 * idim + jdim] = cell[idim][jdim];

            if (pbc[0] == false && pbc[1] == false && pbc[2] == false)
              {
                std::vector<double> shift(3, 0.0);
                for (dftfe::uInt idim = 0; idim < 3; idim++)
                  {
                    shift[idim] = 0;
                    for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
                      shift[idim] -= cell[jdim][idim] / 2.0;
                  }
                for (dftfe::uInt i = 0; i < dftfeCoordinates.size(); i++)
                  {
                    dftfeCoordinates[i][0] = atomicNumbers[i];
                    dftfeCoordinates[i][1] =
                      atomicNumberToValenceNumberMap[atomicNumbers[i]];

                    std::vector<double> coord(3, 0.0);
                    coord[0] = atomicPositionsCart[i][0];
                    coord[1] = atomicPositionsCart[i][1];
                    coord[2] = atomicPositionsCart[i][2];

                    std::vector<double> frac =
                      dftUtils::getFractionalCoordinates(cellVectorsFlattened,
                                                         coord);
                    for (dftfe::uInt idim = 0; idim < 3; idim++)
                      AssertThrow(
                        frac[idim] > 1e-7 && frac[idim] < (1.0 - 1e-7),
                        dealii::ExcMessage(
                          "DFT-FE Error: all coordinates are not inside the cell. Please check input atomicPositionsCart."));

                    dftfeCoordinates[i][2] =
                      atomicPositionsCart[i][0] + shift[0];
                    dftfeCoordinates[i][3] =
                      atomicPositionsCart[i][1] + shift[1];
                    dftfeCoordinates[i][4] =
                      atomicPositionsCart[i][2] + shift[2];
                  }
              }
            else
              {
                for (dftfe::uInt i = 0; i < dftfeCoordinates.size(); i++)
                  {
                    dftfeCoordinates[i][0] = atomicNumbers[i];
                    dftfeCoordinates[i][1] =
                      atomicNumberToValenceNumberMap[atomicNumbers[i]];
                    std::vector<double> coord(3, 0.0);
                    coord[0] = atomicPositionsCart[i][0];
                    coord[1] = atomicPositionsCart[i][1];
                    coord[2] = atomicPositionsCart[i][2];

                    // Fold out-of-cell atoms back into [0,1] along periodic
                    // directions using DFT-FE's own periodic wrap -- the same
                    // routine updateAtomPositionsAndMoveMesh() and the restart
                    // path use. External drivers (ASE optimizers, MD wrappers)
                    // legitimately step an atom outside the cell; for a
                    // periodic direction that atom is its own image, so this
                    // is a relabelling and leaves the physics untouched.
                    //
                    // corner is the zero point because atomicPositionsCart is
                    // expressed relative to the cell origin, matching the
                    // two-argument getFractionalCoordinates() used previously
                    // here. Non-periodic directions are left alone and still
                    // assert, since folding them would move the atom through
                    // vacuum to a genuinely different structure.
                    dealii::Point<3> atomCoor;
                    atomCoor[0] = coord[0];
                    atomCoor[1] = coord[1];
                    atomCoor[2] = coord[2];

                    const dealii::Point<3> corner;

                    std::vector<bool> periodicBc(3, false);
                    for (dftfe::uInt idim = 0; idim < 3; idim++)
                      periodicBc[idim] = pbc[idim];

                    std::vector<double> frac =
                      internal::wrapAtomsAcrossPeriodicBc(atomCoor,
                                                          corner,
                                                          cellVectorsFlattened,
                                                          periodicBc);

                    // Mehul: this bound must be the one dft.cc actually
                    // enforces. initImageChargesUpdateKPoints() requires a
                    // non-periodic fractional coordinate to lie *strictly*
                    // inside (1e-6, 1-1e-6); the check here used to accept
                    // [-1e-7, 1+1e-7], so an atom sitting exactly on the face
                    // -- what ase.build.surface() produces, its bottom layer at
                    // z=0 -- passed here and then aborted much deeper in, with
                    // a message naming neither the atom nor the axis. Failing
                    // at the same threshold, at the point where the atom index
                    // is still in hand, costs nothing and rejects nothing that
                    // would otherwise have run.
                    for (dftfe::uInt idim = 0; idim < 3; idim++)
                      if (!periodicBc[idim])
                        AssertThrow(
                          frac[idim] > 1e-6 && frac[idim] < (1.0 - 1e-6),
                          dealii::ExcMessage(
                            "DFT-FE Error: atom " + std::to_string(i) +
                            " has fractional coordinate " +
                            std::to_string(frac[idim]) + " along non-periodic axis " +
                            std::to_string(idim) +
                            ", which is not strictly inside (0,1). There is no mesh "
                            "outside the cell along a non-periodic direction and no "
                            "periodic image to fold to, so the atom cannot be placed. "
                            "Translate the whole system inside the cell (a rigid shift "
                            "leaves energy and forces unchanged), or add vacuum. From "
                            "ASE this is handled automatically by dftfe_ase.geometry."));

                    dftfeCoordinates[i][2] = frac[0];
                    dftfeCoordinates[i][3] = frac[1];
                    dftfeCoordinates[i][4] = frac[2];
                  }
              }

            const std::string dftfeCoordsFileName =
              d_scratchFolderName + "/coordinates.inp";
            {
              std::ofstream coordsFile(dftfeCoordsFileName);
              for (const auto &row : dftfeCoordinates)
                {
                  for (size_t i = 0; i < row.size(); ++i)
                    {
                      if (i < 2) // atomic number and valence charge: integers
                        coordsFile << static_cast<int>(row[i]);
                      else       // fractional/Cartesian coordinates: fixed 16 d.p.
                        coordsFile << std::fixed << std::setprecision(16) << row[i];
                      coordsFile << (i == row.size() - 1 ? "" : " ");
                    }
                  coordsFile << "\n";
                }
              coordsFile.close();
            }

            //
            // write domainVectors.inp
            //
            const std::string dftfeCellFileName =
              d_scratchFolderName + "/domainVectors.inp";
            {
              std::ofstream cellFile(dftfeCellFileName);
              cellFile << std::fixed << std::setprecision(16);
              for (const auto &row : cell)
                {
                  for (size_t i = 0; i < row.size(); ++i)
                    {
                      cellFile << row[i] << (i == row.size() - 1 ? "" : " ");
                    }
                  cellFile << "\n";
                }
              cellFile.close();
            }



            std::string dftfePath = DFTFE_PATH;
            std::string sourceFilePath =
              dftfePath + "/helpers/parameterFile.prm";

            // Mehul: the template is generated from dftParameters.cc
            // declare_parameters() (see helpers/gen_parameter_template.py), so
            // every declared key is already present at its DFT-FE default with
            // the correct nesting. Injection below only ever REPLACES an entry;
            // a false return means the key does not exist in this build, which
            // is reported rather than silently dropped.
            internalWrapper::PrmFile prm(sourceFilePath);

            const bool reportPrm =
              verbosity >= 1 &&
              dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent) == 0;
            auto applyPrm = [&](const std::string &section,
                                const std::string &key,
                                const std::string &value) {
              if (!prm.set(section, key, value) && reportPrm)
                std::cout << "DFT-FE warning: .prm entry '" << key
                          << "' (subsection '" << section
                          << "') is not declared in this build - ignored."
                          << std::endl;
            };

            applyPrm("Geometry",
                     "NATOMS",
                     std::to_string(atomicPositionsCart.size()));
            applyPrm("Geometry",
                     "NATOM TYPES",
                     std::to_string(atomicNumbersUniqueVec.size()));
            applyPrm("Geometry",
                     "ATOMIC COORDINATES FILE",
                     d_scratchFolderName + "/coordinates.inp");
            applyPrm("Geometry",
                     "DOMAIN VECTORS FILE",
                     d_scratchFolderName + "/domainVectors.inp");
            applyPrm("DFT functional parameters",
                     "PSEUDOPOTENTIAL FILE NAMES LIST",
                     d_scratchFolderName + "/pseudo.inp");

            if (pbc.size() >= 3)
              {
                // CELL STRESS is only meaningful for a periodic cell.
                if (!pbc[0] && !pbc[1] && !pbc[2])
                  applyPrm("Optimization", "CELL STRESS", "false");

                applyPrm("Boundary conditions",
                         "PERIODIC1",
                         pbc[0] ? "true" : "false");
                applyPrm("Boundary conditions",
                         "PERIODIC2",
                         pbc[1] ? "true" : "false");
                applyPrm("Boundary conditions",
                         "PERIODIC3",
                         pbc[2] ? "true" : "false");
              }

            if (mpGrid.size() >= 3)
              {
                applyPrm("Monkhorst-Pack (MP) grid generation",
                         "SAMPLING POINTS 1",
                         std::to_string(mpGrid[0]));
                applyPrm("Monkhorst-Pack (MP) grid generation",
                         "SAMPLING POINTS 2",
                         std::to_string(mpGrid[1]));
                applyPrm("Monkhorst-Pack (MP) grid generation",
                         "SAMPLING POINTS 3",
                         std::to_string(mpGrid[2]));
              }

            if (mpGridShift.size() >= 3)
              {
                applyPrm("Monkhorst-Pack (MP) grid generation",
                         "SAMPLING SHIFT 1",
                         std::to_string(mpGridShift[0]));
                applyPrm("Monkhorst-Pack (MP) grid generation",
                         "SAMPLING SHIFT 2",
                         std::to_string(mpGridShift[1]));
                applyPrm("Monkhorst-Pack (MP) grid generation",
                         "SAMPLING SHIFT 3",
                         std::to_string(mpGridShift[2]));
              }

            if (spinPolarizedDFT != -1)
              applyPrm("DFT functional parameters",
                       "SPIN POLARIZATION",
                       std::to_string(spinPolarizedDFT));

            // Current DFT-FE declares TOTAL MAGNETIZATION only; the per-atom
            // START MAGNETIZATION entry the old sed also wrote has never been
            // declared in this branch, so it was always a no-op.
            if (startMagnetization != -1.0)
              applyPrm("DFT functional parameters",
                       "TOTAL MAGNETIZATION",
                       std::to_string(startMagnetization * 2 *
                                      atomicNumbersUniqueVec.size()));

            if (polynomialOrder != -1)
              applyPrm("Finite element mesh parameters",
                       "POLYNOMIAL ORDER",
                       std::to_string(polynomialOrder));

            if (tolerance != -1.0)
              {
                std::ostringstream oss;
                oss << std::scientific << std::setprecision(13) << tolerance;
                applyPrm("SCF parameters", "TOLERANCE", oss.str());
              }

            if (xc != "Unprovided")
              applyPrm("DFT functional parameters",
                       "EXCHANGE CORRELATION TYPE",
                       xc);

            if (fermiDiracSmearingTemp != -1.0)
              applyPrm("SCF parameters",
                       "TEMPERATURE",
                       std::to_string(fermiDiracSmearingTemp));

            if (scfMixingParameter != -1.0)
              applyPrm("SCF parameters",
                       "MIXING PARAMETER",
                       std::to_string(scfMixingParameter));

            if (mixingScheme != "Unprovided")
              applyPrm("SCF parameters", "MIXING METHOD", mixingScheme);

            applyPrm("Optimization",
                     "ION FORCE",
                     computeIonForces ? "true" : "false");
            applyPrm("Optimization",
                     "CELL STRESS",
                     computeStress ? "true" : "false");

            if (npkpt != 999999)
              {
                const dftfe::Int totalIrreducibleKpt =
                  mpGrid[0] * mpGrid[1] * mpGrid[2] / 2;
                const dftfe::Int npkptSet =
                  npkpt > 0 ?
                    npkpt :
                    internalWrapper::divisor_closest(totalMPIProcesses,
                                                     totalIrreducibleKpt);
                applyPrm("Parallelization", "NPKPT", std::to_string(npkptSet));
              }

            if (meshSize != -1.0)
              applyPrm("Auto mesh generation parameters",
                       "MESH SIZE AROUND ATOM",
                       std::to_string(meshSize));

            if (verbosity != -1)
              applyPrm("", "VERBOSITY", std::to_string(verbosity));

            if (atomBallRadius != -1.0)
              applyPrm("Auto mesh generation parameters",
                       "ATOM BALL RADIUS",
                       std::to_string(atomBallRadius));

            if (orthogonalizationType != "Unprovided")
              applyPrm("Eigen-solver parameters",
                       "ORTHOGONALIZATION TYPE",
                       orthogonalizationType);

            if (smearedNuclearCharges != -1)
              applyPrm("Boundary conditions",
                       "SMEARED NUCLEAR CHARGES",
                       smearedNuclearCharges ? "true" : "false");

            if (useGroupSymmetry != -1)
              applyPrm("Brillouin zone k point sampling options",
                       "USE GROUP SYMMETRY",
                       useGroupSymmetry ? "true" : "false");

            if (useTimeReversalSymmetry != -1)
              applyPrm("Brillouin zone k point sampling options",
                       "USE TIME REVERSAL SYMMETRY",
                       useTimeReversalSymmetry ? "true" : "false");

            if (mixingHistory != -1)
              applyPrm("SCF parameters",
                       "MIXING HISTORY",
                       std::to_string(mixingHistory));

            if (maxSCFIterations != -1)
              applyPrm("SCF parameters",
                       "MAXIMUM ITERATIONS",
                       std::to_string(maxSCFIterations));

            if (dispersionCorrectionType != -1)
              applyPrm("Dispersion Correction",
                       "DISPERSION CORRECTION TYPE",
                       std::to_string(dispersionCorrectionType));

            if (pseudopotentialCalculation != -1)
              applyPrm("DFT functional parameters",
                       "PSEUDOPOTENTIAL CALCULATION",
                       pseudopotentialCalculation ? "true" : "false");

            if (numBands != 999999)
              applyPrm("Eigen-solver parameters",
                       "NUMBER OF KOHN-SHAM WAVEFUNCTIONS",
                       std::to_string(numBands));

            if (wfcBlockSize != 999999)
              applyPrm("Eigen-solver parameters",
                       "WFC BLOCK SIZE",
                       std::to_string(wfcBlockSize));

            if (chebyWfcBlockSize != 999999)
              applyPrm("Eigen-solver parameters",
                       "CHEBY WFC BLOCK SIZE",
                       std::to_string(chebyWfcBlockSize));

            if (densityQuadratureRule != -1)
              applyPrm("Finite element mesh parameters",
                       "DENSITY QUADRATURE RULE",
                       std::to_string(densityQuadratureRule));

            if (useSinglePrecCheby != -1)
              applyPrm("Eigen-solver parameters",
                       "USE SINGLE PREC CHEBY",
                       useSinglePrecCheby ? "true" : "false");

            if (keepScratch)
              applyPrm("", "KEEP SCRATCH FOLDER", "true");

            // Generic .prm overrides (Mehul): applied AFTER all typed injection
            // so an explicit user override always wins. Format:
            // "section|||key|||value@@@section|||key|||value", empty section =
            // top level. This is the single path used by the calculator's
            // prm_file=/extra_prm= escape hatches and by every "planned" kwarg.
            if (!d_socketPrmOverrides.empty())
              {
                const std::string ov = d_socketPrmOverrides;
                std::size_t       pos = 0;
                while (pos < ov.size())
                  {
                    const std::size_t e = ov.find("@@@", pos);
                    const std::string entry = ov.substr(
                      pos, e == std::string::npos ? std::string::npos : e - pos);
                    pos = (e == std::string::npos) ? ov.size() : e + 3;

                    const std::size_t d1 = entry.find("|||");
                    if (d1 == std::string::npos)
                      continue;
                    const std::size_t d2 = entry.find("|||", d1 + 3);
                    if (d2 == std::string::npos)
                      continue;

                    applyPrm(entry.substr(0, d1),
                             entry.substr(d1 + 3, d2 - d1 - 3),
                             entry.substr(d2 + 3));
                  }
              }

            prm.write(parameter_file_path);
            system("sync"); // Force filesystem flush
          }

        MPI_Barrier(d_mpi_comm_parent);
        sleep(3); // Increased wait for NFS consistency
        d_dftfeParamsPtr                    = new dftfe::dftParameters;
        d_dftfeParamsPtr->parse_parameters(parameter_file_path,
                                           d_mpi_comm_parent,
                                           (verbosity >= 1),
                                           "GS",
                                           ".",
                                           verbosity,
#ifdef DFTFE_WITH_DEVICE
                                           useDevice
#else
                                           false
#endif
                                           );
        d_dftfeParamsPtr->keepScratchFolder = keepScratch;

#ifdef DFTFE_WITH_DEVICE
        d_dftfeParamsPtr->useDevice = useDevice;
#endif
      }
    initialize(setDeviceToMPITaskBindingInternally, useDevice);
  }


  void
  dftfeWrapper::createScratchFolder()
  {
    if (d_mpi_comm_parent != MPI_COMM_NULL)
      {
        if (dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent) == 0)
          {
            d_scratchFolderName =
              "dftfeScratch" +
              std::to_string(
                dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +
              "t" +
              std::to_string(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count());
          }

        dftfe::Int line_size = d_scratchFolderName.size();
        MPI_Bcast(&line_size,
                  1,
                  dftfe::dataTypes::mpi_type_id(&line_size),
                  0,
                  d_mpi_comm_parent);
        if (dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent) != 0)
          d_scratchFolderName.resize(line_size);
        MPI_Bcast(const_cast<char *>(d_scratchFolderName.data()),
                  line_size,
                  MPI_CHAR,
                  0,
                  d_mpi_comm_parent);

        if (dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent) == 0)
          mkdir(d_scratchFolderName.c_str(), ACCESSPERMS);

        MPI_Barrier(d_mpi_comm_parent);
      }
  }

  void
  dftfeWrapper::initialize(const bool setDeviceToMPITaskBindingInternally,
                           const bool useDevice)
  {
    if (d_mpi_comm_parent != MPI_COMM_NULL)
      {
#ifdef DFTFE_WITH_DEVICE
        if (useDevice && setDeviceToMPITaskBindingInternally &&
            !d_isDeviceToMPITaskBindingSetInternally)
          {
            dftfe::utils::deviceKernelsGeneric::setupDevice(
              dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent));
            d_isDeviceToMPITaskBindingSetInternally = true;
          }
#endif

        dftfe::dftUtils::Pool kPointPool(d_mpi_comm_parent,
                                         d_dftfeParamsPtr->npool,
                                         d_dftfeParamsPtr->verbosity);
        dftfe::dftUtils::Pool bandGroupsPool(kPointPool.get_intrapool_comm(),
                                             d_dftfeParamsPtr->nbandGrps,
                                             d_dftfeParamsPtr->verbosity);

        if (d_dftfeParamsPtr->verbosity >= 1)
          {
            dealii::ConditionalOStream pcout(
              std::cout,
              (dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent) ==
               0));
            pcout
              << "=================================MPI Parallelization========================================="
              << std::endl;
            pcout << "Total number of MPI tasks: "
                  << dealii::Utilities::MPI::n_mpi_processes(d_mpi_comm_parent)
                  << std::endl;
            pcout << "k-point parallelization processor groups: "
                  << dealii::Utilities::MPI::n_mpi_processes(
                       kPointPool.get_interpool_comm())
                  << std::endl;
            pcout << "Band parallelization processor groups: "
                  << dealii::Utilities::MPI::n_mpi_processes(
                       bandGroupsPool.get_interpool_comm())
                  << std::endl;
            pcout
              << "Number of MPI tasks for finite-element domain decomposition: "
              << dealii::Utilities::MPI::n_mpi_processes(
                   bandGroupsPool.get_intrapool_comm())
              << std::endl;
            pcout
              << "============================================================================================"
              << std::endl;
          }


        // set stdout precision
        std::cout << std::scientific << std::setprecision(18);

        dftfe::Int order = d_dftfeParamsPtr->finiteElementPolynomialOrder;
        dftfe::Int orderElectro =
          d_dftfeParamsPtr->finiteElementPolynomialOrderElectrostatics;

        if (!useDevice)
          {
            dftfe::internalWrapper::create_dftfe<
              dftfe::utils::MemorySpace::HOST>(
              d_mpi_comm_parent,
              bandGroupsPool.get_intrapool_comm(),
              kPointPool.get_interpool_comm(),
              bandGroupsPool.get_interpool_comm(),
              d_scratchFolderName,
              *d_dftfeParamsPtr,
              &d_dftfeBasePtr);
          }
#ifdef DFTFE_WITH_DEVICE
        else if (useDevice)
          {
            dftfe::internalWrapper::create_dftfe<
              dftfe::utils::MemorySpace::DEVICE>(
              d_mpi_comm_parent,
              bandGroupsPool.get_intrapool_comm(),
              kPointPool.get_interpool_comm(),
              bandGroupsPool.get_interpool_comm(),
              d_scratchFolderName,
              *d_dftfeParamsPtr,
              &d_dftfeBasePtr);
          }
#endif
        d_dftfeBasePtr->set();
        d_dftfeBasePtr->init();
      }
  }

  void
  dftfeWrapper::clear()
  {
    if (d_mpi_comm_parent != MPI_COMM_NULL)
      {
        if (d_dftfeBasePtr != nullptr)
          {
            delete d_dftfeBasePtr;

            if (!d_dftfeParamsPtr->keepScratchFolder &&
                dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent) ==
                  0)
              {
                std::string command = "rm -rf " + d_scratchFolderName;
                system(command.c_str());
              }
            MPI_Barrier(d_mpi_comm_parent);
          }
        if (d_dftfeParamsPtr != nullptr)
          delete d_dftfeParamsPtr;
        MPI_Comm_free(&d_mpi_comm_parent);
      }
    d_dftfeBasePtr    = nullptr;
    d_dftfeParamsPtr  = nullptr;
    d_mpi_comm_parent = MPI_COMM_NULL;
  }

  void
  dftfeWrapper::run()
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    d_dftfeBasePtr->run();
  }

  void
  dftfeWrapper::writeMesh()
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    d_dftfeBasePtr->writeMesh();
  }


  std::tuple<double, bool, double>
  dftfeWrapper::computeDFTFreeEnergy(const bool computeIonForces,
                                     const bool computeCellStress)
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    std::tuple<bool, double> t =
      d_dftfeBasePtr->solve(computeIonForces, computeCellStress);
    return std::make_tuple(d_dftfeBasePtr->getFreeEnergy(),
                           std::get<0>(t),
                           std::get<1>(t));
  }

  double
  dftfeWrapper::getDFTFreeEnergy() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    return d_dftfeBasePtr->getFreeEnergy();
  }


  double
  dftfeWrapper::getElectronicEntropicEnergy() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    return d_dftfeBasePtr->getEntropicEnergy();
  }


  std::vector<std::vector<double>>
  dftfeWrapper::getForcesAtoms() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    std::vector<double> ionicForcesVec = d_dftfeBasePtr->getForceonAtoms();
    
    // Mehul: Safety check if forces were not computed
    if (ionicForcesVec.empty())
      return std::vector<std::vector<double>>();

    std::vector<std::vector<double>> ionicForces(
      ionicForcesVec.size() / 3,
      std::vector<double>(3, 0.0));
    for (dftfe::uInt i = 0; i < ionicForces.size(); ++i)
      for (dftfe::uInt j = 0; j < 3; ++j)
        ionicForces[i][j] = -ionicForcesVec[3 * i + j];
    return ionicForces;
  }

  std::vector<std::vector<double>>
  dftfeWrapper::getCellStress() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    std::vector<std::vector<double>> cellStress(3, std::vector<double>(3, 0.0));
    dealii::Tensor<2, 3, double>     cellStressTensor =
      d_dftfeBasePtr->getCellStress();

    // Returned as computed, sigma[i][j] = (1/Omega) dE/deps_ij, which is what
    // printStress() writes to the output file and what geoOptCell uses as its
    // optimizer gradient. This used to be negated here; the negation belonged to
    // the MDI path, where <STRESS is pressure-positive, and it is applied in
    // MDIEngine::send_stress() instead.
    for (dftfe::uInt i = 0; i < 3; ++i)
      for (dftfe::uInt j = 0; j < 3; ++j)
        cellStress[i][j] = cellStressTensor[i][j];
    return cellStress;
  }

  void
  dftfeWrapper::setSocketPrmOverrides(const std::string &overrides)
  {
    d_socketPrmOverrides = overrides;
  }

  void
  dftfeWrapper::updateAtomPositions(
    const std::vector<std::vector<double>> atomsDisplacements)
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    AssertThrow(
      atomsDisplacements.size() ==
        d_dftfeBasePtr->getAtomLocationsCart().size(),
      dealii::ExcMessage(
        "DFT-FE error: Incorrect size of atomsDisplacements vector."));

    // Minimum image on the displacement, periodic directions only.
    //
    // Native GEOOPT never needed this. geoOptIon.cc:679-701 hands over the
    // optimizer's own solution vector, which is a *step*, and DFT-FE holds
    // exactly one copy of where the atoms are, so there is nothing for that
    // step to disagree with -- geoOptIon.cc:817-821 does not even implement
    // solution(), it throws. Every caller of this function instead keeps a
    // second copy of the positions and reconstructs the step by subtracting:
    // socket_interface.cc:800 and MDIEngine.cpp:619-621 both compute
    // new_coords - getAtomPositionsCart(). DFT-FE's copy is wrapped into the
    // cell (moveAtoms.cc:269-298) while the driver's is deliberately not
    // (calculator.py:519-527 keeps it unwrapped so that a boundary crossing
    // does not look like a lattice jump to the optimizer), so the subtraction
    // picks up a whole lattice vector the moment an atom crosses a cell face.
    // The bug needs two copies of the geometry; native has one.
    //
    // Under PBC a displacement between two configurations is defined only
    // modulo a lattice vector, and the physical one is the shortest. This is
    // not a new convention: it is the rule internal::wrapAtomsAcrossPeriodicBc
    // (moveAtoms.cc:80-119) already applies to a *point*, applied here to a
    // *difference*. In the run that first exposed this, the raw x component was
    // 24.0249 Bohr against a 24.0081 Bohr cell edge; the minimum image is
    // 0.0168 Bohr, which is the step the atom actually took. Left unfolded it
    // exceeded moveAtoms.cc:258's 0.5 Bohr threshold on every ionic step, so
    // DFT-FE rebuilt the vself bins from scratch (createAtomBins) instead of
    // updating their boundary conditions, and eventually died in the rebuild.
    //
    // The fold has to go through fractional coordinates. DFT-FE constrains the
    // domain vectors only to be three in number and right handed
    // (dft.cc:517-546); its own tests carry fully triclinic cells, including
    // testsGPU/pseudopotential/real/domainVectors_ReS2.inp with all three
    // vectors mutually non-orthogonal. Subtracting one lattice vector
    // therefore shifts all three Cartesian components, and a per-Cartesian
    // component fold would be silently wrong for every non-orthogonal cell.
    std::vector<std::vector<double>> disp = atomsDisplacements;

    std::vector<bool> periodicBc(3, false);
    periodicBc[0] = d_dftfeParamsPtr->periodicX;
    periodicBc[1] = d_dftfeParamsPtr->periodicY;
    periodicBc[2] = d_dftfeParamsPtr->periodicZ;

    if (periodicBc[0] || periodicBc[1] || periodicBc[2])
      {
        const std::vector<std::vector<double>> cell = d_dftfeBasePtr->getCell();
        std::vector<double>                    cellVectorsFlattened(9, 0.0);
        for (dftfe::uInt idim = 0; idim < 3; idim++)
          for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
            cellVectorsFlattened[3 * idim + jdim] = cell[idim][jdim];

        const dftfe::uInt numberGlobalAtoms = disp.size();

        // imageVec[3*i+idim] is the whole number of lattice vector idim to be
        // removed from atom i's displacement. It stays zero for every
        // non-periodic direction: there is no image to fold to there, and
        // dft.cc:1102-1109 requires a non-periodic fractional coordinate to
        // lie strictly inside the cell, so folding one would move the atom
        // through vacuum into a different structure. On a mixed-PBC slab,
        // folding a periodic direction can still change the Cartesian
        // component along the non-periodic one when that lattice vector has a
        // projection on it; that is correct -- it is the same atom.
        std::vector<dftfe::Int> imageVec(3 * numberGlobalAtoms, 0);
        std::vector<double>     residualFrac(3 * numberGlobalAtoms, 0.0);

        for (dftfe::uInt i = 0; i < numberGlobalAtoms; ++i)
          {
            // The cell corner cancels identically in a difference of two
            // positions, so the two-argument overload -- which expects a
            // coordinate already taken relative to the corner -- is the right
            // one here, and no corner has to be constructed.
            const std::vector<double> frac =
              dftUtils::getFractionalCoordinates(cellVectorsFlattened, disp[i]);
            for (dftfe::uInt idim = 0; idim < 3; ++idim)
              if (periodicBc[idim])
                {
                  const double image         = std::round(frac[idim]);
                  imageVec[3 * i + idim]     = static_cast<dftfe::Int>(image);
                  residualFrac[3 * i + idim] = frac[idim] - image;
                }
          }

        // Synchronize the way moveAtoms.cc:250-267 and :292 synchronize the
        // wrap decision and the wrapped coordinates: the branch is a
        // std::round() of an LU solve, and two ranks that disagreed by one
        // image would move the mesh differently and then diverge inside the
        // collectives of initNoRemesh(). Broadcasting the integers rather than
        // the folded doubles makes the arithmetic below identical on every
        // rank by construction.
        if (numberGlobalAtoms > 0)
          {
            dftfe::Int imageVecSize = imageVec.size();
            MPI_Bcast(&(imageVec[0]),
                      imageVecSize,
                      dftfe::dataTypes::mpi_type_id(&imageVec[0]),
                      0,
                      d_mpi_comm_parent);
          }

        dealii::ConditionalOStream pcout(
          std::cout,
          (dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent) == 0));

        dftfe::uInt numberAtomsFolded = 0;
        dftfe::Int  maxAbsImage       = 0;
        double      maxAbsResidual    = 0.0;
        double      maxFoldedNorm     = 0.0;

        // Fold whatever whole number of lattice vectors separates the two copies
        // of the geometry, not just one. What arrives here is not the optimizer's
        // step: it is step + n*a, where n is how far the driver's copy has
        // drifted from DFT-FE's. DFT-FE wraps its own copy back into the cell
        // whenever a step is large enough to force it (moveAtoms.cc:246-298; with
        // FLOATING NUCLEAR CHARGES the small-step branch at :301-345 defers the
        // wrap instead), while the external driver deliberately never wraps
        // (calculator.py:519-527, and LAMMPS and i-PI keep unwrapped coordinates
        // by default). So n grows by one every time an atom nets a crossing of
        // the same face, without bound, and a diffusing ion in an MD run reaches
        // n = 2 and beyond as a matter of course. Subtracting whole lattice
        // vectors is an exact symmetry of the periodic system, so the fold is
        // right for any n; refusing n >= 2 would abort a correct trajectory mid-
        // run, which is why this counts rather than asserts.
        for (dftfe::uInt i = 0; i < numberGlobalAtoms; ++i)
          {
            bool folded = false;
            for (dftfe::uInt idim = 0; idim < 3; ++idim)
              {
                const dftfe::Int image = imageVec[3 * i + idim];
                if (image == 0)
                  continue;
                folded = true;

                const dftfe::Int absImage = image < 0 ? -image : image;
                if (absImage > maxAbsImage)
                  maxAbsImage = absImage;
                const double absResidual =
                  std::fabs(residualFrac[3 * i + idim]);
                if (absResidual > maxAbsResidual)
                  maxAbsResidual = absResidual;
              }

            if (!folded)
              continue;

            // Nothing below runs for an atom whose image vector is zero, so an
            // ordinary small step leaves the caller's doubles untouched rather
            // than reconstructed. That matters: getFractionalCoordinates() is
            // an LU solve (dftUtils.h:123-151), and round tripping every
            // displacement through it would perturb the last bits of every
            // step DFT-FE has ever taken through this path, and with them
            // every stored reference energy. Subtracting whole lattice vectors
            // from the original Cartesian components keeps the fold exact in
            // the same sense.
            ++numberAtomsFolded;
            for (dftfe::uInt idim = 0; idim < 3; ++idim)
              {
                const double image =
                  static_cast<double>(imageVec[3 * i + idim]);
                if (image == 0.0)
                  continue;
                for (dftfe::uInt jdim = 0; jdim < 3; ++jdim)
                  disp[i][jdim] -= image * cell[idim][jdim];
              }

            const double foldedNorm =
              std::sqrt(disp[i][0] * disp[i][0] + disp[i][1] * disp[i][1] +
                        disp[i][2] * disp[i][2]);
            if (foldedNorm > maxFoldedNorm)
              maxFoldedNorm = foldedNorm;
          }

        // Report the fold even when it is entirely routine, and report the
        // three numbers that say which kind of fold it was. This bug survived
        // because the wrong displacement was applied in silence for a whole
        // relaxation campaign; a boundary crossing is a normal event, but it
        // has to be visible in the output when it happens, so this sits at
        // verbosity >= 1 alongside moveAtoms.cc:313's coordinate dump rather
        // than a level above it.
        //
        // Nothing here aborts, deliberately. The one input that would justify
        // aborting -- a driver handing over absolute positions where a step
        // was expected -- cannot be recognised from the image count: both
        // positions lie inside the cell, so their difference has every
        // fractional component strictly inside (-1,1) and rounds to an image
        // of 0 or +-1, exactly like a genuine crossing. What separates the two
        // is the shape of the fold, which is what these numbers carry. A
        // boundary crossing on a converging trajectory folds a handful of
        // atoms out of thousands and leaves a rounding residual of order 1e-3
        // (0.0168 Bohr left in a 24.0081 Bohr edge, in the run that first
        // exposed this). A driver sending a geometry folds a large fraction of
        // the atoms at once and leaves residuals spread over the whole
        // interval, and the displacements it applies are a sizeable fraction of
        // the cell rather than a step. Measured on a 2000-atom LLZO cell with
        // random absolute positions on both sides: 57% of atoms fold, 76% of
        // the folded axes leave a residual above 0.25, and the largest applied
        // displacement is 20.5 Bohr, against 1 atom, 5e-4 and 0.017 Bohr for a
        // real crossing on the same cell. Note also that this input cannot be
        // told apart by the image count -- both positions lie inside the cell,
        // so |image| never exceeds 1, which is why an earlier version of this
        // code that aborted on |image| >= 2 could not have caught it and
        // aborted diffusive MD instead.
        //
        // The last number also covers the one case where the fold itself can
        // be wrong. Rounding recovers the true image only while every
        // fractional component of the true step is below 0.5 in magnitude,
        // i.e. while the step is shorter than half the smallest INTERPLANAR
        // SPACING V/|a_j x a_k| -- not half the shortest cell edge, which is
        // larger and can be much larger for a compact primitive cell (the
        // shipped testsGPU/pseudopotential/complex/domainVectorsGaAs.inp has
        // 7.68 Bohr edges but 6.27 Bohr spacings). A step that long is far
        // outside anything the callers produce -- NEB clamps each component to
        // 0.4 Bohr (nudgedElasticBandClass.cc:1181-1184) and external
        // optimizers move by the order of MAXIMUM ION UPDATE STEP, 0.5 Bohr by
        // default -- but if it ever happened the applied displacement printed
        // here is what would show it.
        if (numberAtomsFolded > 0 && d_dftfeParamsPtr->verbosity >= 1)
          pcout << "Minimum image applied to the displacement of "
                << numberAtomsFolded << " of " << numberGlobalAtoms
                << " atom(s) crossing a periodic cell face: largest image "
                << maxAbsImage << " lattice vector(s), largest rounding "
                << "residual " << maxAbsResidual
                << " of a lattice vector, largest applied displacement "
                << maxFoldedNorm
                << " Bohr. A residual approaching 0.5, or a fold touching a "
                << "large fraction of the atoms, means the driver is not "
                << "sending what this function expects: a step from the "
                << "current positions, not a new geometry." << std::endl;
      }

    // Snapshot DFT-FE's copy of the geometry before the move, so that what the
    // driver asked for can be checked against what DFT-FE ended up holding.
    // See the invariant below for why this is worth two extra O(N) passes.
    const std::vector<std::vector<double>> positionsBefore =
      getAtomPositionsCart();

    std::vector<dealii::Tensor<1, 3, double>> dispVec(
      atomsDisplacements.size());
    for (dftfe::uInt i = 0; i < dispVec.size(); ++i)
      for (dftfe::uInt j = 0; j < 3; ++j)
        dispVec[i][j] = disp[i][j];
    d_dftfeBasePtr->updateAtomPositionsAndMoveMesh(dispVec);

    // The invariant that closes the failure CLASS, not just the fold above.
    //
    // Every caller of this function keeps its own copy of the geometry and
    // reconstructs a step by subtracting DFT-FE's copy from it
    // (socket_interface.cc:800, MDIEngine.cpp:619-621). Nothing anywhere ever
    // checked that the two copies still agree afterwards. That is precisely how
    // the 24 Bohr displacement of job 8761021 survived: it was applied in
    // silence for a whole relaxation campaign, every energy and force landing on
    // a geometry the driver had not asked for, and it surfaced only as a SIGSEGV
    // three ionic steps later on 32 nodes. A fold that repairs one such
    // disagreement leaves the next one just as invisible, so state the invariant
    // and enforce it:
    //
    //   after  ==  before + (what the driver asked for),  modulo whole lattice
    //   vectors along periodic directions, and exactly along open ones.
    //
    // Note this is checked against atomsDisplacements -- the caller's RAW
    // request -- not against the folded disp, which would make it a tautology.
    // The fold is allowed to change the answer by whole lattice vectors and by
    // nothing else, which is exactly what "modulo" states.
    //
    // It holds by construction of moveAtoms.cc, and that is what makes it a
    // useful check rather than a guess about DFT-FE's behaviour: positions are
    // only ever advanced by atomLocations += disp (:308-310, :383-385) or by
    // wrapAtomsAcrossPeriodicBc(coor + disp) (:277-296). No branch clamps,
    // symmetrizes, projects or rescales a displacement, so any violation is a
    // genuine defect -- in the fold, in DFT-FE's own wrap, in the mesh move, or
    // in a driver whose frame drifted in a way this code does not model.
    //
    // Deliberately wrap-agnostic: with FLOATING NUCLEAR CHARGES a small step
    // takes moveAtoms.cc:301-345, which defers the wrap, while a large one takes
    // :269-298 and wraps immediately. Both satisfy the invariant -- the
    // unwrapped case with an image of exactly zero -- so the check does not have
    // to know which branch ran, and cannot go stale if that choice changes.
    //
    // This aborts rather than warning. A violation means every energy and force
    // from here on describes a geometry nobody asked for; continuing would
    // produce numbers that look fine and are not, which is the one outcome this
    // project has already paid for once.
    {
      const std::vector<std::vector<double>> positionsAfter =
        getAtomPositionsCart();
      const std::vector<std::vector<double>> cellCheck = d_dftfeBasePtr->getCell();
      std::vector<double>                    cellCheckFlattened(9, 0.0);
      for (dftfe::uInt idim = 0; idim < 3; idim++)
        for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
          cellCheckFlattened[3 * idim + jdim] = cellCheck[idim][jdim];

      // In fractional units, so the same number means the same thing on every
      // cell and every axis. 1e-8 of a lattice vector is ~2e-7 Bohr on the cells
      // this runs on -- six orders above the ~1e-13 Bohr an LU round trip and a
      // wrap cost, and six orders below the 0.02 of a lattice vector that the
      // smallest defect of this class would produce (moveAtoms.cc:258's 0.5 Bohr
      // remesh threshold against a ~24 Bohr edge). A whole missed lattice vector
      // is 1.0, eight orders clear.
      const double fracTol        = 1e-8;
      double       worstResidual  = 0.0;
      double       worstDeviation = 0.0;
      dftfe::uInt  worstAtom      = 0;
      dftfe::uInt  worstDim       = 0;

      for (dftfe::uInt i = 0; i < positionsAfter.size(); ++i)
        {
          std::vector<double> deviation(3, 0.0);
          for (dftfe::uInt j = 0; j < 3; ++j)
            deviation[j] = positionsAfter[i][j] - positionsBefore[i][j] -
                           atomsDisplacements[i][j];

          const std::vector<double> frac =
            dftUtils::getFractionalCoordinates(cellCheckFlattened, deviation);
          for (dftfe::uInt idim = 0; idim < 3; ++idim)
            {
              // Periodic: the deviation must be a whole lattice vector. Open:
              // there is no image to fold to, so it must be nothing at all.
              const double residual =
                periodicBc[idim] ?
                  std::fabs(frac[idim] - std::round(frac[idim])) :
                  std::fabs(frac[idim]);
              if (residual > worstResidual)
                {
                  worstResidual = residual;
                  worstAtom     = i;
                  worstDim      = idim;
                  worstDeviation =
                    std::sqrt(deviation[0] * deviation[0] +
                              deviation[1] * deviation[1] +
                              deviation[2] * deviation[2]);
                }
            }
        }

      if (worstResidual >= fracTol)
        {
          std::ostringstream message;
          message
            << "DFT-FE error: the atom positions DFT-FE now holds do not match "
            << "what the driver asked for. Atom " << worstAtom << ", direction "
            << worstDim << " is off by " << worstResidual
            << " of a lattice vector (tolerance " << fracTol
            << "), with a total deviation of " << worstDeviation
            << " Bohr from the requested position. The requested displacement "
            << "must be reproduced exactly along open directions and up to whole "
            << "lattice vectors along periodic ones; it was not, so every energy "
            << "and force computed from here would describe a geometry that was "
            << "never requested.";
          AssertThrow(false, dealii::ExcMessage(message.str()));
        }
    }
  }

  void
  dftfeWrapper::deformCell(
    const std::vector<std::vector<double>> deformationGradient)
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    dealii::Tensor<2, 3, double> defGradTensor;
    for (dftfe::uInt i = 0; i < 3; ++i)
      for (dftfe::uInt j = 0; j < 3; ++j)
        defGradTensor[i][j] = deformationGradient[i][j];
    d_dftfeBasePtr->deformDomain(defGradTensor);
  }

  std::vector<std::vector<double>>
  dftfeWrapper::getAtomPositionsCart() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    // dftfe stores cell centered coordinates
    std::vector<std::vector<double>> temp =
      d_dftfeBasePtr->getAtomLocationsCart();
    std::vector<std::vector<double>> atomLocationsCart(
      d_dftfeBasePtr->getAtomLocationsCart().size(),
      std::vector<double>(3, 0.0));

    std::vector<std::vector<double>> cell = d_dftfeBasePtr->getCell();
    std::vector<double>              shift(3, 0.0);
    for (dftfe::uInt idim = 0; idim < 3; idim++)
      {
        shift[idim] = 0;
        for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
          shift[idim] += cell[jdim][idim] / 2.0;
      }

    for (dftfe::uInt i = 0; i < atomLocationsCart.size(); ++i)
      for (dftfe::uInt j = 0; j < 3; ++j)
        atomLocationsCart[i][j] = temp[i][j + 2] + shift[j];
    return atomLocationsCart;
  }

  std::vector<std::vector<double>>
  dftfeWrapper::getAtomPositionsFrac() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    std::vector<std::vector<double>> temp =
      d_dftfeBasePtr->getAtomLocationsFrac();
    std::vector<std::vector<double>> atomLocationsFrac(
      d_dftfeBasePtr->getAtomLocationsFrac().size(),
      std::vector<double>(3, 0.0));
    for (dftfe::uInt i = 0; i < atomLocationsFrac.size(); ++i)
      for (dftfe::uInt j = 0; j < 3; ++j)
        atomLocationsFrac[i][j] = temp[i][j + 2];
    return atomLocationsFrac;
  }

  std::vector<std::vector<double>>
  dftfeWrapper::getCell() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    return d_dftfeBasePtr->getCell();
  }

  std::vector<bool>
  dftfeWrapper::getPBC() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    std::vector<bool> pbc(3, false);
    pbc[0] = d_dftfeParamsPtr->periodicX;
    pbc[1] = d_dftfeParamsPtr->periodicY;
    pbc[2] = d_dftfeParamsPtr->periodicZ;
    return pbc;
  }

  std::vector<dftfe::Int>
  dftfeWrapper::getAtomicNumbers() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    std::vector<std::vector<double>> temp =
      d_dftfeBasePtr->getAtomLocationsCart();
    std::vector<dftfe::Int> atomicNumbers(
      d_dftfeBasePtr->getAtomLocationsCart().size(), 0);
    for (dftfe::uInt i = 0; i < atomicNumbers.size(); ++i)
      atomicNumbers[i] = temp[i][0];
    return atomicNumbers;
  }


  std::vector<dftfe::Int>
  dftfeWrapper::getValenceElectronNumbers() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    std::vector<std::vector<double>> temp =
      d_dftfeBasePtr->getAtomLocationsCart();
    std::vector<dftfe::Int> valenceNumbers(
      d_dftfeBasePtr->getAtomLocationsCart().size(), 0);
    for (dftfe::uInt i = 0; i < valenceNumbers.size(); ++i)
      valenceNumbers[i] = temp[i][1];
    return valenceNumbers;
  }

  dftBase *
  dftfeWrapper::getDftfeBasePtr()
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    return d_dftfeBasePtr;
  }


  dftParameters *
  dftfeWrapper::getDftfeParamsPtr()
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    return d_dftfeParamsPtr;
  }



  void
  dftfeWrapper::writeDomainAndAtomCoordinates(const std::string Path) const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    d_dftfeBasePtr->writeDomainAndAtomCoordinates(Path);
  }
} // namespace dftfe
