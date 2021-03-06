// This file has been generated by Py++.

// (C) Christopher Woods, GPL >= 2 License


#include "boost/python.hpp"

#include "Amber.pypp.hpp"

#include "Cube.pypp.hpp"

#include "FlexibilityLibrary.pypp.hpp"

#include "FlexibilityTemplate.pypp.hpp"

#include "IOBase.pypp.hpp"

#include "IOParametersBase.pypp.hpp"

#include "NullIO.pypp.hpp"

#include "PDB.pypp.hpp"

#include "PDBParameters.pypp.hpp"

#include "PerturbationsLibrary.pypp.hpp"

#include "PerturbationsTemplate.pypp.hpp"

#include "ProtoMS.pypp.hpp"

#include "ProtoMSParameters.pypp.hpp"

#include "Tinker.pypp.hpp"

#include "TinkerParameters.pypp.hpp"

#include "TrajectoryMonitor.pypp.hpp"

#include "ZmatrixMaker.pypp.hpp"

namespace bp = boost::python;

#include "SireIO_registrars.h"

#include "SireIO_properties.h"

BOOST_PYTHON_MODULE(_IO){
    register_SireIO_objects();

    register_Amber_class();

    register_Cube_class();

    register_FlexibilityLibrary_class();

    register_FlexibilityTemplate_class();

    register_IOBase_class();

    register_IOParametersBase_class();

    register_NullIO_class();

    register_PDB_class();

    register_PDBParameters_class();

    register_PerturbationsLibrary_class();

    register_PerturbationsTemplate_class();

    register_ProtoMS_class();

    register_ProtoMSParameters_class();

    register_Tinker_class();

    register_TinkerParameters_class();

    register_TrajectoryMonitor_class();

    register_SireIO_properties();

    register_ZmatrixMaker_class();
}

