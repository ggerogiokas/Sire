//WARNING - AUTOGENERATED FILE - CONTENTS WILL BE OVERWRITTEN!
#include <Python.h>

#include "Squire_registrars.h"

#include "qmpotential.h"
#include "pointcharge.h"
#include "sgto.h"
#include "qmchargecalculator.h"
#include "am1bcc.h"
#include "molpro.h"
#include "mopac.h"
#include "qmmmff.h"
#include "qmff.h"
#include "pgto.h"
#include "qmchargeconstraint.h"
#include "sqm.h"
#include "pointdipole.h"
#include "qmprogram.h"

#include "Helpers/objectregistry.hpp"

void register_Squire_objects()
{

    ObjectRegistry::registerConverterFor< Squire::QMComponent >();
    ObjectRegistry::registerConverterFor< Squire::PointCharge >();
    ObjectRegistry::registerConverterFor< Squire::S_GTO >();
    ObjectRegistry::registerConverterFor< Squire::SS_GTO >();
    ObjectRegistry::registerConverterFor< Squire::NullQMChargeCalculator >();
    ObjectRegistry::registerConverterFor< Squire::AM1BCC >();
    ObjectRegistry::registerConverterFor< Squire::Molpro >();
    ObjectRegistry::registerConverterFor< Squire::Mopac >();
    ObjectRegistry::registerConverterFor< Squire::QMMMFF >();
    ObjectRegistry::registerConverterFor< Squire::QMFF >();
    ObjectRegistry::registerConverterFor< Squire::P_GTO >();
    ObjectRegistry::registerConverterFor< Squire::PS_GTO >();
    ObjectRegistry::registerConverterFor< Squire::PP_GTO >();
    ObjectRegistry::registerConverterFor< Squire::QMChargeConstraint >();
    ObjectRegistry::registerConverterFor< Squire::SQM >();
    ObjectRegistry::registerConverterFor< Squire::PointDipole >();
    ObjectRegistry::registerConverterFor< Squire::NullQM >();

}

