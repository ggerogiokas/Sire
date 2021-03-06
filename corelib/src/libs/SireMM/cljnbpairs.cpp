/********************************************\
  *
  *  Sire - Molecular Simulation Framework
  *
  *  Copyright (C) 2007  Christopher Woods
  *
  *  This program is free software; you can redistribute it and/or modify
  *  it under the terms of the GNU General Public License as published by
  *  the Free Software Foundation; either version 2 of the License, or
  *  (at your option) any later version.
  *
  *  This program is distributed in the hope that it will be useful,
  *  but WITHOUT ANY WARRANTY; without even the implied warranty of
  *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  *  GNU General Public License for more details.
  *
  *  You should have received a copy of the GNU General Public License
  *  along with this program; if not, write to the Free Software
  *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
  *
  *  For full details of the license please see the COPYING file
  *  that should have come with this distribution.
  *
  *  You can contact the authors via the developer's mailing list
  *  at http://siremol.org
  *
\*********************************************/

#include "cljnbpairs.h"

#include "SireStream/datastream.h"

using namespace SireMM;
using namespace SireMol;
using namespace SireBase;
using namespace SireStream;

static const RegisterMetaType<CoulombNBPairs> r_coulnbpairs;
static const RegisterMetaType<LJNBPairs> r_ljnbpairs;
static const RegisterMetaType<CLJNBPairs> r_cljnbpairs;

////////
//////// Fully instantiate the template class
////////

namespace SireMM
{
    template class AtomPairs<CoulombScaleFactor>;
    template class CGAtomPairs<CoulombScaleFactor>;

    template class AtomPairs<LJScaleFactor>;
    template class CGAtomPairs<LJScaleFactor>;

    template class AtomPairs<CLJScaleFactor>;
    template class CGAtomPairs<CLJScaleFactor>;
}

////////
//////// Implementation of CoulombScaleFactor
////////

/** Serialise to a binary datastream */
QDataStream SIREMM_EXPORT &operator<<(QDataStream &ds, const CoulombScaleFactor &sclfac)
{
    ds << sclfac.cscl;
       
    return ds;
}

/** Extract from a binary datastream */
QDataStream SIREMM_EXPORT &operator>>(QDataStream &ds, CoulombScaleFactor &sclfac)
{
    ds >> sclfac.cscl;
       
    return ds;
}

/** Construct with the Coulomb scale factor equal to 'scl' */
CoulombScaleFactor::CoulombScaleFactor(double scl) : cscl(scl)
{}

/** Copy constructor */
CoulombScaleFactor::CoulombScaleFactor(const CoulombScaleFactor &other)
                   : cscl(other.cscl)
{}

/** Destructor */
CoulombScaleFactor::~CoulombScaleFactor()
{}

/** Copy assignment operator */
CoulombScaleFactor& CoulombScaleFactor::operator=(const CoulombScaleFactor &other)
{
    cscl = other.cscl;
    
    return *this;
}

/** Comparison operator */
bool CoulombScaleFactor::operator==(const CoulombScaleFactor &other) const
{
    return cscl == other.cscl;
}

/** Comparison operator */
bool CoulombScaleFactor::operator!=(const CoulombScaleFactor &other) const
{
    return cscl != other.cscl;
}

/** Return the Coulomb parameter scaling factor */
double CoulombScaleFactor::coulomb() const
{
    return cscl;
}

////////
//////// Implementation of LJScaleFactor
////////

/** Serialise to a binary datastream */
QDataStream SIREMM_EXPORT &operator<<(QDataStream &ds, const LJScaleFactor &sclfac)
{
    ds << sclfac.ljscl;
       
    return ds;
}

/** Extract from a binary datastream */
QDataStream SIREMM_EXPORT &operator>>(QDataStream &ds, LJScaleFactor &sclfac)
{
    ds >> sclfac.ljscl;
       
    return ds;
}

/** Construct with the LJ scale factor equal to 'scl' */
LJScaleFactor::LJScaleFactor(double scl) : ljscl(scl)
{}

/** Copy constructor */
LJScaleFactor::LJScaleFactor(const LJScaleFactor &other)
              : ljscl(other.ljscl)
{}

/** Destructor */
LJScaleFactor::~LJScaleFactor()
{}

/** Copy assignment operator */
LJScaleFactor& LJScaleFactor::operator=(const LJScaleFactor &other)
{
    ljscl = other.ljscl;
    
    return *this;
}

/** Comparison operator */
bool LJScaleFactor::operator==(const LJScaleFactor &other) const
{
    return ljscl == other.ljscl;
}

/** Comparison operator */
bool LJScaleFactor::operator!=(const LJScaleFactor &other) const
{
    return ljscl != other.ljscl;
}

/** Return the LJ parameter scaling factor */
double LJScaleFactor::lj() const
{
    return ljscl;
}

////////
//////// Implementation of CLJScaleFactor
////////

/** Serialise to a binary datastream */
QDataStream SIREMM_EXPORT &operator<<(QDataStream &ds, const CLJScaleFactor &sclfac)
{
    ds << static_cast<const CoulombScaleFactor&>(sclfac) 
       << static_cast<const LJScaleFactor&>(sclfac);
       
    return ds;
}

/** Extract from a binary datastream */
QDataStream SIREMM_EXPORT &operator>>(QDataStream &ds, CLJScaleFactor &sclfac)
{
    ds >> static_cast<CoulombScaleFactor&>(sclfac) 
       >> static_cast<LJScaleFactor&>(sclfac);
       
    return ds;
}

/** Construct with both the Coulomb and LJ scale factors equal to 'scl' */
CLJScaleFactor::CLJScaleFactor(double scl)
               : CoulombScaleFactor(scl), LJScaleFactor(scl)
{}

/** Construct with 'scale_coul' Coulomb scaling, and 'scale_lj'
    LJ scaling. */
CLJScaleFactor::CLJScaleFactor(double scale_coul, double scale_lj)
               : CoulombScaleFactor(scale_coul),
                 LJScaleFactor(scale_lj)
{}

/** Copy constructor */
CLJScaleFactor::CLJScaleFactor(const CLJScaleFactor &other)
               : CoulombScaleFactor(other), LJScaleFactor(other)
{}

/** Destructor */
CLJScaleFactor::~CLJScaleFactor()
{}

/** Copy assignment operator */
CLJScaleFactor& CLJScaleFactor::operator=(const CLJScaleFactor &other)
{
    CoulombScaleFactor::operator=(other);
    LJScaleFactor::operator=(other);
    
    return *this;
}

/** Comparison operator */
bool CLJScaleFactor::operator==(const CLJScaleFactor &other) const
{
    return CoulombScaleFactor::operator==(other) and
           LJScaleFactor::operator==(other);
}

/** Comparison operator */
bool CLJScaleFactor::operator!=(const CLJScaleFactor &other) const
{
    return CoulombScaleFactor::operator!=(other) or
           LJScaleFactor::operator!=(other);
}

////////
//////// Implementation of CoulombNBPairs
////////

/** Serialise to a binary datastream */
QDataStream SIREMM_EXPORT &operator<<(QDataStream &ds, const CoulombNBPairs &coulnbpairs)
{
    writeHeader(ds, r_coulnbpairs, 1)
        << static_cast<const AtomPairs<CoulombScaleFactor>&>(coulnbpairs);

    return ds;
}

/** Deserialise from a binary datastream */
QDataStream SIREMM_EXPORT &operator>>(QDataStream &ds, CoulombNBPairs &coulnbpairs)
{
    VersionID v = readHeader(ds, r_coulnbpairs);

    if (v == 1)
    {
        ds >> static_cast<AtomPairs<CoulombScaleFactor>&>(coulnbpairs);
    }
    else
        throw version_error(v, "1", r_coulnbpairs, CODELOC);

    return ds;
}

/** Null constructor */
CoulombNBPairs::CoulombNBPairs() 
               : ConcreteProperty<CoulombNBPairs,
                    AtomPairs<CoulombScaleFactor> >( CoulombScaleFactor(1) )
{}

/** Construct, using 'default_scale' for all of the atom-atom
    interactions in the molecule 'molinfo' */
CoulombNBPairs::CoulombNBPairs(const MoleculeInfoData &molinfo, 
                               const CoulombScaleFactor &default_scale)
               : ConcreteProperty<CoulombNBPairs,
                    AtomPairs<CoulombScaleFactor> >(molinfo, default_scale)
{}

/** Construct for the molecule viewed in 'molview' */
CoulombNBPairs::CoulombNBPairs(const MoleculeView &molview,
                               const CoulombScaleFactor &default_scale)
               : ConcreteProperty<CoulombNBPairs,
                    AtomPairs<CoulombScaleFactor> >(molview, default_scale)
{}

/** Construct from the coulomb scaling factors in 'cljpairs' */
CoulombNBPairs::CoulombNBPairs(const CLJNBPairs &cljpairs)
               : ConcreteProperty<CoulombNBPairs,
                    AtomPairs<CoulombScaleFactor> >(
                        static_cast<const AtomPairs<CLJScaleFactor>&>(cljpairs) )
{}

/** Copy constructor */
CoulombNBPairs::CoulombNBPairs(const CoulombNBPairs &other)
               : ConcreteProperty< CoulombNBPairs,
                    AtomPairs<CoulombScaleFactor> >(other)
{}

/** Destructor */
CoulombNBPairs::~CoulombNBPairs()
{}

/** Copy assignment operator */
CoulombNBPairs& CoulombNBPairs::operator=(const CoulombNBPairs &other)
{
    AtomPairs<CoulombScaleFactor>::operator=(other);
    return *this;
}

/** Copy from a CLJNBPairs object */
CoulombNBPairs& CoulombNBPairs::operator=(const CLJNBPairs &cljpairs)
{
    return this->operator=( CoulombNBPairs(cljpairs) );
}

/** Comparison operator */
bool CoulombNBPairs::operator==(const CoulombNBPairs &other) const
{
    return AtomPairs<CoulombScaleFactor>::operator==(other);
}

/** Comparison operator */
bool CoulombNBPairs::operator!=(const CoulombNBPairs &other) const
{
    return AtomPairs<CoulombScaleFactor>::operator!=(other);
}

////////
//////// Implementation of LJNBPairs
////////

/** Serialise to a binary datastream */
QDataStream SIREMM_EXPORT &operator<<(QDataStream &ds, const LJNBPairs &ljnbpairs)
{
    writeHeader(ds, r_ljnbpairs, 1)
        << static_cast<const AtomPairs<LJScaleFactor>&>(ljnbpairs);

    return ds;
}

/** Deserialise from a binary datastream */
QDataStream SIREMM_EXPORT &operator>>(QDataStream &ds, LJNBPairs &ljnbpairs)
{
    VersionID v = readHeader(ds, r_ljnbpairs);

    if (v == 1)
    {
        ds >> static_cast<AtomPairs<LJScaleFactor>&>(ljnbpairs);
    }
    else
        throw version_error(v, "1", r_ljnbpairs, CODELOC);

    return ds;
}

/** Null constructor */
LJNBPairs::LJNBPairs() : ConcreteProperty<LJNBPairs,
                              AtomPairs<LJScaleFactor> >( LJScaleFactor(1) )
{}

/** Construct, using 'default_scale' for all of the atom-atom
    interactions in the molecule 'molinfo' */
LJNBPairs::LJNBPairs(const MoleculeInfoData &molinfo, 
                     const LJScaleFactor &default_scale)
           : ConcreteProperty<LJNBPairs,
                   AtomPairs<LJScaleFactor> >(molinfo, default_scale)
{}

/** Construct for the molecule viewed in 'molview' */
LJNBPairs::LJNBPairs(const MoleculeView &molview,
                     const LJScaleFactor &default_scale)
          : ConcreteProperty<LJNBPairs,
                    AtomPairs<LJScaleFactor> >(molview, default_scale)
{}

/** Construct from the LJ scaling factors in 'cljpairs' */
LJNBPairs::LJNBPairs(const CLJNBPairs &cljpairs)
          : ConcreteProperty<LJNBPairs,
                 AtomPairs<LJScaleFactor> >(
                     static_cast<const AtomPairs<CLJScaleFactor>&>(cljpairs) )
{}

/** Copy constructor */
LJNBPairs::LJNBPairs(const LJNBPairs &other)
          : ConcreteProperty< LJNBPairs, AtomPairs<LJScaleFactor> >(other)
{}

/** Destructor */
LJNBPairs::~LJNBPairs()
{}

/** Copy assignment operator */
LJNBPairs& LJNBPairs::operator=(const LJNBPairs &other)
{
    AtomPairs<LJScaleFactor>::operator=(other);
    return *this;
}

/** Copy from a LJNBPairs object */
LJNBPairs& LJNBPairs::operator=(const CLJNBPairs &cljpairs)
{
    return this->operator=( LJNBPairs(cljpairs) );
}

/** Comparison operator */
bool LJNBPairs::operator==(const LJNBPairs &other) const
{
    return AtomPairs<LJScaleFactor>::operator==(other);
}

/** Comparison operator */
bool LJNBPairs::operator!=(const LJNBPairs &other) const
{
    return AtomPairs<LJScaleFactor>::operator!=(other);
}

////////
//////// Implementation of CLJNBPairs
////////

/** Serialise to a binary datastream */
QDataStream SIREMM_EXPORT &operator<<(QDataStream &ds, const CLJNBPairs &cljnbpairs)
{
    writeHeader(ds, r_cljnbpairs, 1)
        << static_cast<const AtomPairs<CLJScaleFactor>&>(cljnbpairs);

    return ds;
}

/** Deserialise from a binary datastream */
QDataStream SIREMM_EXPORT &operator>>(QDataStream &ds, CLJNBPairs &cljnbpairs)
{
    VersionID v = readHeader(ds, r_cljnbpairs);

    if (v == 1)
    {
        ds >> static_cast<AtomPairs<CLJScaleFactor>&>(cljnbpairs);
    }
    else
        throw version_error(v, "1", r_cljnbpairs, CODELOC);

    return ds;
}

/** Null constructor */
CLJNBPairs::CLJNBPairs() : ConcreteProperty<CLJNBPairs,
                               AtomPairs<CLJScaleFactor> >( CLJScaleFactor(1,1) )
{}

/** Construct, using 'default_scale' for all of the atom-atom
    interactions in the molecule 'molinfo' */
CLJNBPairs::CLJNBPairs(const MoleculeInfoData &molinfo, 
                       const CLJScaleFactor &default_scale)
           : ConcreteProperty<CLJNBPairs,
                   AtomPairs<CLJScaleFactor> >(molinfo, default_scale)
{}

/** Construct for the molecule viewed in 'molview' */
CLJNBPairs::CLJNBPairs(const MoleculeView &molview,
                       const CLJScaleFactor &default_scale)
           : ConcreteProperty<CLJNBPairs,
                    AtomPairs<CLJScaleFactor> >(molview, default_scale)
{}

/** Copy constructor */
CLJNBPairs::CLJNBPairs(const CLJNBPairs &other)
           : ConcreteProperty< CLJNBPairs, AtomPairs<CLJScaleFactor> >(other)
{}

/** Destructor */
CLJNBPairs::~CLJNBPairs()
{}

/** Copy assignment operator */
CLJNBPairs& CLJNBPairs::operator=(const CLJNBPairs &other)
{
    AtomPairs<CLJScaleFactor>::operator=(other);
    return *this;
}

/** Comparison operator */
bool CLJNBPairs::operator==(const CLJNBPairs &other) const
{
    return AtomPairs<CLJScaleFactor>::operator==(other);
}

/** Comparison operator */
bool CLJNBPairs::operator!=(const CLJNBPairs &other) const
{
    return AtomPairs<CLJScaleFactor>::operator!=(other);
}

const char* CoulombScaleFactor::typeName()
{
    return QMetaType::typeName( qMetaTypeId<CoulombScaleFactor>() );
}

const char* LJScaleFactor::typeName()
{
    return QMetaType::typeName( qMetaTypeId<LJScaleFactor>() );
}

const char* CLJScaleFactor::typeName()
{
    return QMetaType::typeName( qMetaTypeId<CLJScaleFactor>() );
}

const char* CLJNBPairs::typeName()
{
    return QMetaType::typeName( qMetaTypeId<CLJNBPairs>() );
}

const char* CoulombNBPairs::typeName()
{
    return QMetaType::typeName( qMetaTypeId<CoulombNBPairs>() );
}

const char* LJNBPairs::typeName()
{
    return QMetaType::typeName( qMetaTypeId<LJNBPairs>() );
}
