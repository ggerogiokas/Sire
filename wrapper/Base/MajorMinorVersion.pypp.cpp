// This file has been generated by Py++.

// (C) Christopher Woods, GPL >= 2 License

#include "boost/python.hpp"
#include "Helpers/clone_const_reference.hpp"
#include "MajorMinorVersion.pypp.hpp"

namespace bp = boost::python;

#include "SireStream/datastream.h"

#include "majorminorversion.h"

#include "majorminorversion.h"

SireBase::MajorMinorVersion __copy__(const SireBase::MajorMinorVersion &other){ return SireBase::MajorMinorVersion(other); }

const char* pvt_get_name(const SireBase::MajorMinorVersion&){ return "SireBase::MajorMinorVersion";}

void register_MajorMinorVersion_class(){

    { //::SireBase::MajorMinorVersion
        typedef bp::class_< SireBase::MajorMinorVersion > MajorMinorVersion_exposer_t;
        MajorMinorVersion_exposer_t MajorMinorVersion_exposer = MajorMinorVersion_exposer_t( "MajorMinorVersion", "This is a class that provides a version numbering scheme that\nis guaranteed to provide unique version numbers, even for\ndifferent copies of this object\n\nAuthor: Christopher Woods\n", bp::init< >("Null constructor") );
        bp::scope MajorMinorVersion_scope( MajorMinorVersion_exposer );
        MajorMinorVersion_exposer.def( bp::init< boost::shared_ptr< SireBase::detail::MajorMinorVersionData > const & >(( bp::arg("other") ), "Construct from a raw data object - this should only be called by\nthe registry function") );
        MajorMinorVersion_exposer.def( bp::init< quint64, quint64 >(( bp::arg("vmaj"), bp::arg("vmin") ), "Construct the object for a specific version") );
        MajorMinorVersion_exposer.def( bp::init< SireBase::MajorMinorVersion const & >(( bp::arg("other") ), "Copy constructor") );
        { //::SireBase::MajorMinorVersion::incrementMajor
        
            typedef void ( ::SireBase::MajorMinorVersion::*incrementMajor_function_type)(  ) ;
            incrementMajor_function_type incrementMajor_function_value( &::SireBase::MajorMinorVersion::incrementMajor );
            
            MajorMinorVersion_exposer.def( 
                "incrementMajor"
                , incrementMajor_function_value
                , "Increment the major version number - this resets the\nminor version number to 0" );
        
        }
        { //::SireBase::MajorMinorVersion::incrementMinor
        
            typedef void ( ::SireBase::MajorMinorVersion::*incrementMinor_function_type)(  ) ;
            incrementMinor_function_type incrementMinor_function_value( &::SireBase::MajorMinorVersion::incrementMinor );
            
            MajorMinorVersion_exposer.def( 
                "incrementMinor"
                , incrementMinor_function_value
                , "Increment the minor version number" );
        
        }
        { //::SireBase::MajorMinorVersion::majorVersion
        
            typedef ::quint64 ( ::SireBase::MajorMinorVersion::*majorVersion_function_type)(  ) const;
            majorVersion_function_type majorVersion_function_value( &::SireBase::MajorMinorVersion::majorVersion );
            
            MajorMinorVersion_exposer.def( 
                "majorVersion"
                , majorVersion_function_value
                , "" );
        
        }
        { //::SireBase::MajorMinorVersion::minorVersion
        
            typedef ::quint64 ( ::SireBase::MajorMinorVersion::*minorVersion_function_type)(  ) const;
            minorVersion_function_type minorVersion_function_value( &::SireBase::MajorMinorVersion::minorVersion );
            
            MajorMinorVersion_exposer.def( 
                "minorVersion"
                , minorVersion_function_value
                , "" );
        
        }
        MajorMinorVersion_exposer.def( bp::self != bp::self );
        { //::SireBase::MajorMinorVersion::operator=
        
            typedef ::SireBase::MajorMinorVersion & ( ::SireBase::MajorMinorVersion::*assign_function_type)( ::SireBase::MajorMinorVersion const & ) ;
            assign_function_type assign_function_value( &::SireBase::MajorMinorVersion::operator= );
            
            MajorMinorVersion_exposer.def( 
                "assign"
                , assign_function_value
                , ( bp::arg("other") )
                , bp::return_self< >()
                , "" );
        
        }
        MajorMinorVersion_exposer.def( bp::self == bp::self );
        { //::SireBase::MajorMinorVersion::typeName
        
            typedef char const * ( *typeName_function_type )(  );
            typeName_function_type typeName_function_value( &::SireBase::MajorMinorVersion::typeName );
            
            MajorMinorVersion_exposer.def( 
                "typeName"
                , typeName_function_value
                , "" );
        
        }
        { //::SireBase::MajorMinorVersion::version
        
            typedef ::SireBase::Version const & ( ::SireBase::MajorMinorVersion::*version_function_type)(  ) const;
            version_function_type version_function_value( &::SireBase::MajorMinorVersion::version );
            
            MajorMinorVersion_exposer.def( 
                "version"
                , version_function_value
                , bp::return_value_policy<bp::clone_const_reference>()
                , "" );
        
        }
        { //::SireBase::MajorMinorVersion::what
        
            typedef char const * ( ::SireBase::MajorMinorVersion::*what_function_type)(  ) const;
            what_function_type what_function_value( &::SireBase::MajorMinorVersion::what );
            
            MajorMinorVersion_exposer.def( 
                "what"
                , what_function_value
                , "" );
        
        }
        MajorMinorVersion_exposer.staticmethod( "typeName" );
        MajorMinorVersion_exposer.def( "__copy__", &__copy__);
        MajorMinorVersion_exposer.def( "__deepcopy__", &__copy__);
        MajorMinorVersion_exposer.def( "clone", &__copy__);
        MajorMinorVersion_exposer.def( "__str__", &pvt_get_name);
        MajorMinorVersion_exposer.def( "__repr__", &pvt_get_name);
    }

}
