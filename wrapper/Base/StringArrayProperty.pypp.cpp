// This file has been generated by Py++.

// (C) Christopher Woods, GPL >= 2 License

#include "boost/python.hpp"
#include "StringArrayProperty.pypp.hpp"

namespace bp = boost::python;

#include "SireError/errors.h"

#include "SireStream/datastream.h"

#include "SireStream/shareddatastream.h"

#include "arrayproperty.hpp"

#include "numberproperty.h"

#include "propertylist.h"

#include "stringproperty.h"

#include "tostring.h"

#include "propertylist.h"

SireBase::StringArrayProperty __copy__(const SireBase::StringArrayProperty &other){ return SireBase::StringArrayProperty(other); }

#include "Qt/qdatastream.hpp"

#include "Helpers/str.hpp"

#include "Helpers/len.hpp"

void register_StringArrayProperty_class(){

    { //::SireBase::StringArrayProperty
        typedef bp::class_< SireBase::StringArrayProperty, bp::bases< SireBase::Property > > StringArrayProperty_exposer_t;
        StringArrayProperty_exposer_t StringArrayProperty_exposer = StringArrayProperty_exposer_t( "StringArrayProperty", "", bp::init< >("") );
        bp::scope StringArrayProperty_scope( StringArrayProperty_exposer );
        StringArrayProperty_exposer.def( bp::init< QList< QString > const & >(( bp::arg("array") ), "") );
        StringArrayProperty_exposer.def( bp::init< QVector< QString > const & >(( bp::arg("array") ), "") );
        StringArrayProperty_exposer.def( bp::init< SireBase::StringArrayProperty const & >(( bp::arg("other") ), "") );
        StringArrayProperty_exposer.def( bp::self != bp::self );
        StringArrayProperty_exposer.def( bp::self + bp::self );
        { //::SireBase::StringArrayProperty::operator=
        
            typedef ::SireBase::StringArrayProperty & ( ::SireBase::StringArrayProperty::*assign_function_type)( ::SireBase::StringArrayProperty const & ) ;
            assign_function_type assign_function_value( &::SireBase::StringArrayProperty::operator= );
            
            StringArrayProperty_exposer.def( 
                "assign"
                , assign_function_value
                , ( bp::arg("other") )
                , bp::return_self< >()
                , "" );
        
        }
        StringArrayProperty_exposer.def( bp::self == bp::self );
        { //::SireBase::StringArrayProperty::typeName
        
            typedef char const * ( *typeName_function_type )(  );
            typeName_function_type typeName_function_value( &::SireBase::StringArrayProperty::typeName );
            
            StringArrayProperty_exposer.def( 
                "typeName"
                , typeName_function_value
                , "" );
        
        }
        StringArrayProperty_exposer.staticmethod( "typeName" );
        StringArrayProperty_exposer.def( "__copy__", &__copy__);
        StringArrayProperty_exposer.def( "__deepcopy__", &__copy__);
        StringArrayProperty_exposer.def( "clone", &__copy__);
        StringArrayProperty_exposer.def( "__rlshift__", &__rlshift__QDataStream< ::SireBase::StringArrayProperty >,
                            bp::return_internal_reference<1, bp::with_custodian_and_ward<1,2> >() );
        StringArrayProperty_exposer.def( "__rrshift__", &__rrshift__QDataStream< ::SireBase::StringArrayProperty >,
                            bp::return_internal_reference<1, bp::with_custodian_and_ward<1,2> >() );
        StringArrayProperty_exposer.def( "__str__", &__str__< ::SireBase::StringArrayProperty > );
        StringArrayProperty_exposer.def( "__repr__", &__str__< ::SireBase::StringArrayProperty > );
        StringArrayProperty_exposer.def( "__len__", &__len_size< ::SireBase::StringArrayProperty > );
        StringArrayProperty_exposer.def( "__getitem__", &::SireBase::StringArrayProperty::getitem );
    }

}
