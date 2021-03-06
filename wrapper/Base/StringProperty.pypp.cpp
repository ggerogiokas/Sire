// This file has been generated by Py++.

// (C) Christopher Woods, GPL >= 2 License

#include "boost/python.hpp"
#include "StringProperty.pypp.hpp"

namespace bp = boost::python;

#include "SireStream/datastream.h"

#include "SireStream/shareddatastream.h"

#include "SireStream/sharestrings.h"

#include "stringproperty.h"

#include "stringproperty.h"

SireBase::StringProperty __copy__(const SireBase::StringProperty &other){ return SireBase::StringProperty(other); }

#include "Qt/qdatastream.hpp"

#include "Helpers/str.hpp"

void register_StringProperty_class(){

    { //::SireBase::StringProperty
        typedef bp::class_< SireBase::StringProperty, bp::bases< SireBase::Property > > StringProperty_exposer_t;
        StringProperty_exposer_t StringProperty_exposer = StringProperty_exposer_t( "StringProperty", "This class provides a thin Property wrapper around a QString\n\nAuthor: Christopher Woods\n", bp::init< >("Constructor") );
        bp::scope StringProperty_scope( StringProperty_exposer );
        StringProperty_exposer.def( bp::init< QString const & >(( bp::arg("s") ), "Construct from the passed string") );
        StringProperty_exposer.def( bp::init< SireBase::VariantProperty const & >(( bp::arg("other") ), "Cast from the passed VariantProperty") );
        StringProperty_exposer.def( bp::init< SireBase::StringProperty const & >(( bp::arg("other") ), "Copy constructor") );
        StringProperty_exposer.def( bp::self != bp::self );
        { //::SireBase::StringProperty::operator=
        
            typedef ::SireBase::StringProperty & ( ::SireBase::StringProperty::*assign_function_type)( ::SireBase::StringProperty const & ) ;
            assign_function_type assign_function_value( &::SireBase::StringProperty::operator= );
            
            StringProperty_exposer.def( 
                "assign"
                , assign_function_value
                , ( bp::arg("other") )
                , bp::return_self< >()
                , "" );
        
        }
        StringProperty_exposer.def( bp::self == bp::self );
        { //::SireBase::StringProperty::toString
        
            typedef ::QString ( ::SireBase::StringProperty::*toString_function_type)(  ) const;
            toString_function_type toString_function_value( &::SireBase::StringProperty::toString );
            
            StringProperty_exposer.def( 
                "toString"
                , toString_function_value
                , "" );
        
        }
        { //::SireBase::StringProperty::typeName
        
            typedef char const * ( *typeName_function_type )(  );
            typeName_function_type typeName_function_value( &::SireBase::StringProperty::typeName );
            
            StringProperty_exposer.def( 
                "typeName"
                , typeName_function_value
                , "" );
        
        }
        StringProperty_exposer.staticmethod( "typeName" );
        StringProperty_exposer.def( "__copy__", &__copy__);
        StringProperty_exposer.def( "__deepcopy__", &__copy__);
        StringProperty_exposer.def( "clone", &__copy__);
        StringProperty_exposer.def( "__rlshift__", &__rlshift__QDataStream< ::SireBase::StringProperty >,
                            bp::return_internal_reference<1, bp::with_custodian_and_ward<1,2> >() );
        StringProperty_exposer.def( "__rrshift__", &__rrshift__QDataStream< ::SireBase::StringProperty >,
                            bp::return_internal_reference<1, bp::with_custodian_and_ward<1,2> >() );
        StringProperty_exposer.def( "__str__", &__str__< ::SireBase::StringProperty > );
        StringProperty_exposer.def( "__repr__", &__str__< ::SireBase::StringProperty > );
    }

}
