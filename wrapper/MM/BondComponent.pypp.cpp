// This file has been generated by Py++.

// (C) Christopher Woods, GPL >= 2 License

#include "boost/python.hpp"
#include "Helpers/clone_const_reference.hpp"
#include "BondComponent.pypp.hpp"

namespace bp = boost::python;

#include "SireFF/ff.h"

#include "SireStream/datastream.h"

#include "internalcomponent.h"

#include "internalcomponent.h"

SireMM::BondComponent __copy__(const SireMM::BondComponent &other){ return SireMM::BondComponent(other); }

#include "Qt/qdatastream.hpp"

#include "Helpers/str.hpp"

void register_BondComponent_class(){

    { //::SireMM::BondComponent
        typedef bp::class_< SireMM::BondComponent, bp::bases< SireFF::FFComponent, SireCAS::Symbol, SireCAS::ExBase > > BondComponent_exposer_t;
        BondComponent_exposer_t BondComponent_exposer = BondComponent_exposer_t( "BondComponent", "This class represents a Bond component of a forcefield", bp::init< bp::optional< SireFF::FFName const & > >(( bp::arg("ffname")=SireFF::FFName() ), "Constructor") );
        bp::scope BondComponent_scope( BondComponent_exposer );
        BondComponent_exposer.def( bp::init< SireCAS::Symbol const & >(( bp::arg("symbol") ), "Construct from a symbol\nThrow: SireError::incompatible_error\n") );
        BondComponent_exposer.def( bp::init< SireMM::BondComponent const & >(( bp::arg("other") ), "Copy constructor") );
        { //::SireMM::BondComponent::changeEnergy
        
            typedef void ( ::SireMM::BondComponent::*changeEnergy_function_type)( ::SireFF::FF &,::SireMM::BondEnergy const & ) const;
            changeEnergy_function_type changeEnergy_function_value( &::SireMM::BondComponent::changeEnergy );
            
            BondComponent_exposer.def( 
                "changeEnergy"
                , changeEnergy_function_value
                , ( bp::arg("ff"), bp::arg("bondnrg") )
                , "Change the component of the energy in the forcefield ff\nby delta" );
        
        }
        { //::SireMM::BondComponent::setEnergy
        
            typedef void ( ::SireMM::BondComponent::*setEnergy_function_type)( ::SireFF::FF &,::SireMM::BondEnergy const & ) const;
            setEnergy_function_type setEnergy_function_value( &::SireMM::BondComponent::setEnergy );
            
            BondComponent_exposer.def( 
                "setEnergy"
                , setEnergy_function_value
                , ( bp::arg("ff"), bp::arg("bondnrg") )
                , "Set the component of the energy in the forcefield ff\nto be equal to the passed energy" );
        
        }
        { //::SireMM::BondComponent::symbols
        
            typedef ::SireCAS::Symbols ( ::SireMM::BondComponent::*symbols_function_type)(  ) const;
            symbols_function_type symbols_function_value( &::SireMM::BondComponent::symbols );
            
            BondComponent_exposer.def( 
                "symbols"
                , symbols_function_value
                , "" );
        
        }
        { //::SireMM::BondComponent::total
        
            typedef ::SireMM::BondComponent const & ( ::SireMM::BondComponent::*total_function_type)(  ) const;
            total_function_type total_function_value( &::SireMM::BondComponent::total );
            
            BondComponent_exposer.def( 
                "total"
                , total_function_value
                , bp::return_value_policy<bp::clone_const_reference>()
                , "" );
        
        }
        { //::SireMM::BondComponent::typeName
        
            typedef char const * ( *typeName_function_type )(  );
            typeName_function_type typeName_function_value( &::SireMM::BondComponent::typeName );
            
            BondComponent_exposer.def( 
                "typeName"
                , typeName_function_value
                , "" );
        
        }
        { //::SireMM::BondComponent::what
        
            typedef char const * ( ::SireMM::BondComponent::*what_function_type)(  ) const;
            what_function_type what_function_value( &::SireMM::BondComponent::what );
            
            BondComponent_exposer.def( 
                "what"
                , what_function_value
                , "" );
        
        }
        BondComponent_exposer.staticmethod( "typeName" );
        BondComponent_exposer.def( "__copy__", &__copy__);
        BondComponent_exposer.def( "__deepcopy__", &__copy__);
        BondComponent_exposer.def( "clone", &__copy__);
        BondComponent_exposer.def( "__rlshift__", &__rlshift__QDataStream< ::SireMM::BondComponent >,
                            bp::return_internal_reference<1, bp::with_custodian_and_ward<1,2> >() );
        BondComponent_exposer.def( "__rrshift__", &__rrshift__QDataStream< ::SireMM::BondComponent >,
                            bp::return_internal_reference<1, bp::with_custodian_and_ward<1,2> >() );
        BondComponent_exposer.def( "__str__", &__str__< ::SireMM::BondComponent > );
        BondComponent_exposer.def( "__repr__", &__str__< ::SireMM::BondComponent > );
        BondComponent_exposer.def( "__hash__", &::SireMM::BondComponent::hash );
    }

}
