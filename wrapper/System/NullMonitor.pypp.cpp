// This file has been generated by Py++.

// (C) Christopher Woods, GPL >= 2 License

#include "boost/python.hpp"
#include "NullMonitor.pypp.hpp"

namespace bp = boost::python;

#include "SireStream/datastream.h"

#include "SireStream/shareddatastream.h"

#include "system.h"

#include "systemmonitor.h"

#include <QMutex>

#include "systemmonitor.h"

SireSystem::NullMonitor __copy__(const SireSystem::NullMonitor &other){ return SireSystem::NullMonitor(other); }

#include "Qt/qdatastream.hpp"

#include "Helpers/str.hpp"

void register_NullMonitor_class(){

    { //::SireSystem::NullMonitor
        typedef bp::class_< SireSystem::NullMonitor, bp::bases< SireSystem::SystemMonitor, SireBase::Property > > NullMonitor_exposer_t;
        NullMonitor_exposer_t NullMonitor_exposer = NullMonitor_exposer_t( "NullMonitor", "This is a null monitor that doesnt monitor anything", bp::init< >("Constructor") );
        bp::scope NullMonitor_scope( NullMonitor_exposer );
        NullMonitor_exposer.def( bp::init< SireSystem::NullMonitor const & >(( bp::arg("other") ), "Copy constructor") );
        { //::SireSystem::NullMonitor::clearStatistics
        
            typedef void ( ::SireSystem::NullMonitor::*clearStatistics_function_type)(  ) ;
            clearStatistics_function_type clearStatistics_function_value( &::SireSystem::NullMonitor::clearStatistics );
            
            NullMonitor_exposer.def( 
                "clearStatistics"
                , clearStatistics_function_value
                , "There are no statistics to clear" );
        
        }
        { //::SireSystem::NullMonitor::monitor
        
            typedef void ( ::SireSystem::NullMonitor::*monitor_function_type)( ::SireSystem::System & ) ;
            monitor_function_type monitor_function_value( &::SireSystem::NullMonitor::monitor );
            
            NullMonitor_exposer.def( 
                "monitor"
                , monitor_function_value
                , ( bp::arg("system") )
                , "A null monitor doesnt monitor anything" );
        
        }
        NullMonitor_exposer.def( bp::self != bp::self );
        { //::SireSystem::NullMonitor::operator=
        
            typedef ::SireSystem::NullMonitor & ( ::SireSystem::NullMonitor::*assign_function_type)( ::SireSystem::NullMonitor const & ) ;
            assign_function_type assign_function_value( &::SireSystem::NullMonitor::operator= );
            
            NullMonitor_exposer.def( 
                "assign"
                , assign_function_value
                , ( bp::arg("other") )
                , bp::return_self< >()
                , "" );
        
        }
        NullMonitor_exposer.def( bp::self == bp::self );
        { //::SireSystem::NullMonitor::typeName
        
            typedef char const * ( *typeName_function_type )(  );
            typeName_function_type typeName_function_value( &::SireSystem::NullMonitor::typeName );
            
            NullMonitor_exposer.def( 
                "typeName"
                , typeName_function_value
                , "" );
        
        }
        NullMonitor_exposer.staticmethod( "typeName" );
        NullMonitor_exposer.def( "__copy__", &__copy__);
        NullMonitor_exposer.def( "__deepcopy__", &__copy__);
        NullMonitor_exposer.def( "clone", &__copy__);
        NullMonitor_exposer.def( "__rlshift__", &__rlshift__QDataStream< ::SireSystem::NullMonitor >,
                            bp::return_internal_reference<1, bp::with_custodian_and_ward<1,2> >() );
        NullMonitor_exposer.def( "__rrshift__", &__rrshift__QDataStream< ::SireSystem::NullMonitor >,
                            bp::return_internal_reference<1, bp::with_custodian_and_ward<1,2> >() );
        NullMonitor_exposer.def( "__str__", &__str__< ::SireSystem::NullMonitor > );
        NullMonitor_exposer.def( "__repr__", &__str__< ::SireSystem::NullMonitor > );
    }

}
