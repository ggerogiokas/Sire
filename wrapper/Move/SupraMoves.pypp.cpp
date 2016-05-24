// This file has been generated by Py++.

// (C) Christopher Woods, GPL >= 2 License

#include "boost/python.hpp"
#include "Helpers/clone_const_reference.hpp"
#include "SupraMoves.pypp.hpp"

namespace bp = boost::python;

#include "SireError/errors.h"

#include "SireID/index.h"

#include "SireStream/datastream.h"

#include "SireStream/shareddatastream.h"

#include "supramoves.h"

#include "suprasystem.h"

#include "supramoves.h"

#include "Qt/qdatastream.hpp"

#include "Helpers/str.hpp"

#include "Helpers/len.hpp"

void register_SupraMoves_class(){

    { //::SireMove::SupraMoves
        typedef bp::class_< SireMove::SupraMoves, bp::bases< SireBase::Property >, boost::noncopyable > SupraMoves_exposer_t;
        SupraMoves_exposer_t SupraMoves_exposer = SupraMoves_exposer_t( "SupraMoves", "This is the base class of all sets of moves that can be applied\nto supra-systems\n\nAuthor: Christopher Woods\n", bp::no_init );
        bp::scope SupraMoves_scope( SupraMoves_exposer );
        { //::SireMove::SupraMoves::clearStatistics
        
            typedef void ( ::SireMove::SupraMoves::*clearStatistics_function_type)(  ) ;
            clearStatistics_function_type clearStatistics_function_value( &::SireMove::SupraMoves::clearStatistics );
            
            SupraMoves_exposer.def( 
                "clearStatistics"
                , clearStatistics_function_value
                , "" );
        
        }
        { //::SireMove::SupraMoves::count
        
            typedef int ( ::SireMove::SupraMoves::*count_function_type)(  ) const;
            count_function_type count_function_value( &::SireMove::SupraMoves::count );
            
            SupraMoves_exposer.def( 
                "count"
                , count_function_value
                , "Return the number of different types of move in this set" );
        
        }
        { //::SireMove::SupraMoves::move
        
            typedef void ( ::SireMove::SupraMoves::*move_function_type)( ::SireMove::SupraSystem &,int,bool ) ;
            move_function_type move_function_value( &::SireMove::SupraMoves::move );
            
            SupraMoves_exposer.def( 
                "move"
                , move_function_value
                , ( bp::arg("system"), bp::arg("nmoves"), bp::arg("record_stats")=(bool)(true) )
                , "" );
        
        }
        { //::SireMove::SupraMoves::moves
        
            typedef ::QList< SireBase::PropPtr< SireMove::SupraMove > > ( ::SireMove::SupraMoves::*moves_function_type)(  ) const;
            moves_function_type moves_function_value( &::SireMove::SupraMoves::moves );
            
            SupraMoves_exposer.def( 
                "moves"
                , moves_function_value
                , "" );
        
        }
        { //::SireMove::SupraMoves::nMoves
        
            typedef int ( ::SireMove::SupraMoves::*nMoves_function_type)(  ) const;
            nMoves_function_type nMoves_function_value( &::SireMove::SupraMoves::nMoves );
            
            SupraMoves_exposer.def( 
                "nMoves"
                , nMoves_function_value
                , "" );
        
        }
        { //::SireMove::SupraMoves::nSubMoveTypes
        
            typedef int ( ::SireMove::SupraMoves::*nSubMoveTypes_function_type)(  ) const;
            nSubMoveTypes_function_type nSubMoveTypes_function_value( &::SireMove::SupraMoves::nSubMoveTypes );
            
            SupraMoves_exposer.def( 
                "nSubMoveTypes"
                , nSubMoveTypes_function_value
                , "Return the number of different types of move in this set" );
        
        }
        { //::SireMove::SupraMoves::null
        
            typedef ::SireMove::SameSupraMoves const & ( *null_function_type )(  );
            null_function_type null_function_value( &::SireMove::SupraMoves::null );
            
            SupraMoves_exposer.def( 
                "null"
                , null_function_value
                , bp::return_value_policy< bp::copy_const_reference >()
                , "Return the global null SupraMoves object" );
        
        }
        { //::SireMove::SupraMoves::operator[]
        
            typedef ::SireMove::SupraMove const & ( ::SireMove::SupraMoves::*__getitem___function_type)( int ) const;
            __getitem___function_type __getitem___function_value( &::SireMove::SupraMoves::operator[] );
            
            SupraMoves_exposer.def( 
                "__getitem__"
                , __getitem___function_value
                , ( bp::arg("i") )
                , bp::return_value_policy<bp::clone_const_reference>()
                , "" );
        
        }
        { //::SireMove::SupraMoves::size
        
            typedef int ( ::SireMove::SupraMoves::*size_function_type)(  ) const;
            size_function_type size_function_value( &::SireMove::SupraMoves::size );
            
            SupraMoves_exposer.def( 
                "size"
                , size_function_value
                , "Return the number of different types of move in this set" );
        
        }
        { //::SireMove::SupraMoves::toString
        
            typedef ::QString ( ::SireMove::SupraMoves::*toString_function_type)(  ) const;
            toString_function_type toString_function_value( &::SireMove::SupraMoves::toString );
            
            SupraMoves_exposer.def( 
                "toString"
                , toString_function_value
                , "" );
        
        }
        { //::SireMove::SupraMoves::typeName
        
            typedef char const * ( *typeName_function_type )(  );
            typeName_function_type typeName_function_value( &::SireMove::SupraMoves::typeName );
            
            SupraMoves_exposer.def( 
                "typeName"
                , typeName_function_value
                , "" );
        
        }
        SupraMoves_exposer.staticmethod( "null" );
        SupraMoves_exposer.staticmethod( "typeName" );
        SupraMoves_exposer.def( "__rlshift__", &__rlshift__QDataStream< ::SireMove::SupraMoves >,
                            bp::return_internal_reference<1, bp::with_custodian_and_ward<1,2> >() );
        SupraMoves_exposer.def( "__rrshift__", &__rrshift__QDataStream< ::SireMove::SupraMoves >,
                            bp::return_internal_reference<1, bp::with_custodian_and_ward<1,2> >() );
        SupraMoves_exposer.def( "__str__", &__str__< ::SireMove::SupraMoves > );
        SupraMoves_exposer.def( "__repr__", &__str__< ::SireMove::SupraMoves > );
        SupraMoves_exposer.def( "__len__", &__len_size< ::SireMove::SupraMoves > );
    }

}
