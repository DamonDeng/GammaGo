#include <iostream>

#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/utility.hpp>

#include "SgDebug.h"
#include "SgException.h"
#include "GtpInputStream.h"
#include "GtpOutputStream.h"
#include "GoGtpEngine.h"


using namespace std;

using std::ostream;
using std::string;
using std::vector;

using boost::filesystem::path;
namespace po = boost::program_options;

namespace{
    
    struct CommandLineOptions {
        bool m_quiet;
    };

    void Help(po::options_description& desc, ostream& out)
    {
        out << "Usage: omegago [options] [input files]\n" << desc << "\n";
        exit(0);
    }

    void ParseOptions(int argc, char** argv, struct CommandLineOptions& options)
    {
        options.m_quiet = false;

        po::options_description normalOptions("Options");
        normalOptions.add_options()
            ("help", "Displays this help and exit")
            ("quiet", "don't print debug messages");

        po::options_description allOptions;
        allOptions.add(normalOptions);

        po::variables_map vm;
        try
        {
            po::store(po::command_line_parser(argc, argv).options(allOptions).run(), vm);
            po::notify(vm);
        }
        catch (...)
        {
            Help(normalOptions, std::cerr);
        }
        if (vm.count("help")){
            Help(normalOptions, std::cout);
        }

        if (vm.count("quiet")){
            options.m_quiet = true;
        }
    }


} // namespace



int main(int argc, char** argv){

    /** Settings from command line options */
    struct CommandLineOptions options;

    if (argc > 0 && argv != 0){
        try{
            ParseOptions(argc, argv, options);
        }
        catch (const SgException& e){
            SgDebug() << e.what() << "\n";
            return 1;
        }
    }

    if (options.m_quiet){
        // cout << "setting sgdebug to quite." << endl;
        SgDebugToNull();
    }

    SgDebug() << "# Starting the main program of GammaGo. \n";

    GoGtpEngine engine;

    GtpInputStream in(std::cin);
    GtpOutputStream out(std::cout);
    engine.MainLoop(in, out);

}