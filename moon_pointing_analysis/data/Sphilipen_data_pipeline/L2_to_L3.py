#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
#METAPROJECT /cvmfs/icecube.opensciencegrid.org/users/sphilippen/metaprojectV05-02-00a/build
import sys, os, glob
from icecube import icetray, dataio, dataclasses, hdfwriter, phys_services, lilliput, gulliver, gulliver_modules, linefit
from I3Tray import *
from os.path import expandvars
from icecube.icetray import I3Units
import icecube.lilliput.segments
load('libDomTools')
from writeOutputBlacklist import WriteOutput
from recoFunctions import GetBestTrack
from reconstructions import DoReconstructions
from SplitHiveSplitter import SplitAndRecoHiveSplitter, TimeWindowCollector
from time import time

MC = False
dataset = 11937 # if MC=True

rloglCut = 14	
NChCut = 12 
horizon = 1.361356816555577

photonicsdir = "/cvmfs/icecube.opensciencegrid.org/data/photon-tables/SPICEMie/"
photonicsdriverdir = "/cvmfs/icecube.opensciencegrid.org/data/photon-tables/SPICEMie/driverfiles"
photonicsdriverfile = "mu_photorec.list"
infmuonampsplinepath = "/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/InfBareMu_mie_abs_z20a10_V2.fits"
infmuonprobsplinepath = "/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/InfBareMu_mie_prob_z20a10_V2.fits"
cascadeampsplinepath = "/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/ems_mie_z20_a10.abs.fits"
cascadeprobsplinepath = "/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/ems_mie_z20_a10.prob.fits"
stochampsplinepath = "/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/InfHighEStoch_mie_abs_z20a10.fits"
stochprobsplinepath = "/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/InfHighEStoch_mie_prob_z20a10.fits"

name = "MoonLevel3"
suffix = ""
class RunParameters():
        GCDFile=''
        InputFile=[]
        OutputDir=''
        OutputFile=''
        HD5File=''
        ROOTFile=''
        stats={}
        InfMuonAmpSplinePath='/data/sim/sim-new/spline-tables/InfBareMu_mie_abs_z20a10.fits'
        InfMuonProbSplinePath='/data/sim/sim-new/spline-tables/InfBareMu_mie_prob_z20a10.fits'
        CascadeAmpSplinePath='/data/sim/sim-new/spline-tables/ems_mie_z20_a10.abs.fits'
        CascadeProbSplinePath='/data/sim/sim-new/spline-tables/ems_mie_z20_a10.prob.fits'
        PhotonicsTableDir='/home/jfeintzeig/netuser/Tables/mie/SPICEMie/level2/standard/'
        PhotonicsDriverDir=PhotonicsTableDir+'misc/'
        PhotonicsDriverFile='level2_table_DEGEN.list'

def main(params):
        files = [params.GCDFile]
        files += params.InputFile

        tray = I3Tray()

        tray.AddSegment(dataio.I3Reader, 'reader', FilenameList=files)
        
        def takeEvents(frame):
                if MC:
                        if "MPEFit" in frame:
                                if frame["MPEFit"].dir.zenith < 1.919862177 and frame["MPEFit"].dir.zenith > 0.8726646259971648:
                                        return True
                                else:
                                        return False
                        else:
                                return False
                else:
                        try:
                                if (frame['FilterMask']["MoonFilter_13"].condition_passed and frame["I3DST"].ndom >= NChCut):
                                        return True
                                else:
                                        return False
                        except KeyError:
                                return False
        def OnlyInIce(frame):
                if frame['I3EventHeader'].sub_event_stream == 'InIceSplit':
                        return True
                else:
                        return False
	

        tray.AddModule(OnlyInIce,'inice')
        tray.AddModule(takeEvents,"firstNChCut")

        tray.AddSegment(SplitAndRecoHiveSplitter, Suffix="HV", NChCut=NChCut)
	
        if MC:
                from icecube.weighting import CORSIKAWeightCalculator
                from icecube.weighting import fluxes
                from icecube import weighting
                tray.Add(CORSIKAWeightCalculator, 'GaisserH3aWeight', Dataset=dataset, Flux=fluxes.GaisserH3a())


        #Do level3 reconstructions
        tray.AddSegment(DoReconstructions,
	        Pulses="TWSRTInIcePulses",
	        Suffix=suffix, rloglCut = rloglCut, 
	        photonicsdir=photonicsdir,
	        photonicsdriverdir=photonicsdriverdir,
	        photonicsdriverfile=photonicsdriverfile,
	        infmuonampsplinepath=infmuonampsplinepath,
	        infmuonprobsplinepath=infmuonprobsplinepath,
	        cascadeampsplinepath=cascadeampsplinepath,
	        cascadeprobsplinepath=cascadeprobsplinepath,
		stochprobsplinepath=stochprobsplinepath,
		stochampsplinepath=stochampsplinepath,
		If=lambda frame: frame["I3EventHeader"].sub_event_stream=="Final")
	       
        tray.AddModule(GetBestTrack,"best_track",
                        Suffix=suffix,
                        If=lambda frame: frame["I3EventHeader"].sub_event_stream=="Final")

        def compressTruncatedEnergy(frame):
	        truncKeys = [
                        "SplineMPETruncatedEnergy_SPICEMie_AllBINS_MuEres", 
                        "SplineMPETruncatedEnergy_SPICEMie_AllBINS_Muon", 
                        "SplineMPETruncatedEnergy_SPICEMie_AllBINS_Neutrino", 
                        "SplineMPETruncatedEnergy_SPICEMie_AllBINS_dEdX", 
		        "SplineMPETruncatedEnergy_SPICEMie_AllDOMS_MuEres", 
                        "SplineMPETruncatedEnergy_SPICEMie_AllDOMS_Muon", 
                        "SplineMPETruncatedEnergy_SPICEMie_AllDOMS_Neutrino", 
                        "SplineMPETruncatedEnergy_SPICEMie_AllDOMS_dEdX",
		        "SplineMPETruncatedEnergy_SPICEMie_BINS_MuEres", 
                        "SplineMPETruncatedEnergy_SPICEMie_BINS_Muon", 
                        "SplineMPETruncatedEnergy_SPICEMie_BINS_Neutrino", 
                        "SplineMPETruncatedEnergy_SPICEMie_BINS_dEdX",
		        "SplineMPETruncatedEnergy_SPICEMie_DOMS_MuEres", 
                        "SplineMPETruncatedEnergy_SPICEMie_DOMS_Muon", 
                        "SplineMPETruncatedEnergy_SPICEMie_DOMS_Neutrino", 
                        "SplineMPETruncatedEnergy_SPICEMie_DOMS_dEdX",
		        "SplineMPETruncatedEnergy_SPICEMie_ORIG_Muon", 
                        "SplineMPETruncatedEnergy_SPICEMie_ORIG_Neutrino", 
                        "SplineMPETruncatedEnergy_SPICEMie_ORIG_dEdX",
                ]

                dicKeys = [
                        "ABRes", 
                        "ABMu", 
                        "ABNu", 
                        "ABdEdX", 
                        "ADRes", 
                        "ADMu", 
                        "ADNu", 
                        "ADdEdX", 
                        "BRes", 
                        "BMu", 
                        "BNu", 
                        "BdEdX", 
                        "DRes", 
                        "DMu", 
                        "DNu", 
                        "DdEdX", 
                        "ORIGMu", 
                        "ORIGNu", 
                        "ORIGdEdX",
                ]

	        TruncatedEnergyDic = {}
	        for i in range(len(truncKeys)):
		        if truncKeys[i] in frame: 
		    	        try: truncEn = frame[truncKeys[i]].value 
		    	        except: truncEn = frame[truncKeys[i]].energy
		        else: truncEn = -1
		        frame.Delete(truncKeys[i])
		        TruncatedEnergyDic[dicKeys[i]] = truncEn 
	        frame.Put("SplineMPETruncatedEnergy", dataclasses.I3MapStringDouble(TruncatedEnergyDic))
	tray.Add(compressTruncatedEnergy)

	def compressMuExDiff_list_Millipede(frame):
		if "SplineMPEICMuEXDifferential_list" in frame:
	    		MuExEn = []
	    		for i in range(len(frame["SplineMPEICMuEXDifferential_list"])):
	    			MuExEn.append(frame["SplineMPEICMuEXDifferential_list"][i].energy)
	    		frame.Delete("SplineMPEICMuEXDifferential_list")
	    		frame.Put("SplineMPEICMuEXDifferential_list", dataclasses.I3VectorFloat(MuExEn))
		if "SplineMPEIC_MillipedeHighEnergyMIE" in frame:
	    		MillipedeEn = []
	    		for i in range(len(frame["SplineMPEIC_MillipedeHighEnergyMIE"])):
	    			MillipedeEn.append(frame["SplineMPEIC_MillipedeHighEnergyMIE"][i].energy)
	    		frame.Delete("SplineMPEIC_MillipedeHighEnergyMIE")
	    		frame.Put("SplineMPEIC_MillipedeHighEnergyMIE", dataclasses.I3VectorFloat(MillipedeEn ))

	tray.Add(compressMuExDiff_list_Millipede)


        tray.AddSegment(WriteOutput,'write',params=params, Suffix=suffix)

        tray.AddModule('TrashCan','can')

        tray.Execute()


        usagemap = tray.Usage()

        for mod in usagemap:
                print(mod)

        tray.Finish()

        #if os.path.isfile(params.OutputDir+params.OutputFile) and 'bz2' not in params.OutputFile:
                #os.system('bzip2 -f %s' % params.OutputDir+params.OutputFile)

def parseOptions(parser,params):
        parser.add_option('-g', '--gcdfile', default='',
                          dest='GCDFILE', help='complete path to gcd file')
        parser.add_option('-i', '--inputfile', default=[],
                          dest='INPUTFILE', help='complete path to input file')
        parser.add_option('-o', '--outputdir', default='',
                          dest='OUTPUTDIR', help='complete path to output directory')

        (options,args) = parser.parse_args()

        params.GCDFile = options.GCDFILE
        #InputFileList = glob.glob(options.INPUTFILE)
	InputFileList = [options.INPUTFILE]
        for item in InputFileList:
                if 'EHE' not in item and 'SLOP' not in item and 'IT' not in item:
                        params.InputFile+=[item]
        params.InputFile.sort()
        params.OutputDir = options.OUTPUTDIR

        params.OutputFile = options.INPUTFILE.split('/')[-1].replace('Level2','Level3').strip('.bz2').replace('*','').replace('[0-9]','')
        #params.HD5File = params.OutputFile.replace('.i3','.hd5')
        #params.ROOTFile = params.OutputFile.replace('.i3','.root')

        print("Infiles:", params.InputFile)
        print("Outfiles:", params.OutputFile) #, params.HD5File, params.ROOTFile

#iceprod stuff
try:
   from iceprod.modules import ipmodule
except ImportError, e:
   print('Module iceprod.modules not found. Will not define IceProd Class')
else:
  class IC86MuonLevel3(ipmodule.IPBaseClass):
    '''
    Wrapper class that runs Level 1 and Level 2 filters for IC86
    '''
    def __init__(self):
       ipmodule.IPBaseClass.__init__(self)

       params = RunParameters()

       self.AddParameter('infile',
                        'files to be processed',
                        params.InputFile)

       self.AddParameter('gcdfile',
                        'input gcd file to use',
                        params.GCDFile)

       self.AddParameter('outdir',
                        'directory where the output files will be written',
                        params.OutputDir)

       self.AddParameter('InfMuonAmpSplinePath',
                        'path for amplitude spline table for SplineMPE',
                        params.InfMuonAmpSplinePath)

       self.AddParameter('InfMuonProbSplinePath',
                        'path for probability spline table for SplineMPE',
                        params.InfMuonProbSplinePath)

       self.AddParameter('CascadeAmpSplinePath',
                        'path for amplitude spline table for Millipede',
                        params.CascadeAmpSplinePath)

       self.AddParameter('CascadeProbSplinePath',
                        'path for probability spline table for Millipede',
                        params.CascadeProbSplinePath)

       self.AddParameter('PhotonicsTableDir',
                        'path to directory for level2 muon tables for Truncated',
                        params.PhotonicsTableDir)

       self.AddParameter('PhotonicsDriverDir',
                        'path to directory for level2 driver file for Truncated',
                        params.PhotonicsDriverDir)

       self.AddParameter('PhotonicsDriverFile',
                        'name of level2 driver file for Truncated',
                        params.PhotonicsDriverFile)

    def Execute(self,stats):
            if not ipmodule.IPBaseClass.Execute(self,stats): return 0

            import icecube.icetray

            params               = RunParameters()
            params.stats         = stats

            InputFileParam       = self.GetParameter('infile')
            InputFileList = glob.glob(InputFileParam)
            for item in InputFileList:
                if 'EHE' not in item and 'SLOP' not in item and 'IT' not in item:
                    params.InputFile+=[item]
            params.InputFile.sort()
            print(params.InputFile)
            params.GCDFile       = self.GetParameter('gcdfile')
            params.OutputDir     = self.GetParameter('outdir')
            params.OutputFile    = params.InputFileParam.split('/')[-1].replace('Level2','Level3').strip('.bz2').replace('*','')
            params.HD5File       = params.OutputFile.replace('.i3','.hd5')
            params.ROOTFile      = params.OutputFile.replace('.i3','.root')

            print("Infiles:", params.InputFile)
            print("Outfiles:", params.OutputFile, params.HD5File, params.ROOTFile)

            params.InfMuonAmpSplinePath = self.GetParameter('InfMuonAmpSplinePath')
            params.InfMuonProbSplinePath = self.GetParameter('InfMuonProbSplinePath')
            params.CascadeAmpSplinePath = self.GetParameter('CascadeAmpSplinePath')
            params.CascadeProbSplinePath = self.GetParameter('CascadeProbSplinePath')
            params.PhotonicsTableDir = self.GetParameter('PhotonicsTableDir')
            params.PhotonicsDriverDir = self.GetParameter('PhotonicsDriverDir')
            params.PhotonicsDriverFile = self.GetParameter('PhotonicsDriverFile')

            main(params)
            return 0

# the business!
if (__name__=='__main__'):
     t = time()
     from optparse import OptionParser

     #get parameters, parse etc
     params = RunParameters()

     usage = 'usage: %prog [options]'
     parser = OptionParser(usage)

     parseOptions(parser, params)

     main(params)
     print(time() - t)
