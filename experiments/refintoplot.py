from earthadjutil import *
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams.update({'font.size': 28})

of="/home/sven/exa/adjoint/forward/output/"
th="/home/sven/thesis/images/"



plotrefsave(np.load(of+"pwaves/manual12.npy"),(30,15),(-5,0),th+"pw12manual.png",5,5,12,5)
plotrefsave(np.load(of+"pwaves/inner12.npy"),(30,15),(-5,0),th+"pw12inner.png",5,5,12,5)




plotrefsave(np.load(of+"simple/inner13a.npy"),(20,30),(0,0),th+"smpl13a.png",7,15,9,13)
plotrefsave(np.load(of+"simple/inner13stress.npy"),(20,30),(0,0),th+"smpl13stressa.png",7,15,9,13)
plotrefsave(np.load(of+"simple/manual13a.npy"),(20,30),(0,0),th+"smpl13manual.png",7,15,9,13)

plotrefsave(np.load(of+"simple/inner18a.npy"),(20,30),(0,0),th+"smpl18a.png",7,15,13,18)
plotrefsave(np.load(of+"simple/inner18stress.npy"),(20,30),(0,0),th+"smpl18stressa.png",7,15,13,18)
plotrefsave(np.load(of+"simple/manual18a.npy"),(20,30),(0,0),th+"smpl18manual.png",7,15,13,18)

plotrefsave(np.load(of+"wel/manual11D.npy"),(30,15),(-5,0),th+"wel11manualD",5,5,11,2)
plotrefsave(np.load(of+"wel/manual11C.npy"),(30,15),(-5,0),th+"wel11manualC",5,5,11,2)
plotrefsave(np.load(of+"wel/inner11C.npy"),(30,15),(-5,0),th+"wel11C",5,5,11,2)
plotrefsave(np.load(of+"wel/inner11D.npy"),(30,15),(-5,0),th+"wel11D",5,5,11,2)
plotrefsave(np.load(of+"wel/inner11stressD.npy"),(30,15),(-5,0),th+"wel11stressD",5,5,11,2)
plotrefsave(np.load(of+"wel/inner11stressC.npy"),(30,15),(-5,0),th+"wel11stressC",5,5,11,2)