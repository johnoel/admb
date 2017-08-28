@echo off
setlocal
if [%1]==[--help] goto HELP
REM ############################################################################
REM                                                                            #
REM Script:   ide-distribution                                                 #
REM                                                                            #
REM Purpose:  Arrange ADMB-IDE components for distribution                     #
REM                                                                            #
REM Requires: Directory 'components' created by ide-components                 #
REM                                                                            #
REM Returns:  Creates a directory called 'distribution'                        #
REM                                                                            #
REM History:  26 Aug 2017 Arni Magnusson created                               #
REM                                                                            #
REM ############################################################################

set FROM=components
set TO=distribution
set CP=%SYSTEMROOT%\system32\robocopy.exe

rd /q /s %TO% 2>NUL
md %TO%

echo on
@echo.
@echo *** Populating ~ ...
%CP% %FROM% %TO%\~ .emacs >NUL
%CP% %FROM% %TO%\~\emacs\lisp\admb admb.el > NUL
%CP% %FROM%\auctex %TO%\~\emacs\lisp\auctex /e > NUL
%CP% %FROM%\ess %TO%\~\emacs\lisp\ess /e > NUL
%CP% %FROM% %TO%\~\emacs\lisp\tmb tmb.el > NUL
%CP% %FROM% %TO%\~\icons admb64.ico > NUL
@echo.
@echo *** Populating admb ...
for /F "usebackq tokens=*" %%F in (`dir /ad /b %FROM%\admb*-*-*`) do set AD=%%F
%CP% %FROM%\%AD% %TO%\admb\%AD% /e > NUL
%CP% %FROM% %TO%\admb *.pdf NEWS > NUL
@echo.
@echo *** Populating gnu ...
%CP% %FROM%\emacs %TO%\gnu\emacs /e > NUL
@echo.
@echo *** Populating Rtools ...
%CP% %FROM%\Rtools %TO%\Rtools /e > NUL
@echo.
@echo Done

@goto EOF

:HELP
echo Usage: ide-distribution
echo.

:EOF