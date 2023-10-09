/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 */
#pragma once

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#include <stdio.h>

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif

size_t getPeakRSS();
size_t getCurrentRSS();

#define GET_PRE_RSS(funcName, rssBefore) \
    rssBefore = getCurrentRSS(); \
    SNN_LOGI("Memory usage %s before call : %.2lf MB", funcName, rssBefore / 1024.0 / 1024.0)

#define GET_POST_RSS(funcName, rssBefore, rssAfter) \
    rssAfter = getCurrentRSS(); \
    SNN_LOGI("Memory usage %s after call  : %.2lf MB", funcName, rssAfter / 1024.0 / 1024.0); \
    SNN_LOGI("Memory usage %s diff on call: %.2lf MB", funcName, ((int64_t)rssAfter - (int64_t)rssBefore) / 1024.0 / 1024.0); \
    SNN_LOGI("Memory usage %s peak on call: %.2lf MB", funcName, getPeakRSS() / 1024.0 / 1024.0)
