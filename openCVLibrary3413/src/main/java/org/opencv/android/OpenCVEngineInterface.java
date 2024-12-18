package org.opencv.android;


/**
 * Class provides a Java interface for OpenCV Engine Service. It's synchronous with native OpenCVEngine class.
 */
interface OpenCVEngineInterface
{
    Object Stub = 1;

    /**
     * @return Returns service version.
     */
    int getEngineVersion();

    /**
     * Finds an installed OpenCV library.
     * @return Returns path to OpenCV native libs or an empty string if OpenCV can not be found.
     */
    String getLibPathByVersion(String version);

    /**
     * Tries to install defined version of OpenCV from Google Play Market.
     * @return Returns true if installation was successful or OpenCV package has been already installed.
     */
    boolean installVersion(String version);

    /**
     * Returns list of libraries in loading order, separated by semicolon.
     * @return Returns names of OpenCV libraries, separated by semicolon.
     */
    String getLibraryList(String version);
}
