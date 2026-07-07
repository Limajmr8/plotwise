import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.plotwise.app',
  appName: 'Plotwise',
  // Bundled offline app: the mobile UI + AI model + data ship INSIDE the APK
  // (built into www/ by scripts/build_bundle.sh), so disease detection and the
  // data tools work with zero network. Live features call the hosted server
  // via the API base in plotwise-core.js. No server.url — assets load locally.
  webDir: 'www',
  android: {
    backgroundColor: '#060905',
    allowMixedContent: false,
    buildOptions: {
      keystorePath: undefined,
      keystoreAlias: undefined
    }
  },
  plugins: {
    SplashScreen: {
      launchShowDuration: 2000,
      launchAutoHide: true,
      backgroundColor: '#060905',
      showSpinner: false,
      androidSplashResourceName: 'splash',
      androidScaleType: 'CENTER_CROP',
      splashFullScreen: true,
      splashImmersive: true
    },
    StatusBar: {
      style: 'DARK',
      backgroundColor: '#060905'
    },
    Camera: {
      presentationStyle: 'fullscreen',
      quality: 85
    }
  }
};

export default config;
