import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.plotwise.app',
  appName: 'Plotwise',
  webDir: 'frontend/src',
  server: {
    url: 'https://limajmr-plotwise.hf.space/mobile',
    cleartext: false
  },
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
