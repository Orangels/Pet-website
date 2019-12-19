const { override, fixBabelImports, addLessLoader } = require('customize-cra');

module.exports = override(
    fixBabelImports('import', {
        libraryName: 'antd',
        libraryDirectory: 'es',
        style: true,
    }),
    addLessLoader({
        javascriptEnabled: true,
        modifyVars: { '@primary-color': '#FC8732',
            '@link-color':'#FC492A',
            // '@link-color':'#25b864',
            '@switch-color':'#25b864',
            '@font-family': "-apple-system, , BlinkMacSystemFont,'Segoe UI', 'PingFang SC', 'Hiragino Sans GB',\n" +
                "  'Microsoft YaHei', 'Helvetica Neue', Helvetica, Arial, sans-serif, 'Apple Color Emoji',\n" +
                "  'Segoe UI Emoji', 'Segoe UI Symbol';"},
    }),
);

// '@primary-color': '#FBA339'