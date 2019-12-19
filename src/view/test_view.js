

import Tween from 'rc-tween-one';
import React from 'react';
import ReactDom from 'react-dom';
import BezierPlugin from 'rc-tween-one/lib/plugin/BezierPlugin';

Tween.plugins.push(BezierPlugin);


function Demo() {
    return (
        <div style={{ position: 'relative', height: 600, marginTop:100 }}>
            <Tween
                animation={{
                    bezier: {
                        type: 'thru', autoRotate: true,
                        vars: [{ x: 300, y: 0 }, { x: 300+300, y: 300 }, { x: 300, y: 300+300 }, { x: 0, y: 300 },
                            { x: 300, y: 0 },],
                    },
                    repeat: 11,
                    // yoyo: true,
                    duration: 5000,
                }}
                style={{ width: 100 }}
            >
                <div style={{width:10,height:10,borderRadius:5,backgroundColor:'red'}}></div>
            </Tween>
            {/*<div*/}
            {/*    style={{*/}
            {/*        width: 5, height: 5, background: '#000',*/}
            {/*        position: 'absolute', top: 0, transform: 'translate(200px,200px)',*/}
            {/*    }}*/}
            {/*/>*/}
            {/*<div*/}
            {/*    style={{*/}
            {/*        width: 5, height: 5, background: '#000', position: 'absolute',*/}
            {/*        top: 0, transform: 'translate(400px,0px)',*/}
            {/*    }}*/}
            {/*/>*/}
            {/*<div*/}
            {/*    style={{*/}
            {/*        width: 5, height: 5, background: '#000', position: 'absolute',*/}
            {/*        top: 0, transform: 'translate(600px,200px)',*/}
            {/*    }}*/}
            {/*/>*/}
            {/*<div*/}
            {/*    style={{*/}
            {/*        width: 5, height: 5, background: '#000', position: 'absolute',*/}
            {/*        top: 0, transform: 'translate(800px,0px)',*/}
            {/*    }}*/}
            {/*/>*/}
        </div>);
}

export default Demo