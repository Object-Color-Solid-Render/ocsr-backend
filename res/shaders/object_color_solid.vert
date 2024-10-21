varying vec3 vNormal;
varying vec3 vColor;

attribute vec3 color;  // Declare 'color' as an attribute

void main() {
    vNormal = normal;
    vColor = color;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
