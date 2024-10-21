uniform vec3 col;
varying vec3 vColor;
varying vec3 vNormal;

void main() {
    vec3 light = vec3(0.5, 0.2, 1.0);
    light = normalize(light);
    float dProd = max(0.0, dot(vNormal, light));
    dProd = 1.0;
    gl_FragColor = vec4((col * 0.01) + (vColor * dProd), 1.0);
}
