uniform vec3 col;
varying vec3 vColor;
varying vec3 vNormal;

void main() {
    float ambient_strength = 0.8;
    float saturation_factor = 1.2;  // not accurate but looks good
    vec3 vn = normalize(vNormal);
    vec3 light_dir = normalize(vec3(1, 1, 1));
    float light_strength = max(ambient_strength, dot(vn, light_dir));
    vec3 garbage_term = (col * 0.01);   // here to use uniform col so shader compiles
    gl_FragColor = vec4((vColor * light_strength) * saturation_factor + garbage_term, 1.0);
}
