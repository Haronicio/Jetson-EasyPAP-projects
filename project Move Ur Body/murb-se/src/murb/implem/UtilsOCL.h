#ifndef UTILSOCL_HPP_
#define UTILSOCL_HPP_

/*!
 * \struct dataAoS_t
 * \brief  Structure of body characteristics.
 *
 * The characteristics of a body.
 */
struct dataAoS_t {
    float qx; /*!< Position x. */
    float qy; /*!< Position y. */
    float qz; /*!< Position z. */
    float vx; /*!< Velocity x. */
    float vy; /*!< Velocity y. */
    float vz; /*!< Velocity z. */
    float m;  /*!< Mass. */
    float r;  /*!< Radius. */
};

/*!
 * \struct accAoS_t
 * \brief  Structure of body acceleration.
 *
 * The body acceleration.
 */
struct accAoS_t {
    float ax; /*!< Acceleration x. */
    float ay; /*!< Acceleration y. */
    float az; /*!< Acceleration z. */
};

/*!
 * \struct dataSoA_t
 * \brief  Structure of arrays.
 *
 * The dataSoA_t structure represent the characteristics of the bodies.
 */
struct dataSoA_t {
    float *qx; /*!< Array of positions x. */
    float *qy; /*!< Array of positions y. */
    float *qz; /*!< Array of positions z. */
    float *vx; /*!< Array of velocities x. */
    float *vy; /*!< Array of velocities y. */
    float *vz; /*!< Array of velocities z. */
    float *m;  /*!< Array of masses. */
    float *r;  /*!< Array of radiuses. */
};


/*!
 * \struct accSoA_t
 * \brief  Structure of arrays.
 *
 * The accSoA_t structure represent the accelerations of the bodies.
 */
struct accSoA_t {
    float *ax; /*!< Array of accelerations x. */
    float *ay; /*!< Array of accelerations y. */
    float *az; /*!< Array of accelerations z. */
};


#endif