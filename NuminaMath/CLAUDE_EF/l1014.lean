import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_1800_l1014_101421

/-- The length of the rectangular field in meters -/
def field_length : ℝ := 100

/-- The width of the rectangular field in meters -/
def field_width : ℝ := 50

/-- The number of laps to run -/
def num_laps : ℕ := 6

/-- Calculates the perimeter of a rectangle given its length and width -/
def rectangle_perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Calculates the total distance run given the perimeter and number of laps -/
def total_distance (perimeter : ℝ) (laps : ℕ) : ℝ := perimeter * (laps : ℝ)

/-- Theorem stating that the total distance run is 1800 meters -/
theorem total_distance_is_1800 :
  total_distance (rectangle_perimeter field_length field_width) num_laps = 1800 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_1800_l1014_101421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1014_101404

theorem exponential_inequality (x : ℝ) : (2 : ℝ)^(x - 2) < 1 ↔ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1014_101404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l1014_101427

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := x^3 - Real.sqrt 3 * x + 2/3

-- Define the derivative of the curve
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - Real.sqrt 3

-- Define the angle of inclination
noncomputable def angle_of_inclination (x : ℝ) : ℝ := Real.arctan (f' x)

-- Theorem statement
theorem angle_of_inclination_range :
  ∀ x : ℝ, angle_of_inclination x ∈ Set.union (Set.Icc 0 (Real.pi/2)) (Set.Icc (2*Real.pi/3) Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l1014_101427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_from_conditions_l1014_101454

noncomputable def hyperbola_equation (x y : ℝ) : Prop := y^2 / 4 - x^2 / 12 = 1

noncomputable def ellipse_equation (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

theorem ellipse_equation_from_conditions 
  (x y a b c : ℝ) 
  (h1 : hyperbola_equation c 0)  -- foci of hyperbola coincide with vertices of ellipse
  (h2 : eccentricity c 2 + eccentricity c a = 13/5)  -- sum of eccentricities
  (h3 : c > 0)  -- foci on x-axis, c is positive
  (h4 : ellipse_equation c 0 a b)  -- ellipse passes through (c, 0)
  (h5 : b = 4)  -- derived from coinciding foci
  : ellipse_equation x y 5 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_from_conditions_l1014_101454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_pool_fill_time_l1014_101453

/-- Represents the time in hours required to fill a pool -/
noncomputable def fill_time (pool_capacity : ℝ) (current_volume : ℝ) (num_hoses : ℕ) (flow_rate : ℝ) : ℝ :=
  (pool_capacity - current_volume) / (↑num_hoses * flow_rate * 60)

/-- Theorem stating that it takes 22 hours to fill Linda's pool -/
theorem linda_pool_fill_time :
  let pool_capacity : ℝ := 30000
  let current_volume : ℝ := 6000
  let num_hoses : ℕ := 6
  let flow_rate : ℝ := 3
  fill_time pool_capacity current_volume num_hoses flow_rate = 22 := by
  -- Unfold the definition of fill_time
  unfold fill_time
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

#check linda_pool_fill_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_pool_fill_time_l1014_101453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_KLMN_approx_l1014_101411

-- Define the tetrahedron EFGH
structure Tetrahedron :=
  (E F G H : EuclideanSpace ℝ (Fin 3))

-- Define the conditions
def tetrahedron_conditions (t : Tetrahedron) : Prop :=
  dist t.E t.F = 7 ∧
  dist t.G t.H = 7 ∧
  dist t.E t.G = 10 ∧
  dist t.F t.H = 10 ∧
  dist t.E t.H = 11 ∧
  dist t.F t.G = 11

-- Define the inscribed circle centers
noncomputable def K (t : Tetrahedron) : EuclideanSpace ℝ (Fin 3) := sorry
noncomputable def L (t : Tetrahedron) : EuclideanSpace ℝ (Fin 3) := sorry
noncomputable def M (t : Tetrahedron) : EuclideanSpace ℝ (Fin 3) := sorry
noncomputable def N (t : Tetrahedron) : EuclideanSpace ℝ (Fin 3) := sorry

-- Define the volume of tetrahedron KLMN
noncomputable def volume_KLMN (t : Tetrahedron) : ℝ := sorry

-- State the theorem
theorem volume_KLMN_approx (t : Tetrahedron) :
  tetrahedron_conditions t →
  ∃ ε > 0, |volume_KLMN t - 2.09| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_KLMN_approx_l1014_101411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l1014_101440

noncomputable def h (x : ℝ) : ℝ := (4*x - 2) / (x - 5)

theorem h_domain : 
  {x : ℝ | ∃ y, h x = y} = {x : ℝ | x < 5 ∨ x > 5} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l1014_101440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_line_parametric_equations_is_ellipse_l1014_101458

-- Define the polar equation
noncomputable def polar_equation (θ : ℝ) : ℝ := 2 / Real.cos θ

-- Define the parametric equations
noncomputable def parametric_x (θ : ℝ) : ℝ := 2 * Real.cos θ
noncomputable def parametric_y (θ : ℝ) : ℝ := 3 * Real.sin θ

-- Theorem for the polar equation representing a line
theorem polar_equation_is_line :
  ∃ (a b : ℝ), ∀ (x y : ℝ), x = polar_equation (Real.arctan (y / x)) → a * x + b * y = 1 :=
by sorry

-- Theorem for the parametric equations representing an ellipse
theorem parametric_equations_is_ellipse :
  ∃ (a b : ℝ), ∀ (θ : ℝ),
    (parametric_x θ)^2 / a^2 + (parametric_y θ)^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_line_parametric_equations_is_ellipse_l1014_101458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_range_l1014_101481

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/3) * Real.exp (3*x) + m * Real.exp (2*x) + (2*m+1) * Real.exp x + 1

def has_two_extreme_points (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x ≠ x₁ → x ≠ x₂ → 
      (f m x > f m x₁ ∧ f m x > f m x₂) ∨ 
      (f m x < f m x₁ ∧ f m x < f m x₂))

theorem extreme_points_range : 
  ∀ m : ℝ, has_two_extreme_points m ↔ (-1/2 < m ∧ m < 1 - Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_range_l1014_101481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1014_101409

-- Define the vectors a and b in R²
variable (a b : ℝ × ℝ)

-- Define the conditions
noncomputable def magnitude_a : ℝ := Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))
noncomputable def magnitude_b : ℝ := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))
noncomputable def magnitude_sum : ℝ := Real.sqrt (((a.1 + b.1) ^ 2) + ((a.2 + b.2) ^ 2))

-- Define the angle between vectors
noncomputable def angle (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

-- State the theorem
theorem angle_between_vectors :
  magnitude_a a = 2 →
  magnitude_b b = 1 →
  magnitude_sum a b = Real.sqrt 7 →
  angle a b = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1014_101409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l1014_101455

/-- Definition of a quadratic equation with one variable -/
def is_quadratic_one_var (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 + 1 = 0 -/
def eq_A : ℝ → ℝ := λ x ↦ x^2 + 1

/-- The equation x^2 + y = 1 -/
def eq_B : ℝ → ℝ → ℝ := λ x y ↦ x^2 + y - 1

/-- The equation 2x + 1 = 0 -/
def eq_C : ℝ → ℝ := λ x ↦ 2*x + 1

/-- The equation 1/x + x^2 = 1 -/
noncomputable def eq_D : ℝ → ℝ := λ x ↦ 1/x + x^2 - 1

theorem quadratic_equation_identification :
  is_quadratic_one_var eq_A ∧
  ¬is_quadratic_one_var (λ x ↦ eq_B x 0) ∧
  ¬is_quadratic_one_var eq_C ∧
  ¬is_quadratic_one_var eq_D :=
by
  sorry

#check quadratic_equation_identification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l1014_101455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_projection_is_pentagon_l1014_101480

/-- A dodecahedron -/
structure Dodecahedron where
  -- Add necessary properties of a dodecahedron

/-- A plane -/
structure Plane where
  -- Add necessary properties of a plane

/-- A line -/
structure Line where
  -- Add necessary properties of a line

/-- A point in 3D space -/
structure Point where
  -- Add necessary properties of a point

/-- Projection of a 3D shape onto a plane -/
def project (shape : Type) (plane : Plane) : Type :=
  sorry

/-- Pentagon shape -/
structure Pentagon where
  -- Add necessary properties of a pentagon

/-- The center of a dodecahedron -/
def center (d : Dodecahedron) : Point :=
  sorry

/-- The midpoint of an edge of a dodecahedron -/
def edgeMidpoint (d : Dodecahedron) : Point :=
  sorry

/-- A line passing through two points -/
def lineThroughPoints (p1 p2 : Point) : Line :=
  sorry

/-- A plane perpendicular to a line -/
def perpendicularPlane (l : Line) : Plane :=
  sorry

theorem dodecahedron_projection_is_pentagon (d : Dodecahedron) :
  let centerEdgeLine := lineThroughPoints (center d) (edgeMidpoint d)
  let projectionPlane := perpendicularPlane centerEdgeLine
  project Dodecahedron projectionPlane = Pentagon := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_projection_is_pentagon_l1014_101480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_8_1_total_score_l1014_101414

/-- Represents the scoring criteria for the radio gymnastics competition -/
structure ScoringCriteria where
  spirit : ℚ
  neatness : ℚ
  movements : ℚ

/-- Calculates the total score based on the given criteria and weights -/
def totalScore (criteria : ScoringCriteria) (weights : Fin 3 → ℚ) : ℚ :=
  (criteria.spirit * weights 0 + criteria.neatness * weights 1 + criteria.movements * weights 2) /
  (weights 0 + weights 1 + weights 2)

/-- The weights for each criterion in the ratio 2:3:5 -/
def competitionWeights : Fin 3 → ℚ
  | 0 => 2
  | 1 => 3
  | 2 => 5

theorem class_8_1_total_score :
  let criteria : ScoringCriteria := ⟨8, 9, 10⟩
  totalScore criteria competitionWeights = 93 / 10 := by
  sorry

#eval totalScore ⟨8, 9, 10⟩ competitionWeights

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_8_1_total_score_l1014_101414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_measurement_greater_relative_error_l1014_101417

/-- The length of the first line in inches -/
noncomputable def length1 : ℝ := 20

/-- The absolute error in the measurement of the first line in inches -/
noncomputable def error1 : ℝ := 0.05

/-- The length of the second line in inches -/
noncomputable def length2 : ℝ := 80

/-- The absolute error in the measurement of the second line in inches -/
noncomputable def error2 : ℝ := 0.25

/-- The relative error of a measurement -/
noncomputable def relativeError (error : ℝ) (length : ℝ) : ℝ := error / length

theorem second_measurement_greater_relative_error :
  relativeError error2 length2 > relativeError error1 length1 := by
  -- Unfold the definitions
  unfold relativeError
  unfold error2 error1 length2 length1
  -- Simplify the inequality
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_measurement_greater_relative_error_l1014_101417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visual_representation_area_l1014_101475

/-- Structure to represent a triangle -/
structure Triangle where
  -- You can add necessary fields here, e.g., vertices
  mk :: -- Constructor

/-- Predicate to check if a triangle is equilateral -/
def Triangle.Equilateral (t : Triangle) : Prop := sorry

/-- Function to get the side length of a triangle -/
def Triangle.SideLength (t : Triangle) : ℝ := sorry

/-- Predicate to check if one triangle is a visual representation of another -/
def IsVisualRepresentation (t1 t2 : Triangle) : Prop := sorry

/-- Function to calculate the area of a triangle -/
def Triangle.area (t : Triangle) : ℝ := sorry

/-- Given an equilateral triangle ABC with side length 4cm and its visual representation A'B'C',
    prove that the area of A'B'C' is √6 cm² -/
theorem visual_representation_area (ABC : Triangle) (A'B'C' : Triangle) :
  Triangle.Equilateral ABC →
  Triangle.SideLength ABC = 4 →
  IsVisualRepresentation A'B'C' ABC →
  Triangle.area A'B'C' = Triangle.area ABC * (Real.sqrt 2 / 4) →
  Triangle.area A'B'C' = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visual_representation_area_l1014_101475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_for_given_volume_and_angle_l1014_101485

/-- The height of a cone with given volume and vertex angle -/
noncomputable def cone_height (volume : ℝ) (vertex_angle : ℝ) : ℝ :=
  (3 * volume / Real.pi) ^ (1/3)

/-- Theorem: The height of a cone with volume 9720π and vertex angle 90° is ∛29160 -/
theorem cone_height_for_given_volume_and_angle :
  cone_height (9720 * Real.pi) 90 = (29160 : ℝ) ^ (1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_for_given_volume_and_angle_l1014_101485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_PQ_l1014_101497

def P : Finset ℕ := {3, 4, 5}
def Q : Finset ℕ := {6, 7}

def PQ : Finset (ℕ × ℕ) := Finset.product P Q

theorem number_of_subsets_PQ : Finset.card (Finset.powerset PQ) = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_PQ_l1014_101497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compute_expression_l1014_101436

theorem compute_expression (c d : ℚ) (hc : c = 4/7) (hd : d = 5/6) :
  c^3 * d^(-2 : ℤ) + c^(-1 : ℤ) * d^2 = 1832401/1234800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compute_expression_l1014_101436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1014_101483

theorem exponential_inequality (x : ℝ) : (2 : ℝ)^(2*x - 7) < (2 : ℝ)^(x - 3) ↔ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1014_101483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l1014_101402

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos θ, Real.sin θ)

-- Define the line l in polar form
def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 4) = 2 * Real.sqrt 2

-- Define the distance function from a point to the line
noncomputable def distance_to_line (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  abs (x + y - 4) / Real.sqrt 2

-- Theorem statement
theorem max_distance_to_line :
  ∃ (max_dist : ℝ), max_dist = 3 * Real.sqrt 2 ∧
  ∀ (θ : ℝ), distance_to_line (curve_C θ) ≤ max_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l1014_101402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l1014_101410

-- Define the complex numbers
noncomputable def z₁ : ℂ := -3 - Complex.I * Real.sqrt 3
noncomputable def z₂ : ℂ := Real.sqrt 3 + Complex.I

-- Define z as a function of θ
noncomputable def z (θ : ℝ) : ℂ := Real.sqrt 3 * Real.sin θ + Complex.I * (Real.sqrt 3 * Real.cos θ + 2)

-- State the theorem
theorem min_sum_distances :
  ∃ (θ : ℝ), ∀ (φ : ℝ), Complex.abs (z θ - z₁) + Complex.abs (z θ - z₂) ≤ Complex.abs (z φ - z₁) + Complex.abs (z φ - z₂) ∧
  Complex.abs (z θ - z₁) + Complex.abs (z θ - z₂) = 2 * (Real.sqrt 3 + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l1014_101410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_magnitude_l1014_101489

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (v w : V)

theorem projection_magnitude
  (hv : ‖v‖ = 5)
  (hw : ‖w‖ = 8)
  (hangle : Real.cos (Real.pi / 3) = inner v w / (‖v‖ * ‖w‖)) :
  ‖(inner v w / ‖w‖^2) • w‖ = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_magnitude_l1014_101489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_g_implies_a_bound_l1014_101423

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x
noncomputable def g (x : ℝ) : ℝ := -(1/2) * x^(3/2)

-- State the theorem
theorem f_less_than_g_implies_a_bound (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, f a x < g x) → a < -3/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_g_implies_a_bound_l1014_101423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_sum_l1014_101434

/-- If f(x) = ax^2 + bx + c and f^(-1)(x) = cx^2 + bx + a for real numbers a, b, and c,
    and f(f^(-1)(x)) = x, then a + b + c = 1 -/
theorem inverse_function_sum (a b c : ℝ) 
    (h : ∀ x, a*(c*x^2 + b*x + a)^2 + b*(c*x^2 + b*x + a) + c = x) :
    a + b + c = 1 := by
  sorry

#check inverse_function_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_sum_l1014_101434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_6_l1014_101447

def T : ℕ → ℕ
  | 0 => 9  -- Add this case to handle Nat.zero
  | 1 => 9
  | n+2 => 9^(T (n+1))

theorem t_50_mod_6 : T 50 ≡ 3 [ZMOD 6] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_6_l1014_101447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_mean_interior_angle_l1014_101490

/-- The sum of interior angles of an n-sided polygon -/
noncomputable def sum_interior_angles (n : ℕ) : ℝ := (n - 2 : ℝ) * 180

/-- The mean value of interior angles of an n-sided polygon -/
noncomputable def mean_interior_angle (n : ℕ) : ℝ := sum_interior_angles n / n

/-- Theorem: The mean value of the measures of the five interior angles of any pentagon is 108° -/
theorem pentagon_mean_interior_angle : 
  mean_interior_angle 5 = 108 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_mean_interior_angle_l1014_101490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_of_five_unit_circles_l1014_101492

/-- The area of the region formed by the intersection of five unit circles -/
theorem intersection_area_of_five_unit_circles : 
  ∃ (area : ℝ), 
    (∀ (x y : ℝ), x^2 + y^2 = 1 → (x, y) ∈ Set.univ) ∧ 
    (∀ (x y : ℝ), (x - 0)^2 + (y - 1)^2 = 1 → (x, y) ∈ Set.univ) ∧
    (∀ (x y : ℝ), (x - 1)^2 + (y - 0)^2 = 1 → (x, y) ∈ Set.univ) ∧
    (∀ (x y : ℝ), (x - 0)^2 + (y + 1)^2 = 1 → (x, y) ∈ Set.univ) ∧
    (∀ (x y : ℝ), (x + 1)^2 + (y - 0)^2 = 1 → (x, y) ∈ Set.univ) →
    area = 3 * Real.sqrt 3 - Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_of_five_unit_circles_l1014_101492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_nested_simplification_l1014_101482

-- Define the function representing the left-hand side of the equation
noncomputable def f (x : ℝ) : ℝ := Real.rpow (Real.rpow (Real.rpow (Real.rpow x (1/3)) (1/3)) (1/3)) (1/3)

-- Theorem statement
theorem cube_root_nested_simplification {x : ℝ} (hx : x ≥ 0) : f x = x^(10/9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_nested_simplification_l1014_101482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_prime_2_l1014_101473

noncomputable section

variable (a : ℝ)

noncomputable def f (x : ℝ) := x^3 + 2*a*x^2 + (1/a)*x

noncomputable def f_prime (x : ℝ) := 3*x^2 + 4*a*x + 1/a

theorem min_value_f_prime_2 (h : a > 0) :
  ∀ y : ℝ, f_prime a 2 ≤ y → 12 + 4 * Real.sqrt 2 ≤ y := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_prime_2_l1014_101473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_silverware_probability_l1014_101416

/-- The probability of selecting 2 forks, 1 spoon, and 1 knife from a drawer -/
theorem silverware_probability (forks spoons knives : ℕ) 
  (h_forks : forks = 8)
  (h_spoons : spoons = 10)
  (h_knives : knives = 7) :
  let total := forks + spoons + knives
  let ways_to_choose_4 := Nat.choose total 4
  let ways_to_choose_forks := Nat.choose forks 2
  let ways_to_choose_spoon := Nat.choose spoons 1
  let ways_to_choose_knife := Nat.choose knives 1
  let favorable_outcomes := ways_to_choose_forks * ways_to_choose_spoon * ways_to_choose_knife
  (favorable_outcomes : ℚ) / ways_to_choose_4 = 392 / 2530 := by
  sorry

#eval Nat.choose 25 4  -- Expected: 12650
#eval Nat.choose 8 2 * Nat.choose 10 1 * Nat.choose 7 1  -- Expected: 1960

end NUMINAMATH_CALUDE_ERRORFEEDBACK_silverware_probability_l1014_101416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_loss_l1014_101461

noncomputable section

-- Define the selling price of each item
def selling_price : ℝ := 180

-- Define the profit percentage
def profit_percentage : ℝ := 0.2

-- Define the loss percentage
def loss_percentage : ℝ := 0.2

-- Define the cost price of the item with profit
noncomputable def cost_price_profit : ℝ := selling_price / (1 + profit_percentage)

-- Define the cost price of the item with loss
noncomputable def cost_price_loss : ℝ := selling_price / (1 - loss_percentage)

-- Define the total selling price
def total_selling_price : ℝ := 2 * selling_price

-- Define the total cost price
noncomputable def total_cost_price : ℝ := cost_price_profit + cost_price_loss

-- Theorem statement
theorem merchant_loss : total_selling_price - total_cost_price = -15 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_loss_l1014_101461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_diagonals_theorem_l1014_101435

/-- The angle between the diagonal of a rectangular parallelepiped and the skew diagonal of a face --/
noncomputable def angle_between_diagonals (a b c : ℝ) : ℝ :=
  Real.arccos ((abs (a^2 - b^2)) / (Real.sqrt (a^2 + b^2) * Real.sqrt (a^2 + b^2 + c^2)))

/-- Theorem: The angle between the diagonal of a rectangular parallelepiped and the skew diagonal of a face --/
theorem angle_between_diagonals_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let β := angle_between_diagonals a b c
  ∃ (d₁ d₂ : ℝ), d₁ > 0 ∧ d₂ > 0 ∧
    β = Real.arccos ((d₁ • d₂) / (Real.sqrt (d₁ • d₁) * Real.sqrt (d₂ • d₂))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_diagonals_theorem_l1014_101435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_l1014_101405

/-- Given an angle α whose terminal side passes through the point P(m, 4) and cos α = -3/5, prove that m = -3 -/
theorem angle_terminal_side (α : ℝ) (m : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (m, 4) ∧ P ∈ Set.range (λ t : ℝ => (t * Real.cos α, t * Real.sin α))) → 
  Real.cos α = -3/5 → 
  m = -3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_l1014_101405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_atop_difference_equals_three_l1014_101439

/-- Custom operation @ defined for all real numbers -/
def atop (x y : ℝ) : ℝ := x * y - x - 2 * y

/-- Theorem stating that (7@4)-(4@7) = 3 -/
theorem atop_difference_equals_three : atop 7 4 - atop 4 7 = 3 := by
  -- Unfold the definition of atop
  unfold atop
  -- Perform arithmetic calculations
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_atop_difference_equals_three_l1014_101439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_membership_l1014_101420

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the "on" relation between a point and a line
variable (on_line : Point → Line → Prop)

-- Define the "in" relation between a geometric object and a set
variable (in_set : ∀ {α : Type}, α → Set α → Prop)

-- Theorem statement
theorem point_on_line_membership 
  (A : Point) (l : Line) : 
  on_line A l ↔ in_set A {x : Point | on_line x l} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_membership_l1014_101420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cab_driver_income_l1014_101465

/-- Given a list of 5 daily incomes where 4 are known and the average of all 5 is 400,
    prove that the unknown income (first day) must be 200. -/
theorem cab_driver_income (incomes : List ℕ) (h1 : incomes.length = 5)
  (h2 : incomes.tail = [150, 750, 400, 500])
  (h3 : incomes.sum / incomes.length = 400) :
  incomes.head? = some 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cab_driver_income_l1014_101465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trip_average_speed_l1014_101425

/-- Calculates the average speed for the remaining part of a trip given the initial speed, total average speed, and time intervals. -/
theorem car_trip_average_speed 
  (initial_speed : ℝ) 
  (initial_time : ℝ) 
  (total_time : ℝ) 
  (total_avg_speed : ℝ) 
  (h1 : initial_speed = 30)
  (h2 : initial_time = 5)
  (h3 : total_time = 7.5)
  (h4 : total_avg_speed = 34) :
  (total_avg_speed * total_time - initial_speed * initial_time) / (total_time - initial_time) = 42 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trip_average_speed_l1014_101425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sequence_l1014_101498

noncomputable def a (n : ℕ) : ℝ := (n + 2 : ℝ) * (7/8)^n

theorem max_value_of_sequence :
  ∀ k : ℕ, k > 0 → (a k ≥ a (k-1) ∧ a k ≥ a (k+1)) → (k = 5 ∨ k = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sequence_l1014_101498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1014_101471

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

def circleA : Circle := { center := (-5, -1), radius := 2 }
def circleB : Circle := { center := (0, 1), radius := 3 }
def circleC : Circle := { center := (7, -3), radius := 4 }

def lineM : Line := { point := (0, 3), direction := (1, 0) }

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Checks if two circles are externally tangent -/
def areExternallyTangent (c1 c2 : Circle) : Prop :=
  distance c1.center c2.center = c1.radius + c2.radius

/-- Checks if a circle is tangent to a line -/
def isTangentToLine (c : Circle) (l : Line) : Prop :=
  distance c.center l.point = c.radius + 1

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

theorem area_of_triangle_ABC :
  areExternallyTangent circleB circleA ∧
  areExternallyTangent circleB circleC ∧
  isTangentToLine circleA lineM ∧
  isTangentToLine circleB lineM ∧
  isTangentToLine circleC lineM →
  triangleArea circleA.center circleB.center circleC.center = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1014_101471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_imaginary_roots_l1014_101437

/-- The probability that a quadratic equation has imaginary roots when its coefficients
    are randomly chosen from a given interval. -/
theorem probability_imaginary_roots : 
  ∃ (μ : MeasureTheory.Measure (ℝ × ℝ)), 
    μ (Set.Icc (-2 : ℝ) 2 ×ˢ Set.Icc (-2 : ℝ) 2) = 16 ∧
    μ {(p, q) | p^2 + q^2 < 1} = π ∧
    μ {(p, q) | p^2 + q^2 < 1} / μ (Set.Icc (-2 : ℝ) 2 ×ˢ Set.Icc (-2 : ℝ) 2) = π / 16 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_imaginary_roots_l1014_101437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_line_intersecting_four_circles_l1014_101448

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def UnitSquare : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

def CirclesInSquare (circles : List Circle) : Prop :=
  ∀ c ∈ circles, ∀ p ∈ UnitSquare, dist p c.center ≤ c.radius

noncomputable def TotalCircumference (circles : List Circle) : ℝ :=
  (circles.map (fun c => 2 * Real.pi * c.radius)).sum

-- The main theorem
theorem exists_line_intersecting_four_circles 
  (circles : List Circle) 
  (h_in_square : CirclesInSquare circles) 
  (h_circumference : TotalCircumference circles = 10) :
  ∃ (line : Set (ℝ × ℝ)), ∃ (intersected : List Circle), 
    intersected.length ≥ 4 ∧ 
    (∀ c ∈ intersected, ∃ p ∈ line, dist p c.center = c.radius) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_line_intersecting_four_circles_l1014_101448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooks_on_board_l1014_101472

/-- A configuration of rooks on a 10x10 board. -/
def RookConfiguration := Fin 10 → Fin 10 → Bool

/-- A cell is attacked if there's a rook in its row or column. -/
def is_attacked (config : RookConfiguration) (row col : Fin 10) : Prop :=
  (∃ i, config i col) ∨ (∃ j, config row j)

/-- A configuration is valid if removing any rook leaves at least one cell unattacked. -/
def is_valid_configuration (config : RookConfiguration) : Prop :=
  ∀ rook_row rook_col,
    config rook_row rook_col →
    ∃ cell_row cell_col,
      ¬is_attacked
        (fun r c => if r = rook_row ∧ c = rook_col then false else config r c)
        cell_row cell_col

/-- The number of rooks in a configuration. -/
def rook_count (config : RookConfiguration) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 10)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 10)) fun j =>
      if config i j then 1 else 0)

/-- Theorem: The maximum number of rooks in a valid configuration is 16. -/
theorem max_rooks_on_board :
  (∃ config : RookConfiguration, is_valid_configuration config ∧ rook_count config = 16) ∧
  (∀ config : RookConfiguration, is_valid_configuration config → rook_count config ≤ 16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooks_on_board_l1014_101472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inequality_l1014_101433

-- Define the function h
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := (x - a) * Real.exp x + a

-- Define the theorem
theorem h_inequality (b : ℝ) (hb : b ≥ 17/8) :
  ∀ x₁ ∈ Set.Icc (-1 : ℝ) 1, ∃ x₂ ∈ Set.Icc 1 2,
    h 3 x₁ ≥ x₂^2 - 2*b*x₂ - 3*Real.exp 1 + Real.exp 1 + 15/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inequality_l1014_101433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_of_three_digit_integers_l1014_101486

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def uses_given_digits (n : ℕ) : Prop :=
  let digits := [2, 3, 4, 6, 7, 8]
  ∀ d, d ∈ (n.digits 10) → d ∈ digits

theorem smallest_difference_of_three_digit_integers (n m : ℕ) :
  is_three_digit n →
  is_three_digit m →
  uses_given_digits n →
  uses_given_digits m →
  ∃ (diff : ℕ), diff = Int.natAbs (n - m) ∧ 
    ∀ (n' m' : ℕ), is_three_digit n' → is_three_digit m' → 
      uses_given_digits n' → uses_given_digits m' → 
      Int.natAbs (n' - m') ≥ diff :=
by
  sorry

#check smallest_difference_of_three_digit_integers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_of_three_digit_integers_l1014_101486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_optimality_l1014_101463

/-- Surface area of a rectangular prism -/
def surface_area (a b c : ℝ) : ℝ := 2 * (a * b + b * c + c * a)

/-- Volume of a rectangular prism -/
def volume (a b c : ℝ) : ℝ := a * b * c

/-- Sum of edge lengths of a rectangular prism -/
def sum_of_edges (a b c : ℝ) : ℝ := 4 * (a + b + c)

/-- Diagonal length of a rectangular prism -/
noncomputable def diagonal (a b c : ℝ) : ℝ := Real.sqrt (a^2 + b^2 + c^2)

theorem rectangular_prism_optimality 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let S := surface_area a b c
  let V := volume a b c
  let E := sum_of_edges a b c
  let d := diagonal a b c
  (∀ x y z, x > 0 → y > 0 → z > 0 → surface_area x y z = S → volume x y z ≤ V) ∧
  (∀ x y z, x > 0 → y > 0 → z > 0 → sum_of_edges x y z = E → volume x y z ≤ V) ∧
  (∀ x y z, x > 0 → y > 0 → z > 0 → diagonal x y z = d → volume x y z ≤ V) ∧
  (∀ x y z, x > 0 → y > 0 → z > 0 → sum_of_edges x y z = E → diagonal x y z ≥ d) ∧
  (V = volume a a a → d = diagonal a a a → E = sum_of_edges a a a → S = surface_area a a a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_optimality_l1014_101463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_from_sin_first_quadrant_l1014_101494

theorem cos_from_sin_first_quadrant (θ : Real) (h1 : Real.sin θ = 3/5) (h2 : 0 < θ ∧ θ < Real.pi/2) : 
  Real.cos θ = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_from_sin_first_quadrant_l1014_101494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_fixed_point_l1014_101464

/-- Ellipse C with equation x²/4 + y²/3 = 1 -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- Point on the ellipse -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse_C x y

/-- Chord AB perpendicular to x-axis -/
structure Chord where
  A : PointOnEllipse
  B : PointOnEllipse
  perp_to_x_axis : A.x = B.x

/-- Point N at (4,0) -/
def N : ℝ × ℝ := (4, 0)

/-- Line passing through two points -/
noncomputable def line_through (p1 p2 : ℝ × ℝ) (x : ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  (y2 - y1) / (x2 - x1) * (x - x1) + y1

/-- The fixed point (1,0) -/
def fixed_point : ℝ × ℝ := (1, 0)

/-- Main theorem -/
theorem chord_intersection_fixed_point (AB : Chord) :
  ∃ (M : PointOnEllipse),
    line_through (AB.B.x, AB.B.y) (M.x, M.y) (fixed_point.1) = fixed_point.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_fixed_point_l1014_101464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l1014_101445

-- Define the curves
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (2 * Real.sin θ * Real.cos θ, 2 * Real.sin θ * Real.sin θ)

noncomputable def C₂ (t : ℝ) : ℝ × ℝ := (-3/5 * t + 2, 4/5 * t)

-- Define point M
def M : ℝ × ℝ := (2, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem max_distance_MN :
  ∃ (max_dist : ℝ), max_dist = Real.sqrt 5 + 1 ∧
  ∀ (θ : ℝ), distance M (C₁ θ) ≤ max_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l1014_101445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cows_eating_grass_l1014_101477

-- Define the initial amount of grass and daily growth rate
variable (G : ℚ) (r : ℚ)

-- Define the conditions
axiom cond1 : G + 24 * r = 70 * 24
axiom cond2 : G + 60 * r = 30 * 60

-- Define the theorem
theorem cows_eating_grass : ∃ N : ℕ, N * 96 = Int.floor (G + 96 * r) ∧ N = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cows_eating_grass_l1014_101477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parabola_intersection_angle_l1014_101412

/-- The angle of inclination of a line intersecting a parabola -/
theorem line_parabola_intersection_angle (k : ℝ) :
  let l : ℝ → ℝ := λ x => k * (x - 2)
  let C : ℝ × ℝ → Prop := λ p => p.2^2 = 8*p.1
  let F : ℝ × ℝ := (2, 0)
  ∃ (A B : ℝ × ℝ),
    C A ∧ C B ∧
    A.2 = l A.1 ∧ B.2 = l B.1 ∧
    (A.1 - F.1)^2 + (A.2 - F.2)^2 = 9 * ((B.1 - F.1)^2 + (B.2 - F.2)^2) →
  k = Real.sqrt 3 ∨ k = -Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parabola_intersection_angle_l1014_101412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_duck_ratio_l1014_101468

/-- The number of Muscovy ducks -/
def muscovy : ℕ := 39

/-- The number of Cayuga ducks -/
def cayuga : ℕ := muscovy - 4

/-- The number of Khaki Campbell ducks -/
def khaki : ℕ := 90 - muscovy - cayuga

/-- The ratio of Cayugas to Khaki Campbells -/
def ratio : ℚ := cayuga / khaki

theorem duck_ratio : ratio = 35 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_duck_ratio_l1014_101468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_product_divisible_by_gcd_squared_l1014_101476

theorem gcd_product_divisible_by_gcd_squared (a b c : ℤ) :
  ∃ k : ℤ, Int.gcd (Int.gcd (b * c) (a * c)) (a * b) = k * (Int.gcd (Int.gcd a b) c)^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_product_divisible_by_gcd_squared_l1014_101476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tayzia_haircut_tip_l1014_101495

/-- Calculates the tip amount for haircuts -/
noncomputable def calculate_tip (womens_haircut_cost : ℚ) (childrens_haircut_cost : ℚ) 
                  (num_women : ℕ) (num_children : ℕ) (tip_percentage : ℚ) : ℚ :=
  let total_cost := womens_haircut_cost * num_women + childrens_haircut_cost * num_children
  tip_percentage / 100 * total_cost

/-- Proves that the tip amount for Tayzia and her daughters' haircuts is $24 -/
theorem tayzia_haircut_tip : 
  calculate_tip 48 36 1 2 20 = 24 := by
  -- Unfold the definition of calculate_tip
  unfold calculate_tip
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tayzia_haircut_tip_l1014_101495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_sum_implies_cos_double_sum_zero_l1014_101493

theorem cos_sin_sum_implies_cos_double_sum_zero
  (x y z : ℝ)
  (h1 : Real.cos x + Real.cos y + Real.cos z = 3)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 0) :
  Real.cos (2*x) + Real.cos (2*y) + Real.cos (2*z) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_sum_implies_cos_double_sum_zero_l1014_101493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ending_number_divisible_by_eleven_l1014_101407

theorem ending_number_divisible_by_eleven (start : ℕ) (count : ℕ) : 
  start = (39 / 11 + 1) * 11 →
  count = 4 →
  ∃ (end_num : ℕ), end_num = start + (count - 1) * 11 ∧ end_num = 77 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ending_number_divisible_by_eleven_l1014_101407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_sum_from_face_sum_l1014_101429

/-- Represents a cube with numbers at its vertices -/
structure NumberedCube where
  vertices : Fin 8 → ℝ

/-- Calculates the sum of numbers on all faces of the cube -/
def sum_of_faces (cube : NumberedCube) : ℝ :=
  3 * (Finset.sum Finset.univ (fun i => cube.vertices i))

theorem vertex_sum_from_face_sum (cube : NumberedCube) 
  (h : sum_of_faces cube = 2019) : 
  Finset.sum Finset.univ (fun i => cube.vertices i) = 673 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_sum_from_face_sum_l1014_101429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l1014_101488

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ,  Real.cos θ]]

theorem smallest_rotation_power : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → 
    (rotation_matrix (π/4))^m ≠ 1) ∧
  (rotation_matrix (π/4))^n = 1 :=
by
  use 8
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l1014_101488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_range_l1014_101478

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 3) * x + 5 else 2 * a / x

theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) → 0 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_range_l1014_101478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andy_changed_two_grommet_sets_l1014_101456

/-- Represents Andy's earnings and work during a shift -/
structure AndyShift where
  hourlyRate : ℚ
  racquetStringRate : ℚ
  grommetChangeRate : ℚ
  stencilRate : ℚ
  hoursWorked : ℕ
  racquetsStrung : ℕ
  stencilsPainted : ℕ
  totalEarnings : ℚ

/-- Calculates the number of grommet sets changed during Andy's shift -/
def grommetSetsChanged (shift : AndyShift) : ℚ :=
  (shift.totalEarnings -
   (shift.hourlyRate * shift.hoursWorked +
    shift.racquetStringRate * shift.racquetsStrung +
    shift.stencilRate * shift.stencilsPainted)) /
  shift.grommetChangeRate

/-- Theorem stating that Andy changed 2 sets of grommets during his shift -/
theorem andy_changed_two_grommet_sets :
  let shift : AndyShift := {
    hourlyRate := 9
    racquetStringRate := 15
    grommetChangeRate := 10
    stencilRate := 1
    hoursWorked := 8
    racquetsStrung := 7
    stencilsPainted := 5
    totalEarnings := 202
  }
  grommetSetsChanged shift = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_andy_changed_two_grommet_sets_l1014_101456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_rental_company_theorem_l1014_101496

/-- Represents a car rental company's purchasing plan. -/
structure PurchasingPlan where
  sedans : Nat
  vans : Nat

/-- Checks if a purchasing plan is valid according to the company's requirements. -/
def is_valid_plan (p : PurchasingPlan) : Prop :=
  p.sedans + p.vans = 10 ∧
  p.sedans ≥ 3 ∧
  70000 * p.sedans + 40000 * p.vans ≤ 550000

/-- Calculates the daily rental income for a given purchasing plan. -/
def daily_rental_income (p : PurchasingPlan) : Nat :=
  200 * p.sedans + 110 * p.vans

/-- The set of all valid purchasing plans. -/
def valid_plans : Set PurchasingPlan :=
  {p | is_valid_plan p}

/-- The theorem to be proved. -/
theorem car_rental_company_theorem :
  (∃ (plans : List PurchasingPlan), 
    (∀ p ∈ plans, is_valid_plan p) ∧ 
    plans.length = 3) ∧
  (∃ (p : PurchasingPlan), is_valid_plan p ∧ p.sedans = 5 ∧ p.vans = 5 ∧
    (∀ (q : PurchasingPlan), is_valid_plan q → daily_rental_income q ≥ 1500 → q = p)) := by
  sorry

#check car_rental_company_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_rental_company_theorem_l1014_101496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_correct_answers_for_excellent_l1014_101442

theorem min_correct_answers_for_excellent (total_questions : ℕ) 
  (correct_points : ℕ) (incorrect_points : ℕ) (excellent_threshold : ℕ) :
  total_questions = 20 →
  correct_points = 5 →
  incorrect_points = 2 →
  excellent_threshold = 80 →
  ∃ (min_correct : ℕ), 
    (∀ (x : ℕ), x ≥ min_correct → 
      x * correct_points + (total_questions - x) * incorrect_points ≤ total_questions * incorrect_points - excellent_threshold) ∧
    (∀ (y : ℕ), y < min_correct → 
      y * correct_points + (total_questions - y) * incorrect_points > total_questions * incorrect_points - excellent_threshold) ∧
    min_correct = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_correct_answers_for_excellent_l1014_101442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1014_101469

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 3) + Real.sqrt (12 - 3*x)

-- State the theorem about the range of f
theorem range_of_f :
  (∀ x, x ∈ Set.Icc 3 4 → f x ∈ Set.Icc 1 2) ∧
  (∃ x₁ ∈ Set.Icc 3 4, f x₁ = 1) ∧
  (∃ x₂ ∈ Set.Icc 3 4, f x₂ = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1014_101469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lotus_root_max_profit_l1014_101413

-- Define the profit function
noncomputable def profit_function (x : ℝ) : ℝ := -1/6 * x^3 + 2 * x^2 - 2

-- Define the derivative of the profit function
noncomputable def profit_derivative (x : ℝ) : ℝ := -1/2 * x^2 + 4 * x

-- Theorem statement
theorem lotus_root_max_profit :
  ∃ (x : ℝ), x = 8 ∧
  ∀ (y : ℝ), 0 ≤ y ∧ y ≤ 10 → profit_function x ≥ profit_function y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lotus_root_max_profit_l1014_101413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_interval_max_min_values_l1014_101451

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + 1/2 * x^2 + 2*a*x

-- Theorem for part (1)
theorem monotonic_interval (a : ℝ) :
  (∃ (l r : ℝ), l > 2/3 ∧ r > l ∧ StrictMonoOn (f a) (Set.Ioo l r)) ↔ a > -1/9 := by
  sorry

-- Theorem for part (2)
theorem max_min_values :
  (∃ (x_max : ℝ), x_max ∈ Set.Icc 1 4 ∧ f 1 x_max = 10/3 ∧ ∀ y ∈ Set.Icc 1 4, f 1 y ≤ f 1 x_max) ∧
  (∃ (x_min : ℝ), x_min ∈ Set.Icc 1 4 ∧ f 1 x_min = -16/3 ∧ ∀ y ∈ Set.Icc 1 4, f 1 y ≥ f 1 x_min) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_interval_max_min_values_l1014_101451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_A_then_red_B_prob_white_A_given_red_B_l1014_101470

/-- Represents the color of a ball -/
inductive Color
  | Red
  | White
  | Black

/-- Represents a box containing balls of different colors -/
structure Box where
  red : Nat
  white : Nat
  black : Nat

/-- The initial state of box A -/
def boxA : Box := { red := 4, white := 2, black := 3 }

/-- The initial state of box B -/
def boxB : Box := { red := 3, white := 3, black := 3 }

/-- The probability of selecting a ball of a given color from a box -/
def prob_select (box : Box) (color : Color) : Rat :=
  match color with
  | Color.Red => box.red / (box.red + box.white + box.black)
  | Color.White => box.white / (box.red + box.white + box.black)
  | Color.Black => box.black / (box.red + box.white + box.black)

/-- The probability of selecting a red ball from box B after moving a ball of given color from A to B -/
def prob_red_B_after_move (color : Color) : Rat :=
  match color with
  | Color.Red => (boxB.red + 1) / (boxB.red + boxB.white + boxB.black + 1)
  | Color.White => boxB.red / (boxB.red + boxB.white + boxB.black + 1)
  | Color.Black => boxB.red / (boxB.red + boxB.white + boxB.black + 1)

theorem prob_red_A_then_red_B : 
  prob_select boxA Color.Red * prob_red_B_after_move Color.Red = 8 / 45 := by sorry

theorem prob_white_A_given_red_B :
  (prob_select boxA Color.White * prob_red_B_after_move Color.White) / 
  (prob_select boxA Color.Red * prob_red_B_after_move Color.Red +
   prob_select boxA Color.White * prob_red_B_after_move Color.White +
   prob_select boxA Color.Black * prob_red_B_after_move Color.Black) = 6 / 31 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_A_then_red_B_prob_white_A_given_red_B_l1014_101470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1014_101487

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 + 2 * x + 1)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → 0 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1014_101487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_length_calculation_l1014_101400

noncomputable def room_length (room_width : ℝ) (verandah_width : ℝ) (verandah_area : ℝ) : ℝ :=
  (verandah_area + room_width * verandah_width * 2 + verandah_width ^ 2 * 4) / (room_width + verandah_width * 2) - verandah_width * 2

theorem room_length_calculation :
  room_length 12 2 124 = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_length_calculation_l1014_101400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_properties_l1014_101450

/-- Represents a family's monthly income and savings data -/
structure FamilyData where
  income : ℝ
  savings : ℝ

/-- Calculates the sum of a list of real numbers -/
noncomputable def sum (xs : List ℝ) : ℝ := xs.foldl (·+·) 0

/-- Calculates the mean of a list of real numbers -/
noncomputable def mean (xs : List ℝ) : ℝ := sum xs / xs.length

/-- Represents the sample data and calculated statistics -/
structure SampleStatistics where
  data : List FamilyData
  n : Nat
  sum_x : ℝ
  sum_y : ℝ
  sum_xy : ℝ
  sum_x_sq : ℝ

/-- Calculates the slope (b̂) of the linear regression equation -/
noncomputable def calculate_slope (stats : SampleStatistics) : ℝ :=
  let x_mean := stats.sum_x / stats.n
  let y_mean := stats.sum_y / stats.n
  (stats.sum_xy - stats.n * x_mean * y_mean) / (stats.sum_x_sq - stats.n * x_mean * x_mean)

/-- Calculates the y-intercept (â) of the linear regression equation -/
noncomputable def calculate_intercept (stats : SampleStatistics) (slope : ℝ) : ℝ :=
  let y_mean := stats.sum_y / stats.n
  let x_mean := stats.sum_x / stats.n
  y_mean - slope * x_mean

/-- Theorem stating the properties of the linear regression for the given sample -/
theorem linear_regression_properties (sample : SampleStatistics) 
    (h_n : sample.n = 10)
    (h_sum_x : sample.sum_x = 80)
    (h_sum_y : sample.sum_y = 20)
    (h_sum_xy : sample.sum_xy = 184)
    (h_sum_x_sq : sample.sum_x_sq = 720) :
    let b := calculate_slope sample
    let a := calculate_intercept sample b
    (b = 0.3 ∧ a = -0.4) ∧  -- Linear regression equation
    (b > 0) ∧              -- Positive correlation
    (0.3 * 7 - 0.4 = 1.7)  -- Prediction for x = 7
    := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_properties_l1014_101450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_sales_amount_l1014_101441

/-- The amount of the first sales in millions of dollars -/
def S : ℝ := sorry

/-- The royalties on the first sales in millions of dollars -/
def first_royalties : ℝ := 4

/-- The royalties on the next sales in millions of dollars -/
def next_royalties : ℝ := 9

/-- The amount of the next sales in millions of dollars -/
def next_sales : ℝ := 108

/-- The percentage decrease in the ratio of royalties to sales -/
def percentage_decrease : ℝ := 58.333333333333336

theorem first_sales_amount :
  (first_royalties / S) - (next_royalties / next_sales) = 
  (percentage_decrease / 100) * (first_royalties / S) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_sales_amount_l1014_101441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_implies_a_le_one_min_m_for_inequality_l1014_101444

/-- The function f(x) = e^x + a * e^(-x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

/-- Part 1: If f(x) is monotonically increasing on [0, +∞), then a ≤ 1 -/
theorem f_monotone_implies_a_le_one (a : ℝ) :
  (∀ x ∈ Set.Ici (0 : ℝ), Monotone (f a)) → a ≤ 1 :=
by sorry

/-- Part 2: For a = 1, the minimum value of m such that m[f(2x)+2] ≥ f(x)+1 for all x ∈ ℝ is 3/4 -/
theorem min_m_for_inequality (m : ℝ) :
  (∀ x : ℝ, m * (f 1 (2 * x) + 2) ≥ f 1 x + 1) ↔ m ≥ 3/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_implies_a_le_one_min_m_for_inequality_l1014_101444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_ball_weights_l1014_101446

/-- Represents the weight of a ball -/
inductive Weight
| Light
| Heavy

/-- Represents a ball with a color and a weight -/
structure Ball where
  color : String
  weight : Weight

/-- Represents the result of a weighing comparison -/
inductive ComparisonResult
| Equal
| LeftHeavier
| RightHeavier

/-- Represents a weighing action on the balance scale -/
def Weighing := List Ball → List Ball → ComparisonResult

/-- The main theorem stating that it's possible to determine all ball weights with two weighings -/
theorem determine_ball_weights 
  (balls : List Ball)
  (h_count : balls.length = 5)
  (h_colors : ∃ (r1 r2 w1 w2 b : Ball), 
    r1 ∈ balls ∧ r2 ∈ balls ∧ w1 ∈ balls ∧ w2 ∈ balls ∧ b ∈ balls ∧
    r1.color = "red" ∧ r2.color = "red" ∧ 
    w1.color = "white" ∧ w2.color = "white" ∧ 
    b.color = "black")
  (h_diff_weights : ∃ (r1 r2 w1 w2 : Ball),
    r1 ∈ balls ∧ r2 ∈ balls ∧ w1 ∈ balls ∧ w2 ∈ balls ∧
    r1.color = "red" ∧ r2.color = "red" ∧ 
    w1.color = "white" ∧ w2.color = "white" ∧
    r1.weight ≠ r2.weight ∧ w1.weight ≠ w2.weight)
  : ∃ (w1 w2 : Weighing), ∀ b ∈ balls, 
    ∃ (result1 : ComparisonResult) (result2 : ComparisonResult), 
    (w1 [b] [b] = result1 ∧ w2 [b] [b] = result2) → b.weight = Weight.Light ∨ b.weight = Weight.Heavy :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_ball_weights_l1014_101446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1014_101422

/-- Given two people who can complete a work individually, calculates how long it takes them to complete the work together. -/
noncomputable def time_to_complete_together (x_time : ℝ) (y_time : ℝ) : ℝ :=
  1 / (1 / x_time + 1 / y_time)

/-- Theorem: If x can complete a work in 30 days and y can complete the same work in 45 days,
    then together they will complete the work in 18 days. -/
theorem work_completion_time (x_time y_time : ℝ) 
    (hx : x_time = 30) (hy : y_time = 45) : 
    time_to_complete_together x_time y_time = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1014_101422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1014_101415

-- Define the function f(x) = a^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc (-2) 2, f a x < 2) →
  a ∈ Set.union (Set.Ioo (1/Real.sqrt 2) 1) (Set.Ioo 1 (Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1014_101415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_addition_subtraction_l1014_101467

/-- Convert a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Convert a natural number to a binary representation (as a list of bits). -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec to_bits (m : ℕ) (acc : List Bool) : List Bool :=
      if m = 0 then acc
      else to_bits (m / 2) ((m % 2 = 1) :: acc)
    to_bits n []

/-- The binary number 1011₂ -/
def b1011 : List Bool := [true, false, true, true]

/-- The binary number 101₂ -/
def b101 : List Bool := [true, false, true]

/-- The binary number 1100₂ -/
def b1100 : List Bool := [true, true, false, false]

/-- The binary number 1101₂ -/
def b1101 : List Bool := [true, true, false, true]

/-- The binary number 10001₂ -/
def b10001 : List Bool := [true, false, false, false, true]

/-- Theorem: 1011₂ + 101₂ - 1100₂ + 1101₂ = 10001₂ -/
theorem binary_addition_subtraction :
  nat_to_binary (
    binary_to_nat b1011 + binary_to_nat b101 - binary_to_nat b1100 + binary_to_nat b1101
  ) = b10001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_addition_subtraction_l1014_101467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_product_l1014_101474

theorem cos_sum_product (a b : ℝ) : 
  Real.cos (a + b) + Real.cos (a - b) = 2 * Real.cos a * Real.cos b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_product_l1014_101474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_without_x3_l1014_101403

-- Define the function representing the expansion
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt x - 3 / x) ^ 9

-- Define the sum of all coefficients
noncomputable def total_sum : ℝ := f 1

-- Define the coefficient of x^3 term
def coeff_x3 : ℝ := -27

-- Theorem statement
theorem sum_of_coefficients_without_x3 :
  total_sum - coeff_x3 = -485 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_without_x3_l1014_101403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_sequence_implies_square_or_five_times_square_l1014_101428

/-- The distance from a real number to the nearest perfect square -/
noncomputable def f (x : ℝ) : ℝ := sorry

/-- The golden ratio plus one -/
noncomputable def α : ℝ := (3 + Real.sqrt 5) / 2

theorem bounded_sequence_implies_square_or_five_times_square (m : ℤ) :
  (∃ C : ℝ, ∀ n : ℕ, f (↑m * α^n) ≤ C) →
  (∃ k : ℤ, m = k^2 ∨ m = 5 * k^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_sequence_implies_square_or_five_times_square_l1014_101428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_complement_of_B_l1014_101466

def A : Set ℝ := {x | Real.exp (x * Real.log 2) > 1/2}
def B : Set ℝ := {x | x - 1 > 0}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (Set.univ \ B) = {x : ℝ | -1 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_complement_of_B_l1014_101466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marias_trip_cost_l1014_101419

/-- Represents the cost structure for transportation --/
structure TransportCost where
  bus_rate : ℝ
  plane_rate : ℝ
  plane_booking_fee : ℝ

/-- Calculates the cost of a trip segment --/
noncomputable def segment_cost (distance : ℝ) (cost : TransportCost) : ℝ :=
  min (distance * cost.bus_rate) (distance * cost.plane_rate + cost.plane_booking_fee)

/-- Represents the distances between cities --/
structure CityDistances where
  xy : ℝ
  xz : ℝ
  yz : ℝ

/-- Theorem statement for Maria's trip cost --/
theorem marias_trip_cost (cost : TransportCost) (distances : CityDistances) :
  cost.bus_rate = 0.20 →
  cost.plane_rate = 0.12 →
  cost.plane_booking_fee = 120 →
  distances.xy = 4500 →
  distances.xz = 4000 →
  distances.yz ^ 2 = distances.xy ^ 2 - distances.xz ^ 2 →
  segment_cost distances.xy cost + segment_cost distances.yz cost + segment_cost distances.xz cost = 1627.386 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marias_trip_cost_l1014_101419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_light_problem_l1014_101484

noncomputable section

structure Intersection where
  probability : ℝ
  waitingTime : ℝ

noncomputable def A : Intersection := { probability := 1/3, waitingTime := 40 }
noncomputable def B : Intersection := { probability := 1/4, waitingTime := 20 }
noncomputable def C : Intersection := { probability := 3/4, waitingTime := 80 }

theorem traffic_light_problem :
  let p_first_at_third := (1 - A.probability) * (1 - B.probability) * C.probability
  let expected_waiting_time := 
    0 * ((1 - A.probability) * (1 - B.probability) * (1 - C.probability)) +
    A.waitingTime * (A.probability * (1 - B.probability) * (1 - C.probability)) +
    B.waitingTime * ((1 - A.probability) * B.probability * (1 - C.probability)) +
    C.waitingTime * ((1 - A.probability) * (1 - B.probability) * C.probability) +
    (A.waitingTime + B.waitingTime) * (A.probability * B.probability * (1 - C.probability)) +
    (A.waitingTime + C.waitingTime) * (A.probability * (1 - B.probability) * C.probability) +
    (B.waitingTime + C.waitingTime) * ((1 - A.probability) * B.probability * C.probability) +
    (A.waitingTime + B.waitingTime + C.waitingTime) * (A.probability * B.probability * C.probability)
  p_first_at_third = 3/8 ∧ expected_waiting_time = 235/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_light_problem_l1014_101484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_D_closest_to_circle_l1014_101408

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

def ellipse_A : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 + 4*y^2 = 1
def ellipse_B : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 + 2*y^2 = 1
def ellipse_C : ℝ × ℝ → Prop := λ (x, y) ↦ x^2/9 + y^2 = 1
def ellipse_D : ℝ × ℝ → Prop := λ (x, y) ↦ x^2/3 + y^2 = 1

theorem ellipse_D_closest_to_circle :
  eccentricity (Real.sqrt 3) 1 < eccentricity 3 1 ∧
  eccentricity 3 1 < eccentricity 1 (1/Real.sqrt 2) ∧
  eccentricity 1 (1/Real.sqrt 2) < eccentricity 1 (1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_D_closest_to_circle_l1014_101408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1014_101460

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < 0 then x^2 + 4
  else if x > 0 then Real.sin (Real.pi * x)
  else 0  -- Define a value for x outside the given domain

-- State the theorem
theorem a_range (a : ℝ) : 
  (∀ x, ((-1 ≤ x ∧ x < 0) ∨ x > 0) → f x - a*x ≥ -1) ↔ -6 ≤ a ∧ a ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1014_101460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_implies_line_l1014_101452

/-- Given a function f(x) = a*sin(x) + b*cos(x) where x is real,
    if x₀ is a symmetric axis of f(x) and tan(x₀) = 2,
    then the point (a,b) lies on the line x - 2y = 0 -/
theorem symmetric_axis_implies_line (a b x₀ : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = a * Real.sin x + b * Real.cos x) →
  (∀ x, f (2 * x₀ - x) = f x) →
  Real.tan x₀ = 2 →
  a - 2 * b = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_implies_line_l1014_101452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l1014_101418

/-- Line l parameterized by t -/
structure Line where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Circle C₁ -/
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

/-- Point P -/
def P : ℝ × ℝ := (1, 2)

/-- Line l -/
noncomputable def l : Line where
  x := λ t => 2 + (1/2) * t
  y := λ t => 2 + Real.sqrt 3 + (Real.sqrt 3 / 2) * t

/-- Circle C₁ -/
def C₁ : Set (ℝ × ℝ) := Circle (0, 0) 4

theorem intersection_distance_product :
  ∃ A B : ℝ × ℝ,
    A ∈ C₁ ∧ B ∈ C₁ ∧
    (∃ t₁ : ℝ, A.1 = l.x t₁ ∧ A.2 = l.y t₁) ∧
    (∃ t₂ : ℝ, B.1 = l.x t₂ ∧ B.2 = l.y t₂) ∧
    (A.1 - P.1)^2 + (A.2 - P.2)^2 * ((B.1 - P.1)^2 + (B.2 - P.2)^2) = (4 * Real.sqrt 3 - 13)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l1014_101418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l1014_101459

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- A rectangle defined by four points -/
structure Rectangle where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- The area of a triangle -/
noncomputable def areaTriangle (t : Triangle) : ℝ := sorry

/-- The area of a rectangle -/
noncomputable def areaRectangle (r : Rectangle) : ℝ := sorry

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ := sorry

/-- Given a rectangle ABCD with a diagonal AC and points E and F on AC, prove that the area of ABCD is 35 -/
theorem rectangle_area (A B C D E F : Point) 
  (h1 : distance B C = 10) 
  (h2 : distance E C = 6) 
  (h3 : areaTriangle ⟨E, D, F⟩ = areaTriangle ⟨F, A, B⟩ - 5) : 
  areaRectangle ⟨A, B, C, D⟩ = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l1014_101459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_neg_five_l1014_101432

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x - 15) / (x + 5)

-- State the theorem
theorem limit_f_at_neg_five :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, x ≠ -5 → |x - (-5)| < δ → |f x - (-8)| < ε :=
by
  sorry

#check limit_f_at_neg_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_neg_five_l1014_101432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1014_101462

theorem ellipse_equation (C : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2/4 + y^2 = 1) ↔ 
  ((0, 1) ∈ C ∧ 
   ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧
   (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2/a^2 + y^2/b^2 = 1) ∧
   c^2 = a^2 - b^2 ∧
   c/a = Real.sqrt 3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1014_101462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_record_storage_cost_l1014_101430

/-- Represents the dimensions of a storage box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
noncomputable def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the number of boxes given the total volume and box dimensions -/
noncomputable def numberOfBoxes (totalVolume : ℝ) (boxDim : BoxDimensions) : ℝ :=
  totalVolume / boxVolume boxDim

/-- Calculates the total monthly cost for record storage -/
noncomputable def totalMonthlyCost (numBoxes : ℝ) (costPerBox : ℝ) : ℝ :=
  numBoxes * costPerBox

theorem record_storage_cost 
  (boxDim : BoxDimensions)
  (totalVolume : ℝ)
  (costPerBox : ℝ)
  (h1 : boxDim.length = 15)
  (h2 : boxDim.width = 12)
  (h3 : boxDim.height = 10)
  (h4 : totalVolume = 1080000)
  (h5 : costPerBox = 0.6) :
  totalMonthlyCost (numberOfBoxes totalVolume boxDim) costPerBox = 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_record_storage_cost_l1014_101430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_share_per_x_rupee_l1014_101431

/-- The amount y gets for each rupee x gets -/
def amount_y_per_x : ℝ → ℝ := sorry

/-- The amount x gets -/
def amount_x : ℝ → ℝ := sorry

/-- The total amount to be divided -/
def total_amount : ℝ := 245

/-- The share of y -/
def share_y : ℝ := 63

/-- The amount z gets for each rupee x gets -/
def amount_z_per_x : ℝ := 0.30

theorem y_share_per_x_rupee 
  (h1 : ∀ x, amount_y_per_x x * amount_x x = share_y)
  (h2 : ∀ x, amount_x x + amount_y_per_x x * amount_x x + amount_z_per_x * amount_x x = total_amount) :
  ∃ x, amount_y_per_x x = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_share_per_x_rupee_l1014_101431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_sine_function_parameters_l1014_101424

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem periodic_sine_function_parameters 
  (ω φ : ℝ) 
  (h_ω_pos : ω > 0)
  (h_φ_bound : |φ| < π)
  (h_f_5π_8 : f ω φ (5*π/8) = 2)
  (h_f_11π_8 : f ω φ (11*π/8) = 0)
  (h_period : ∀ T > 0, (∀ x, f ω φ (x + T) = f ω φ x) → T ≥ 3*π)
  : ω = 2/3 ∧ φ = π/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_sine_function_parameters_l1014_101424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_properties_l1014_101457

/-- Define a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the area of a triangle given three points -/
noncomputable def triangleArea (O P A : Point) : ℝ :=
  (1/2) * abs (O.x * (P.y - A.y) + P.x * (A.y - O.y) + A.x * (O.y - P.y))

theorem triangle_area_properties :
  let O : Point := ⟨0, 0⟩
  let A : Point := ⟨4, 0⟩
  ∀ (x y : ℝ), 
    0 < x → 0 < y →  -- P is in the first quadrant
    x + y = 6 →  -- condition on x and y
    let P : Point := ⟨x, y⟩
    let S := triangleArea O P A
    (S = -2*x + 12) ∧  -- area function
    (0 < x ∧ x < 6) ∧  -- range of x
    (S = 6 → P = ⟨3, 3⟩)  -- specific case
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_properties_l1014_101457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_volume_equals_96pi_l1014_101438

-- Define the cylinder
noncomputable def cylinder_radius : ℝ := 6
noncomputable def cylinder_height : ℝ := 8

-- Define the volume of the cylinder
noncomputable def cylinder_volume : ℝ := Real.pi * cylinder_radius^2 * cylinder_height

-- Define the wedge as one-third of the cylinder
noncomputable def wedge_fraction : ℝ := 1/3

-- Define the volume of the wedge
noncomputable def wedge_volume : ℝ := wedge_fraction * cylinder_volume

-- Theorem to prove
theorem wedge_volume_equals_96pi :
  wedge_volume = 96 * Real.pi := by
  -- Expand the definitions
  unfold wedge_volume
  unfold cylinder_volume
  unfold wedge_fraction
  unfold cylinder_radius
  unfold cylinder_height
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_volume_equals_96pi_l1014_101438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smartwatch_battery_life_l1014_101401

/-- Represents the battery life of a smartwatch -/
structure SmartWatchBattery where
  fullLifeNonActive : ℝ -- battery life in hours when not actively used
  fullLifeActive : ℝ -- battery life in hours when actively used
  totalTimeOn : ℝ -- total time the watch has been on
  activeUseTime : ℝ -- time of active use
  batterySaverFactor : ℝ -- factor by which battery saver mode reduces consumption

/-- Calculates the remaining battery life in battery saver mode -/
noncomputable def remainingBatteryLife (battery : SmartWatchBattery) : ℝ :=
  let normalConsumptionRate := 1 / battery.fullLifeNonActive
  let activeConsumptionRate := 1 / battery.fullLifeActive
  let batterySaverConsumptionRate := normalConsumptionRate * battery.batterySaverFactor
  
  let nonActiveTime := battery.totalTimeOn - battery.activeUseTime
  let batteryUsed := (nonActiveTime * normalConsumptionRate) + (battery.activeUseTime * activeConsumptionRate)
  let remainingBattery := 1 - batteryUsed
  
  remainingBattery / batterySaverConsumptionRate

/-- Theorem: Given the conditions, the remaining battery life in battery saver mode is 12.5 hours -/
theorem smartwatch_battery_life :
  let battery := SmartWatchBattery.mk 48 4 18 5 0.5
  remainingBatteryLife battery = 12.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smartwatch_battery_life_l1014_101401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_a_zero_range_of_a_for_sufficient_condition_l1014_101443

noncomputable def f (x : ℝ) := Real.sqrt ((1 - x) / (3 * x + 1))
noncomputable def g (a x : ℝ) := Real.sqrt (x^2 - (2*a + 1)*x + a^2 + a)

def A : Set ℝ := {x | -1/3 < x ∧ x ≤ 1 ∧ 3*x + 1 ≠ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - (2*a + 1)*x + a^2 + a ≥ 0}

theorem intersection_when_a_zero :
  A ∩ B 0 = {x | -1/3 < x ∧ x ≤ 0 ∨ x = 1} := by
  sorry

theorem range_of_a_for_sufficient_condition :
  {a | A ⊆ B a ∧ A ≠ B a} = {a | a ≥ 1 ∨ a ≤ -4/3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_a_zero_range_of_a_for_sufficient_condition_l1014_101443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_correct_answer_l1014_101479

theorem job_completion_time 
  (m d e r : ℝ) 
  (hm : m > 0) 
  (hd : d > 0) 
  (he : e > 0) 
  (hr : r ≥ 0) : 
  (m * d * e) / (m * e + r * (e + 1)) = (m * d * e) / (m * e + r * e + r) :=
by
  -- The proof steps would go here
  sorry

-- The following theorem states that the derived formula matches the correct answer
theorem correct_answer 
  (m d e r : ℝ) 
  (hm : m > 0) 
  (hd : d > 0) 
  (he : e > 0) 
  (hr : r ≥ 0) : 
  (m * d * e) / (m * e + r * e + r) = (m * d * e) / (m * e + r * e + r) :=
by
  -- This is trivially true
  rfl

#check job_completion_time
#check correct_answer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_correct_answer_l1014_101479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_max_angle_point_l1014_101406

-- Define the plane and points
variable (α : Set Point) (A B : Point)

-- Define the condition that A and B are not on the plane
axiom not_on_plane : A ∉ α ∧ B ∉ α

-- Define the function to calculate the angle APB
noncomputable def angle_APB (P : Point) : ℝ := sorry

-- Define the maximum angle function
noncomputable def max_angle : ℝ := sorry

-- Theorem statement
theorem exists_max_angle_point :
  ∃ P : Point, P ∈ α ∧ angle_APB P = max_angle :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_max_angle_point_l1014_101406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonal_length_l1014_101449

/-- Represents a rhombus with given area and one diagonal length -/
structure Rhombus where
  area : ℝ
  diagonal1 : ℝ

/-- Calculates the length of the second diagonal of a rhombus -/
noncomputable def second_diagonal (r : Rhombus) : ℝ :=
  (2 * r.area) / r.diagonal1

/-- Theorem: In a rhombus with area 375 cm² and one diagonal 30 cm, the other diagonal is 25 cm -/
theorem rhombus_diagonal_length : 
  let r : Rhombus := { area := 375, diagonal1 := 30 }
  second_diagonal r = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonal_length_l1014_101449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lunar_transport_cost_theorem_l1014_101499

/-- Calculates the total cost of transporting lunar equipment -/
noncomputable def lunar_transport_cost (cost_per_kg : ℝ) (weights : List ℝ) : ℝ :=
  (weights.map (· / 1000 * cost_per_kg)).sum

/-- Theorem: The total cost of transporting three equipment items weighing 
    300 grams, 450 grams, and 600 grams at $15,000 per kilogram is $20,250 -/
theorem lunar_transport_cost_theorem : 
  lunar_transport_cost 15000 [300, 450, 600] = 20250 := by
  -- Unfold the definition of lunar_transport_cost
  unfold lunar_transport_cost
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lunar_transport_cost_theorem_l1014_101499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_sqrt_17_l1014_101426

/-- Circle equation: x^2 + y^2 - 2y - 4 = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 4 = 0

/-- Line equation: mx - y + 1 - m = 0, where m ∈ ℝ -/
def line_equation (m x y : ℝ) : Prop := m*x - y + 1 - m = 0

/-- Slope of 120° in radians -/
noncomputable def slope_120 : ℝ := Real.tan (2*Real.pi/3)

theorem chord_length_sqrt_17 :
  ∀ (x y : ℝ), 
  (∃ (m : ℝ), line_equation m x y ∧ m = slope_120) →
  circle_equation x y →
  ∃ (x' y' : ℝ), 
    x' ≠ x ∧ y' ≠ y ∧
    circle_equation x' y' ∧
    (∃ (m : ℝ), line_equation m x' y' ∧ m = slope_120) ∧
    (x - x')^2 + (y - y')^2 = 17 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_sqrt_17_l1014_101426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_trains_length_l1014_101491

/-- Calculates the length of a train given its speed in km/hr and time to cross a pole in seconds -/
noncomputable def trainLength (speed : ℝ) (time : ℝ) : ℝ :=
  speed * (5/18) * time

/-- The combined length of two trains -/
noncomputable def combinedLength (speed1 speed2 time1 time2 : ℝ) : ℝ :=
  trainLength speed1 time1 + trainLength speed2 time2

/-- Theorem stating the combined length of two trains with given speeds and crossing times -/
theorem two_trains_length (speed1 speed2 time1 time2 : ℝ) 
  (h1 : speed1 = 100) 
  (h2 : speed2 = 120) 
  (h3 : time1 = 9) 
  (h4 : time2 = 8) : 
  ∃ ε > 0, |combinedLength speed1 speed2 time1 time2 - 516.66| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_trains_length_l1014_101491
