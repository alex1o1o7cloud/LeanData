import Mathlib

namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2080_208076

theorem absolute_value_inequality_solution_set : 
  {x : ℝ | |x| > -1} = Set.univ :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2080_208076


namespace NUMINAMATH_CALUDE_equation_solution_l2080_208022

theorem equation_solution : ∃! x : ℝ, (2 / x = 1 / (x + 1)) ∧ (x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2080_208022


namespace NUMINAMATH_CALUDE_opposite_sum_and_sum_opposite_l2080_208055

theorem opposite_sum_and_sum_opposite (a b : ℤ) (h1 : a = -6) (h2 : b = 4) : 
  (-a) + (-b) = 2 ∧ -(a + b) = 2 :=
by sorry

end NUMINAMATH_CALUDE_opposite_sum_and_sum_opposite_l2080_208055


namespace NUMINAMATH_CALUDE_animal_population_canada_animal_population_l2080_208078

/-- The combined population of moose, beavers, caribou, wolves, grizzly bears, and mountain lions in Canada, given the specified ratios and human population. -/
theorem animal_population (human_population : ℝ) : ℝ :=
  let beaver_population := human_population / 19
  let moose_population := beaver_population / 2
  let caribou_population := 3/2 * moose_population
  let wolf_population := 4 * caribou_population
  let grizzly_population := wolf_population / 3
  let mountain_lion_population := grizzly_population / 2
  moose_population + beaver_population + caribou_population + wolf_population + grizzly_population + mountain_lion_population

/-- Theorem stating that the combined animal population in Canada is 13.5 million, given a human population of 38 million. -/
theorem canada_animal_population :
  animal_population 38 = 13.5 := by sorry

end NUMINAMATH_CALUDE_animal_population_canada_animal_population_l2080_208078


namespace NUMINAMATH_CALUDE_unoccupied_volume_of_cube_l2080_208045

/-- The volume of a cube not occupied by five spheres --/
theorem unoccupied_volume_of_cube (π : Real) : 
  let cube_edge : Real := 2
  let sphere_radius : Real := 1
  let cube_volume : Real := cube_edge ^ 3
  let sphere_volume : Real := (4 / 3) * π * sphere_radius ^ 3
  let total_sphere_volume : Real := 5 * sphere_volume
  cube_volume - total_sphere_volume = 8 - (20 / 3) * π := by sorry

end NUMINAMATH_CALUDE_unoccupied_volume_of_cube_l2080_208045


namespace NUMINAMATH_CALUDE_diophantine_equation_7z_squared_l2080_208081

theorem diophantine_equation_7z_squared (x y z : ℕ) : 
  x^2 + y^2 = 7 * z^2 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_7z_squared_l2080_208081


namespace NUMINAMATH_CALUDE_find_F_when_C_is_35_l2080_208072

-- Define the relationship between C and F
def C_F_relation (C F : ℝ) : Prop := C = (4/7) * (F - 40)

-- State the theorem
theorem find_F_when_C_is_35 :
  ∃ F : ℝ, C_F_relation 35 F ∧ F = 101.25 := by
  sorry

end NUMINAMATH_CALUDE_find_F_when_C_is_35_l2080_208072


namespace NUMINAMATH_CALUDE_pattern_equality_l2080_208046

theorem pattern_equality (n : ℕ+) : n * (n + 2) + 1 = (n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_pattern_equality_l2080_208046


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2080_208050

/-- Given an arithmetic sequence with first term 5, second term 12, and last term 40,
    the sum of the two terms immediately preceding 40 is 59. -/
theorem arithmetic_sequence_sum (a : ℕ → ℕ) : 
  a 0 = 5 → a 1 = 12 → 
  (∃ n : ℕ, a n = 40 ∧ ∀ k < n, a k < 40) →
  (∀ i j k : ℕ, i < j → j < k → a j - a i = a k - a j) →
  (∃ m : ℕ, a m + a (m + 1) = 59 ∧ a (m + 2) = 40) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2080_208050


namespace NUMINAMATH_CALUDE_angle_DAB_is_54_degrees_l2080_208032

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A triangle defined by three points -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- A pentagon defined by five points -/
structure Pentagon :=
  (B : Point) (C : Point) (D : Point) (E : Point) (G : Point)

/-- The measure of an angle in degrees -/
def angle_measure (p q r : Point) : ℝ := sorry

/-- The distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Checks if a triangle is isosceles -/
def is_isosceles (t : Triangle) : Prop :=
  distance t.C t.A = distance t.C t.B

/-- Checks if a pentagon is regular -/
def is_regular_pentagon (p : Pentagon) : Prop := sorry

/-- Theorem: In an isosceles triangle with a regular pentagon constructed on one side,
    the angle DAB measures 54 degrees -/
theorem angle_DAB_is_54_degrees 
  (t : Triangle) 
  (p : Pentagon) 
  (h1 : is_isosceles t) 
  (h2 : is_regular_pentagon p)
  (h3 : p.B = t.B ∧ p.C = t.C)
  (D : Point) 
  : angle_measure D t.A t.B = 54 := by sorry

end NUMINAMATH_CALUDE_angle_DAB_is_54_degrees_l2080_208032


namespace NUMINAMATH_CALUDE_dvd_cd_ratio_l2080_208099

theorem dvd_cd_ratio (total : ℕ) (dvds : ℕ) (h1 : total = 273) (h2 : dvds = 168) :
  (dvds : ℚ) / (total - dvds : ℚ) = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_dvd_cd_ratio_l2080_208099


namespace NUMINAMATH_CALUDE_arithmetic_mean_expressions_l2080_208062

theorem arithmetic_mean_expressions (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ((x + a) / y + (y - b) / x) / 2 = (x^2 + a*x + y^2 - b*y) / (2*x*y) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_expressions_l2080_208062


namespace NUMINAMATH_CALUDE_regular_bottle_is_16_oz_l2080_208016

/-- Represents Jon's drinking habits and fluid intake --/
structure DrinkingHabits where
  awake_hours : ℕ := 16
  drinking_interval : ℕ := 4
  larger_bottles_per_day : ℕ := 2
  larger_bottle_size_factor : ℚ := 1.25
  weekly_fluid_intake : ℕ := 728

/-- Calculates the size of Jon's regular water bottle in ounces --/
def regular_bottle_size (h : DrinkingHabits) : ℚ :=
  h.weekly_fluid_intake / (7 * (h.awake_hours / h.drinking_interval + h.larger_bottles_per_day * h.larger_bottle_size_factor))

/-- Theorem stating that Jon's regular water bottle size is 16 ounces --/
theorem regular_bottle_is_16_oz (h : DrinkingHabits) : regular_bottle_size h = 16 := by
  sorry

end NUMINAMATH_CALUDE_regular_bottle_is_16_oz_l2080_208016


namespace NUMINAMATH_CALUDE_cubic_quadratic_relation_l2080_208088

theorem cubic_quadratic_relation (A B C D : ℝ) (p q r : ℝ) (a b : ℝ) : 
  (A * p^3 + B * p^2 + C * p + D = 0) →
  (A * q^3 + B * q^2 + C * q + D = 0) →
  (A * r^3 + B * r^2 + C * r + D = 0) →
  ((p^2 + q)^2 + a * (p^2 + q) + b = 0) →
  ((q^2 + r)^2 + a * (q^2 + r) + b = 0) →
  ((r^2 + p)^2 + a * (r^2 + p) + b = 0) →
  (A ≠ 0) →
  a = (A * B + 2 * A * C - B^2) / A^2 := by
sorry

end NUMINAMATH_CALUDE_cubic_quadratic_relation_l2080_208088


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_2_area_implies_a_eq_1_l2080_208044

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2| + |x - a|

-- Theorem for the first part of the problem
theorem solution_set_for_a_eq_2 :
  {x : ℝ | f 2 x > 6} = {x : ℝ | x > 3 ∨ x < -3} := by sorry

-- Theorem for the second part of the problem
theorem area_implies_a_eq_1 (a : ℝ) (h : a > 0) :
  (∫ x in {x | f a x ≤ 5}, (5 - f a x)) = 8 → a = 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_2_area_implies_a_eq_1_l2080_208044


namespace NUMINAMATH_CALUDE_prime_with_integer_roots_l2080_208085

theorem prime_with_integer_roots (p : ℕ) (h_prime : Nat.Prime p) :
  (∃ x y : ℤ, x^2 + p*x - 204*p = 0 ∧ y^2 + p*y - 204*p = 0) → p = 17 := by
  sorry

end NUMINAMATH_CALUDE_prime_with_integer_roots_l2080_208085


namespace NUMINAMATH_CALUDE_not_perfect_square_l2080_208068

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, 7 * n + 3 = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l2080_208068


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2080_208035

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 4) :
  (1 + 1 / (x + 1)) * ((x + 1) / (x^2 + 4*x + 4)) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2080_208035


namespace NUMINAMATH_CALUDE_centroid_is_unique_interior_point_l2080_208021

/-- A point in the integer lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle defined by three lattice points -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Predicate to check if a point is inside a triangle -/
def IsInside (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- Predicate to check if a point is on the boundary of a triangle -/
def IsOnBoundary (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- The centroid of a triangle -/
def Centroid (t : LatticeTriangle) : LatticePoint := sorry

/-- Main theorem -/
theorem centroid_is_unique_interior_point (t : LatticeTriangle) 
  (h1 : ∀ p : LatticePoint, IsOnBoundary p t → p = t.A ∨ p = t.B ∨ p = t.C)
  (h2 : ∃! p : LatticePoint, IsInside p t) :
  ∃ p : LatticePoint, IsInside p t ∧ p = Centroid t := by
  sorry

end NUMINAMATH_CALUDE_centroid_is_unique_interior_point_l2080_208021


namespace NUMINAMATH_CALUDE_liquid_rise_ratio_in_cones_l2080_208079

theorem liquid_rise_ratio_in_cones (r₁ r₂ r_marble : ℝ) 
  (h₁ h₂ : ℝ) (V : ℝ) :
  r₁ = 4 →
  r₂ = 8 →
  r_marble = 2 →
  V = (1/3) * π * r₁^2 * h₁ →
  V = (1/3) * π * r₂^2 * h₂ →
  let V_marble := (4/3) * π * r_marble^3
  let h₁' := h₁ + V_marble / ((1/3) * π * r₁^2)
  let h₂' := h₂ + V_marble / ((1/3) * π * r₂^2)
  (h₁' - h₁) / (h₂' - h₂) = 4 :=
by sorry

#check liquid_rise_ratio_in_cones

end NUMINAMATH_CALUDE_liquid_rise_ratio_in_cones_l2080_208079


namespace NUMINAMATH_CALUDE_power_quotient_rule_l2080_208060

theorem power_quotient_rule (a : ℝ) : a^5 / a^3 = a^2 := by sorry

end NUMINAMATH_CALUDE_power_quotient_rule_l2080_208060


namespace NUMINAMATH_CALUDE_vector_ratio_bounds_l2080_208012

theorem vector_ratio_bounds (a b : ℝ × ℝ) 
  (h1 : ‖a + b‖ = 3)
  (h2 : ‖a - b‖ = 2) :
  (2 / 5 : ℝ) ≤ ‖a‖ / (a.1 * b.1 + a.2 * b.2) ∧ 
  ‖a‖ / (a.1 * b.1 + a.2 * b.2) ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_vector_ratio_bounds_l2080_208012


namespace NUMINAMATH_CALUDE_least_valid_tree_count_l2080_208014

def is_valid_tree_count (n : ℕ) : Prop :=
  n ≥ 100 ∧ n % 7 = 0 ∧ n % 6 = 0 ∧ n % 4 = 0

theorem least_valid_tree_count :
  ∃ (n : ℕ), is_valid_tree_count n ∧ ∀ m < n, ¬is_valid_tree_count m :=
by sorry

end NUMINAMATH_CALUDE_least_valid_tree_count_l2080_208014


namespace NUMINAMATH_CALUDE_m_range_l2080_208013

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x, x^2 + m*x + 1 ≠ 0

def q (m : ℝ) : Prop := m > 2

-- Define the condition that either p or q is true, but not both
def condition (m : ℝ) : Prop := (p m ∨ q m) ∧ ¬(p m ∧ q m)

-- State the theorem
theorem m_range (m : ℝ) : condition m → ((-2 < m ∧ m < 2) ∨ m > 2) := by
  sorry

end NUMINAMATH_CALUDE_m_range_l2080_208013


namespace NUMINAMATH_CALUDE_yellow_balls_count_l2080_208007

theorem yellow_balls_count (white_balls : ℕ) (total_balls : ℕ) 
  (h1 : white_balls = 4)
  (h2 : (white_balls : ℚ) / total_balls = 2 / 3) :
  total_balls - white_balls = 2 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l2080_208007


namespace NUMINAMATH_CALUDE_square_equals_cube_root_16_l2080_208037

theorem square_equals_cube_root_16 : ∃! x : ℝ, x > 0 ∧ x^2 = (Real.sqrt 16)^3 := by
  sorry

end NUMINAMATH_CALUDE_square_equals_cube_root_16_l2080_208037


namespace NUMINAMATH_CALUDE_tangent_line_is_correct_l2080_208028

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 5

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x

/-- The point on the curve -/
def point : ℝ × ℝ := (-1, -3)

/-- The proposed tangent line equation -/
def tangent_line (x y : ℝ) : Prop := 3*x + y + 6 = 0

theorem tangent_line_is_correct : 
  tangent_line point.1 point.2 ∧ 
  (∀ x : ℝ, tangent_line x (f x) → x = point.1) ∧
  f' point.1 = 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_is_correct_l2080_208028


namespace NUMINAMATH_CALUDE_chocolate_chip_cookie_batches_l2080_208066

/-- Given:
  - Each batch of chocolate chip cookies contains 3 cookies.
  - There are 4 oatmeal cookies.
  - The total number of cookies is 10.
Prove that the number of batches of chocolate chip cookies is 2. -/
theorem chocolate_chip_cookie_batches :
  ∀ (batch_size : ℕ) (oatmeal_cookies : ℕ) (total_cookies : ℕ),
    batch_size = 3 →
    oatmeal_cookies = 4 →
    total_cookies = 10 →
    (total_cookies - oatmeal_cookies) / batch_size = 2 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_chip_cookie_batches_l2080_208066


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2080_208091

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 3*x + 1) * (y^2 + 3*y + 1) * (z^2 + 3*z + 1) / (x*y*z) ≥ 125 :=
by sorry

theorem min_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x^2 + 3*x + 1) * (y^2 + 3*y + 1) * (z^2 + 3*z + 1) / (x*y*z) = 125 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2080_208091


namespace NUMINAMATH_CALUDE_inequality_proofs_l2080_208005

theorem inequality_proofs :
  (∀ x : ℝ, 4*x - 2 < 1 - 2*x → x < 1/2) ∧
  (∀ x : ℝ, 3 - 2*x ≥ x - 6 ∧ (3*x + 1)/2 < 2*x → 1 < x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proofs_l2080_208005


namespace NUMINAMATH_CALUDE_p_toluidine_molecular_weight_l2080_208056

/-- The atomic weight of Carbon in atomic mass units (amu) -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in atomic mass units (amu) -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of Nitrogen in atomic mass units (amu) -/
def nitrogen_weight : ℝ := 14.01

/-- The chemical formula of p-Toluidine -/
structure ChemicalFormula where
  carbon : ℕ
  hydrogen : ℕ
  nitrogen : ℕ

/-- The chemical formula of p-Toluidine (C7H9N) -/
def p_toluidine : ChemicalFormula := ⟨7, 9, 1⟩

/-- Calculate the molecular weight of a chemical compound given its formula -/
def molecular_weight (formula : ChemicalFormula) : ℝ :=
  formula.carbon * carbon_weight + 
  formula.hydrogen * hydrogen_weight + 
  formula.nitrogen * nitrogen_weight

/-- Theorem: The molecular weight of p-Toluidine is 107.152 amu -/
theorem p_toluidine_molecular_weight : 
  molecular_weight p_toluidine = 107.152 := by
  sorry

end NUMINAMATH_CALUDE_p_toluidine_molecular_weight_l2080_208056


namespace NUMINAMATH_CALUDE_tetrahedron_self_dual_cube_octahedron_dual_dodecahedron_icosahedron_dual_l2080_208095

/-- A polyhedron with faces and vertices -/
structure Polyhedron where
  faces : ℕ
  vertices : ℕ
  face_sides : ℕ
  vertex_valence : ℕ

/-- Duality relation between polyhedra -/
def is_dual (p q : Polyhedron) : Prop :=
  p.faces = q.vertices ∧ p.vertices = q.faces ∧
  p.face_sides = q.vertex_valence ∧ p.vertex_valence = q.face_sides

/-- Self-duality of a polyhedron -/
def is_self_dual (p : Polyhedron) : Prop :=
  is_dual p p

/-- Theorem: Tetrahedron is self-dual -/
theorem tetrahedron_self_dual :
  is_self_dual ⟨4, 4, 3, 3⟩ := by sorry

/-- Theorem: Cube and octahedron are dual -/
theorem cube_octahedron_dual :
  is_dual ⟨6, 8, 4, 3⟩ ⟨8, 6, 3, 4⟩ := by sorry

/-- Theorem: Dodecahedron and icosahedron are dual -/
theorem dodecahedron_icosahedron_dual :
  is_dual ⟨12, 20, 5, 3⟩ ⟨20, 12, 3, 5⟩ := by sorry

end NUMINAMATH_CALUDE_tetrahedron_self_dual_cube_octahedron_dual_dodecahedron_icosahedron_dual_l2080_208095


namespace NUMINAMATH_CALUDE_expression_evaluation_l2080_208048

theorem expression_evaluation (m : ℝ) (h : m = 1) : 
  (1 - 1 / (m - 2)) / ((m^2 - 6*m + 9) / (m - 2)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2080_208048


namespace NUMINAMATH_CALUDE_sqrt_x_plus_reciprocal_l2080_208063

theorem sqrt_x_plus_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) : 
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52 := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_reciprocal_l2080_208063


namespace NUMINAMATH_CALUDE_A_disjoint_B_iff_l2080_208043

/-- The set A defined by the quadratic inequality -/
def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}

/-- The set B defined by the linear inequalities involving m -/
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

/-- Theorem stating the condition for A and B to be disjoint -/
theorem A_disjoint_B_iff (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end NUMINAMATH_CALUDE_A_disjoint_B_iff_l2080_208043


namespace NUMINAMATH_CALUDE_athul_rowing_time_l2080_208070

theorem athul_rowing_time (upstream_distance : ℝ) (downstream_distance : ℝ) (stream_speed : ℝ) :
  upstream_distance = 16 →
  downstream_distance = 24 →
  stream_speed = 1 →
  ∃ (rowing_speed : ℝ),
    rowing_speed > stream_speed ∧
    (upstream_distance / (rowing_speed - stream_speed) = downstream_distance / (rowing_speed + stream_speed)) ∧
    (upstream_distance / (rowing_speed - stream_speed) = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_athul_rowing_time_l2080_208070


namespace NUMINAMATH_CALUDE_problem_solution_l2080_208096

theorem problem_solution (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ + 64*x₈ = 2)
  (eq2 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ + 100*x₈ = 24)
  (eq3 : 16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ + 121*x₈ = 246)
  (eq4 : 25*x₁ + 36*x₂ + 49*x₃ + 64*x₄ + 81*x₅ + 100*x₆ + 121*x₇ + 144*x₈ = 1234) :
  36*x₁ + 49*x₂ + 64*x₃ + 81*x₄ + 100*x₅ + 121*x₆ + 144*x₇ + 169*x₈ = 1594 := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l2080_208096


namespace NUMINAMATH_CALUDE_sector_max_area_and_angle_l2080_208084

/-- Given a sector of a circle with perimeter 30 cm, prove that the maximum area is 225/4 cm² 
    and the corresponding central angle is 2 radians. -/
theorem sector_max_area_and_angle (r : ℝ) (l : ℝ) (α : ℝ) (area : ℝ) :
  l + 2 * r = 30 →                            -- Perimeter condition
  l = r * α →                                 -- Arc length formula
  area = (1 / 2) * r * l →                    -- Area formula for sector
  (∀ r' l' α' area', l' + 2 * r' = 30 → l' = r' * α' → area' = (1 / 2) * r' * l' → area' ≤ area) →
  area = 225 / 4 ∧ α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_max_area_and_angle_l2080_208084


namespace NUMINAMATH_CALUDE_vector_operation_l2080_208059

/-- Given two vectors in ℝ², prove that their specific linear combination equals a certain vector. -/
theorem vector_operation (a b : ℝ × ℝ) : 
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l2080_208059


namespace NUMINAMATH_CALUDE_unique_divisor_square_equality_l2080_208042

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := (Nat.divisors n.val).card

/-- Theorem: The only positive integer n that satisfies n = [d(n)]^2 is 1 -/
theorem unique_divisor_square_equality :
  ∀ n : ℕ+, n.val = (num_divisors n)^2 → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisor_square_equality_l2080_208042


namespace NUMINAMATH_CALUDE_rectangle_y_value_l2080_208069

theorem rectangle_y_value (y : ℝ) (h1 : y > 0) : 
  let vertices : List (ℝ × ℝ) := [(-2, y), (10, y), (-2, -1), (10, -1)]
  let length : ℝ := 10 - (-2)
  let height : ℝ := y - (-1)
  let area : ℝ := length * height
  area = 108 → y = 8 := by sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l2080_208069


namespace NUMINAMATH_CALUDE_solve_apple_dealer_problem_l2080_208026

/-- Represents the apple dealer problem -/
def apple_dealer_problem (cost_per_bushel : ℚ) (apples_per_bushel : ℕ) (profit : ℚ) (apples_sold : ℕ) : Prop :=
  let cost_per_apple : ℚ := cost_per_bushel / apples_per_bushel
  let total_cost : ℚ := cost_per_apple * apples_sold
  let total_revenue : ℚ := total_cost + profit
  let price_per_apple : ℚ := total_revenue / apples_sold
  price_per_apple = 40 / 100

/-- Theorem stating the solution to the apple dealer problem -/
theorem solve_apple_dealer_problem :
  apple_dealer_problem 12 48 15 100 := by
  sorry

end NUMINAMATH_CALUDE_solve_apple_dealer_problem_l2080_208026


namespace NUMINAMATH_CALUDE_chord_bisector_l2080_208038

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Theorem statement
theorem chord_bisector :
  ellipse P.1 P.2 →
  ∃ (A B : ℝ × ℝ),
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    P = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧
    (∀ (x y : ℝ), line_equation x y ↔ ∃ t : ℝ, x = A.1 + t*(B.1 - A.1) ∧ y = A.2 + t*(B.2 - A.2)) :=
sorry

end NUMINAMATH_CALUDE_chord_bisector_l2080_208038


namespace NUMINAMATH_CALUDE_first_group_weight_proof_l2080_208071

-- Define the number of girls in the second group
def second_group_count : ℕ := 8

-- Define the average weights
def first_group_avg : ℝ := 50.25
def second_group_avg : ℝ := 45.15
def total_avg : ℝ := 48.55

-- Define the theorem
theorem first_group_weight_proof :
  ∃ (first_group_count : ℕ),
    (first_group_count * first_group_avg + second_group_count * second_group_avg) / 
    (first_group_count + second_group_count) = total_avg →
    first_group_avg = 50.25 := by
  sorry


end NUMINAMATH_CALUDE_first_group_weight_proof_l2080_208071


namespace NUMINAMATH_CALUDE_odd_function_property_l2080_208017

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : f (-1) = 2) :
  f 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l2080_208017


namespace NUMINAMATH_CALUDE_jerry_weekly_earnings_l2080_208006

/-- Jerry's weekly earnings calculation --/
theorem jerry_weekly_earnings
  (rate_per_task : ℕ)
  (hours_per_task : ℕ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (h1 : rate_per_task = 40)
  (h2 : hours_per_task = 2)
  (h3 : hours_per_day = 10)
  (h4 : days_per_week = 7) :
  (rate_per_task * (hours_per_day / hours_per_task) * days_per_week : ℕ) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_jerry_weekly_earnings_l2080_208006


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_value_l2080_208053

theorem sum_and_reciprocal_value (x : ℝ) (h1 : x ≠ 0) (h2 : x^2 + (1/x)^2 = 23) : 
  x + (1/x) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_value_l2080_208053


namespace NUMINAMATH_CALUDE_least_b_value_l2080_208002

def num_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem least_b_value (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h_a_factors : num_factors a = 4) 
  (h_b_factors : num_factors b = a) 
  (h_b_div_a : a ∣ b) : 
  ∀ c, c > 0 ∧ num_factors c = a ∧ a ∣ c → b ≤ c ∧ b = 12 := by
  sorry

end NUMINAMATH_CALUDE_least_b_value_l2080_208002


namespace NUMINAMATH_CALUDE_train_length_l2080_208025

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 90 → time = 10 → speed * time * (1000 / 3600) = 250 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2080_208025


namespace NUMINAMATH_CALUDE_yvonne_swims_10_laps_l2080_208075

/-- The number of laps Yvonne can swim -/
def yvonne_laps : ℕ := sorry

/-- The number of laps Yvonne's younger sister can swim -/
def sister_laps : ℕ := sorry

/-- The number of laps Joel can swim -/
def joel_laps : ℕ := 15

theorem yvonne_swims_10_laps :
  (sister_laps = yvonne_laps / 2) →
  (joel_laps = 3 * sister_laps) →
  (yvonne_laps = 10) :=
by sorry

end NUMINAMATH_CALUDE_yvonne_swims_10_laps_l2080_208075


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_reciprocals_l2080_208089

theorem min_value_of_sum_of_reciprocals (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 2) :
  (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) ≥ 9/4 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_reciprocals_l2080_208089


namespace NUMINAMATH_CALUDE_laundry_dishes_time_difference_l2080_208049

theorem laundry_dishes_time_difference 
  (dawn_dish_time andy_laundry_time : ℕ) 
  (h1 : dawn_dish_time = 20) 
  (h2 : andy_laundry_time = 46) : 
  andy_laundry_time - 2 * dawn_dish_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_laundry_dishes_time_difference_l2080_208049


namespace NUMINAMATH_CALUDE_parallel_line_point_l2080_208020

/-- Given two points on a line and another line it's parallel to, prove the x-coordinate of the second point. -/
theorem parallel_line_point (j : ℝ) : 
  (∃ (m b : ℝ), (2 : ℝ) + 3 * m = -6 ∧ 
                 (19 : ℝ) - (-3) = m * (j - 4)) → 
  j = -29 := by
sorry

end NUMINAMATH_CALUDE_parallel_line_point_l2080_208020


namespace NUMINAMATH_CALUDE_apps_deleted_l2080_208029

theorem apps_deleted (initial_apps final_apps : ℝ) 
  (h1 : initial_apps = 300.5)
  (h2 : final_apps = 129.5) :
  initial_apps - final_apps = 171 := by
  sorry

end NUMINAMATH_CALUDE_apps_deleted_l2080_208029


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l2080_208097

theorem cryptarithm_solution :
  ∃! (C H U K T R I G N S : ℕ),
    C < 10 ∧ H < 10 ∧ U < 10 ∧ K < 10 ∧ T < 10 ∧ R < 10 ∧ I < 10 ∧ G < 10 ∧ N < 10 ∧ S < 10 ∧
    T ≠ 0 ∧
    C ≠ H ∧ C ≠ U ∧ C ≠ K ∧ C ≠ T ∧ C ≠ R ∧ C ≠ I ∧ C ≠ G ∧ C ≠ N ∧ C ≠ S ∧
    H ≠ U ∧ H ≠ K ∧ H ≠ T ∧ H ≠ R ∧ H ≠ I ∧ H ≠ G ∧ H ≠ N ∧ H ≠ S ∧
    U ≠ K ∧ U ≠ T ∧ U ≠ R ∧ U ≠ I ∧ U ≠ G ∧ U ≠ N ∧ U ≠ S ∧
    K ≠ T ∧ K ≠ R ∧ K ≠ I ∧ K ≠ G ∧ K ≠ N ∧ K ≠ S ∧
    T ≠ R ∧ T ≠ I ∧ T ≠ G ∧ T ≠ N ∧ T ≠ S ∧
    R ≠ I ∧ R ≠ G ∧ R ≠ N ∧ R ≠ S ∧
    I ≠ G ∧ I ≠ N ∧ I ≠ S ∧
    G ≠ N ∧ G ≠ S ∧
    N ≠ S ∧
    100000*C + 10000*H + 1000*U + 100*C + 10*K +
    100000*T + 10000*R + 1000*I + 100*G + 10*G +
    100000*T + 10000*U + 1000*R + 100*N + 10*S =
    100000*T + 10000*R + 1000*I + 100*C + 10*K + S ∧
    C = 9 ∧ H = 3 ∧ U = 5 ∧ K = 4 ∧ T = 1 ∧ R = 2 ∧ I = 0 ∧ G = 6 ∧ N = 8 ∧ S = 7 := by
  sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l2080_208097


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l2080_208077

theorem solve_quadratic_equation (k p : ℝ) (hk : k ≠ 0) (hp : p ≠ 0) :
  let y : ℝ := -(p + k^2) / (2*k)
  (y - 2*k)^2 - (y - 3*k)^2 = 4*k^2 - p := by
sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l2080_208077


namespace NUMINAMATH_CALUDE_set_equality_l2080_208087

open Set

-- Define the sets
def R : Set ℝ := univ
def A : Set ℝ := {x | x^2 ≥ 4}
def B : Set ℝ := {y | ∃ x, y = |Real.tan x|}

-- State the theorem
theorem set_equality : (R \ A) ∩ B = {x | 0 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_set_equality_l2080_208087


namespace NUMINAMATH_CALUDE_company_shares_l2080_208086

theorem company_shares (p v s i : Real) : 
  p + v + s + i = 1 → 
  2*p + v + s + i = 1.3 →
  p + 2*v + s + i = 1.4 →
  p + v + 3*s + i = 1.2 →
  ∃ k : Real, k > 3.75 ∧ k * i > 0.75 := by sorry

end NUMINAMATH_CALUDE_company_shares_l2080_208086


namespace NUMINAMATH_CALUDE_circle_properties_l2080_208031

def circle_equation (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

theorem circle_properties :
  (∃ k : ℝ, ∀ x y : ℝ, circle_equation x y → x = 0 ∧ y = k) ∧
  (∀ x y : ℝ, circle_equation x y → (x - 0)^2 + (y - 2)^2 = 1) ∧
  circle_equation 1 2 :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l2080_208031


namespace NUMINAMATH_CALUDE_min_value_expression_l2080_208011

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 2*a + 3*b + 4*c = 1) : 
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → 2*x + 3*y + 4*z = 1 → 
    1/a + 2/b + 3/c ≤ 1/x + 2/y + 3/z) ∧ 
  1/a + 2/b + 3/c = 20 + 4*Real.sqrt 3 + 20*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2080_208011


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficients_l2080_208019

theorem binomial_expansion_coefficients :
  ∃ (b₁ b₂ b₃ b₄ : ℝ),
    (∀ x : ℝ, x^4 = (x+1)^4 + b₁*(x+1)^3 + b₂*(x+1)^2 + b₃*(x+1) + b₄) ∧
    b₁ = -4 ∧ b₂ = 6 ∧ b₃ = -4 ∧ b₄ = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficients_l2080_208019


namespace NUMINAMATH_CALUDE_mice_problem_l2080_208018

theorem mice_problem (x : ℕ) : 
  (x / 2 : ℕ) * 2 = x ∧ 
  ((x - x / 2) / 3 : ℕ) * 3 = x - x / 2 ∧
  (((x - x / 2) - (x - x / 2) / 3) / 4 : ℕ) * 4 = (x - x / 2) - (x - x / 2) / 3 ∧
  ((((x - x / 2) - (x - x / 2) / 3) - ((x - x / 2) - (x - x / 2) / 3) / 4) / 5 : ℕ) * 5 = 
    ((x - x / 2) - (x - x / 2) / 3) - ((x - x / 2) - (x - x / 2) / 3) / 4 ∧
  ((((x - x / 2) - (x - x / 2) / 3) - ((x - x / 2) - (x - x / 2) / 3) / 4) - 
    ((((x - x / 2) - (x - x / 2) / 3) - ((x - x / 2) - (x - x / 2) / 3) / 4) / 5)) = 
    (x - x / 2) / 3 + 2 →
  x = 60 := by
sorry

end NUMINAMATH_CALUDE_mice_problem_l2080_208018


namespace NUMINAMATH_CALUDE_machine_output_percentage_l2080_208052

theorem machine_output_percentage :
  let prob_defect_A : ℝ := 9 / 1000
  let prob_defect_B : ℝ := 1 / 50
  let total_prob_defect : ℝ := 0.0156
  ∃ p : ℝ, 
    0 ≤ p ∧ p ≤ 1 ∧
    total_prob_defect = p * prob_defect_A + (1 - p) * prob_defect_B ∧
    p = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_machine_output_percentage_l2080_208052


namespace NUMINAMATH_CALUDE_square_difference_equality_l2080_208058

theorem square_difference_equality : (23 + 15)^2 - 3 * (23 - 15)^2 = 1252 := by sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2080_208058


namespace NUMINAMATH_CALUDE_smallest_perfect_square_factor_l2080_208064

def y : ℕ := 2^5 * 3^2 * 4^6 * 5^6 * 7^8 * 8^9 * 9^10

theorem smallest_perfect_square_factor (k : ℕ) : 
  (k > 0 ∧ ∃ m : ℕ, k * y = m^2) → k ≥ 100 :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_factor_l2080_208064


namespace NUMINAMATH_CALUDE_IMO_2001_max_sum_l2080_208074

theorem IMO_2001_max_sum : 
  ∀ I M O : ℕ+,
  I ≠ M → I ≠ O → M ≠ O →
  I * M * O = 2001 →
  I + M + O ≤ 671 :=
by
  sorry

end NUMINAMATH_CALUDE_IMO_2001_max_sum_l2080_208074


namespace NUMINAMATH_CALUDE_wall_construction_boys_l2080_208000

/-- The number of boys who can construct the wall in 6 days -/
def num_boys : ℕ := 24

/-- The number of days it takes B boys or 24 girls to construct the wall -/
def days_boys_or_girls : ℕ := 6

/-- The number of days it takes B boys and 12 girls to construct the wall -/
def days_boys_and_girls : ℕ := 4

/-- The number of girls that can construct the wall in the same time as B boys -/
def equivalent_girls : ℕ := 24

theorem wall_construction_boys (B : ℕ) :
  (B * days_boys_or_girls = equivalent_girls * days_boys_or_girls) →
  ((B + 12 * equivalent_girls) * days_boys_and_girls = equivalent_girls * days_boys_or_girls) →
  B = num_boys :=
by sorry

end NUMINAMATH_CALUDE_wall_construction_boys_l2080_208000


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l2080_208003

theorem fourth_term_of_geometric_progression (a b c : ℝ) :
  a = Real.sqrt 2 →
  b = Real.rpow 2 (1/3) →
  c = Real.rpow 2 (1/6) →
  b / a = c / b →
  c * (b / a) = 1 :=
by sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l2080_208003


namespace NUMINAMATH_CALUDE_max_value_f_times_g_l2080_208001

noncomputable def f (x : ℝ) : ℝ := 3 - x

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (2 * x + 5)

def is_non_negative (x : ℝ) : Prop := x ≥ 0

theorem max_value_f_times_g :
  ∃ (M : ℝ), M = 2 * Real.sqrt 3 - 1 ∧
  (∀ (x : ℝ), is_non_negative x →
    (f x * g x = min (f x) (g x)) →
    f x * g x ≤ M) ∧
  (∃ (x : ℝ), is_non_negative x ∧
    (f x * g x = min (f x) (g x)) ∧
    f x * g x = M) :=
sorry

end NUMINAMATH_CALUDE_max_value_f_times_g_l2080_208001


namespace NUMINAMATH_CALUDE_kamals_chemistry_marks_l2080_208030

theorem kamals_chemistry_marks 
  (english_marks : ℕ)
  (math_marks : ℕ)
  (physics_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℕ)
  (num_subjects : ℕ)
  (h1 : english_marks = 96)
  (h2 : math_marks = 65)
  (h3 : physics_marks = 82)
  (h4 : biology_marks = 85)
  (h5 : average_marks = 79)
  (h6 : num_subjects = 5)
  : ∃ (chemistry_marks : ℕ), 
    (english_marks + math_marks + physics_marks + biology_marks + chemistry_marks) / num_subjects = average_marks ∧ 
    chemistry_marks = 67 :=
by sorry

end NUMINAMATH_CALUDE_kamals_chemistry_marks_l2080_208030


namespace NUMINAMATH_CALUDE_fencing_cost_per_meter_l2080_208061

/-- Proves that the cost of fencing per meter for a rectangular plot with given dimensions and total fencing cost is 26.50 Rs. -/
theorem fencing_cost_per_meter
  (length breadth : ℝ)
  (length_relation : length = breadth + 10)
  (length_value : length = 55)
  (total_cost : ℝ)
  (total_cost_value : total_cost = 5300)
  : total_cost / (2 * (length + breadth)) = 26.50 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_per_meter_l2080_208061


namespace NUMINAMATH_CALUDE_chain_rule_with_local_injectivity_l2080_208033

/-- Given two differentiable functions f and g, with f having a local injectivity property,
    prove that their composition is differentiable and satisfies the chain rule. -/
theorem chain_rule_with_local_injectivity 
  (f : ℝ → ℝ) (g : ℝ → ℝ) (x₀ : ℝ) 
  (hf : DifferentiableAt ℝ f x₀)
  (hg : DifferentiableAt ℝ g (f x₀))
  (hU : ∃ U : Set ℝ, IsOpen U ∧ x₀ ∈ U ∧ ∀ x ∈ U, x ≠ x₀ → f x ≠ f x₀) :
  DifferentiableAt ℝ (g ∘ f) x₀ ∧ 
  deriv (g ∘ f) x₀ = deriv g (f x₀) * deriv f x₀ :=
by sorry

end NUMINAMATH_CALUDE_chain_rule_with_local_injectivity_l2080_208033


namespace NUMINAMATH_CALUDE_sum_of_data_l2080_208065

theorem sum_of_data (a b c : ℝ) : 
  a + b = c → 
  b = 3 * a → 
  a = 12 → 
  a + b + c = 96 := by
sorry

end NUMINAMATH_CALUDE_sum_of_data_l2080_208065


namespace NUMINAMATH_CALUDE_abc_sum_zero_product_nonpositive_l2080_208024

theorem abc_sum_zero_product_nonpositive (a b c : ℝ) (h : a + b + c = 0) :
  (∀ x ≤ 0, ∃ a b c : ℝ, a + b + c = 0 ∧ a * b + a * c + b * c = x) ∧
  (∀ a b c : ℝ, a + b + c = 0 → a * b + a * c + b * c ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_zero_product_nonpositive_l2080_208024


namespace NUMINAMATH_CALUDE_souvenir_relationship_l2080_208015

/-- Represents the number of souvenirs of each type -/
structure SouvenirCount where
  x : ℕ  -- 20 cents souvenirs
  y : ℕ  -- 25 cents souvenirs
  z : ℕ  -- 35 cents souvenirs

/-- Conditions of the souvenir distribution problem -/
def SouvenirProblem (s : SouvenirCount) : Prop :=
  s.x + s.y + s.z = 2000 ∧
  20 * s.x + 25 * s.y + 35 * s.z = 52000

/-- Theorem stating the relationship between 25 cents and 35 cents souvenirs -/
theorem souvenir_relationship (s : SouvenirCount) 
  (h : SouvenirProblem s) : 5 * s.y + 15 * s.z = 12000 := by
  sorry

end NUMINAMATH_CALUDE_souvenir_relationship_l2080_208015


namespace NUMINAMATH_CALUDE_total_lives_calculation_l2080_208040

theorem total_lives_calculation (initial_players : ℕ) (new_players : ℕ) (lives_per_player : ℕ) : 
  initial_players = 16 → new_players = 4 → lives_per_player = 10 →
  (initial_players + new_players) * lives_per_player = 200 := by
sorry

end NUMINAMATH_CALUDE_total_lives_calculation_l2080_208040


namespace NUMINAMATH_CALUDE_parabola_translation_l2080_208034

-- Define the original parabola function
def original_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the translation amount
def translation_amount : ℝ := 3

-- Define the translated parabola function
def translated_parabola (x : ℝ) : ℝ := original_parabola x + translation_amount

-- Theorem statement
theorem parabola_translation :
  ∀ x : ℝ, translated_parabola x = -2 * x^2 + 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2080_208034


namespace NUMINAMATH_CALUDE_ferris_wheel_cost_l2080_208080

def rollercoaster_rides : ℕ := 3
def catapult_rides : ℕ := 2
def ferris_wheel_rides : ℕ := 1
def rollercoaster_cost : ℕ := 4
def catapult_cost : ℕ := 4
def total_tickets : ℕ := 21

theorem ferris_wheel_cost :
  total_tickets - (rollercoaster_rides * rollercoaster_cost + catapult_rides * catapult_cost) = ferris_wheel_rides := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_cost_l2080_208080


namespace NUMINAMATH_CALUDE_product_of_real_parts_l2080_208008

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_equation (z : ℂ) : Prop :=
  z^2 + 2*z = 10 - 2*i

-- Define the roots of the quadratic equation
noncomputable def roots : Set ℂ :=
  {z : ℂ | quadratic_equation z}

-- State the theorem
theorem product_of_real_parts :
  ∃ (z₁ z₂ : ℂ), z₁ ∈ roots ∧ z₂ ∈ roots ∧ 
  (z₁.re * z₂.re : ℝ) = -10.25 :=
sorry

end NUMINAMATH_CALUDE_product_of_real_parts_l2080_208008


namespace NUMINAMATH_CALUDE_number_of_stools_l2080_208009

/-- Represents the number of legs on a stool -/
def stool_legs : ℕ := 3

/-- Represents the number of legs on a chair -/
def chair_legs : ℕ := 4

/-- Represents the total number of legs in the room when people sit on all furniture -/
def total_legs : ℕ := 39

/-- Theorem stating that the number of three-legged stools is 3 -/
theorem number_of_stools (x y : ℕ) 
  (h : stool_legs * x + chair_legs * y = total_legs) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_of_stools_l2080_208009


namespace NUMINAMATH_CALUDE_initial_rate_is_three_l2080_208067

/-- Calculates the initial consumption rate per soldier per day -/
def initial_consumption_rate (initial_soldiers : ℕ) (initial_duration : ℕ) 
  (additional_soldiers : ℕ) (new_consumption_rate : ℚ) (new_duration : ℕ) : ℚ :=
  (((initial_soldiers + additional_soldiers) * new_consumption_rate * new_duration) / 
   (initial_soldiers * initial_duration))

/-- Theorem stating that the initial consumption rate is 3 kg per soldier per day -/
theorem initial_rate_is_three :
  initial_consumption_rate 1200 30 528 (5/2) 25 = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_rate_is_three_l2080_208067


namespace NUMINAMATH_CALUDE_triangle_area_problem_l2080_208041

theorem triangle_area_problem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * (2*x) * x = 72 → x = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_problem_l2080_208041


namespace NUMINAMATH_CALUDE_equal_digit_probability_l2080_208093

def num_dice : ℕ := 6
def sides_per_die : ℕ := 16
def one_digit_prob : ℚ := 9 / 16
def two_digit_prob : ℚ := 7 / 16

theorem equal_digit_probability : 
  (num_dice.choose (num_dice / 2)) * (one_digit_prob ^ (num_dice / 2)) * (two_digit_prob ^ (num_dice / 2)) = 3115125 / 10485760 := by
  sorry

end NUMINAMATH_CALUDE_equal_digit_probability_l2080_208093


namespace NUMINAMATH_CALUDE_length_MN_l2080_208083

/-- The length of MN where M and N are points on two lines and S is their midpoint -/
theorem length_MN (M N S : ℝ × ℝ) : 
  S = (10, 8) →
  (∃ x₁, M = (x₁, 14 * x₁ / 9)) →
  (∃ x₂, N = (x₂, 5 * x₂ / 12)) →
  S.1 = (M.1 + N.1) / 2 →
  S.2 = (M.2 + N.2) / 2 →
  ∃ length, length = Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_length_MN_l2080_208083


namespace NUMINAMATH_CALUDE_total_cost_calculation_l2080_208027

def calculate_total_cost (tv_price sound_price warranty_price install_price : ℝ)
  (tv_discount1 tv_discount2 sound_discount warranty_discount : ℝ)
  (tv_sound_tax warranty_install_tax : ℝ) : ℝ :=
  let tv_after_discounts := tv_price * (1 - tv_discount1) * (1 - tv_discount2)
  let sound_after_discount := sound_price * (1 - sound_discount)
  let warranty_after_discount := warranty_price * (1 - warranty_discount)
  let tv_with_tax := tv_after_discounts * (1 + tv_sound_tax)
  let sound_with_tax := sound_after_discount * (1 + tv_sound_tax)
  let warranty_with_tax := warranty_after_discount * (1 + warranty_install_tax)
  let install_with_tax := install_price * (1 + warranty_install_tax)
  tv_with_tax + sound_with_tax + warranty_with_tax + install_with_tax

theorem total_cost_calculation :
  calculate_total_cost 600 400 100 150 0.1 0.15 0.2 0.3 0.08 0.05 = 1072.32 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l2080_208027


namespace NUMINAMATH_CALUDE_inscribed_circle_exists_l2080_208092

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a circle with center and radius -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a rectangle -/
structure Rectangle :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Given configuration satisfies the problem conditions -/
def ValidConfiguration (rect : Rectangle) (circleA : Circle) (circleB : Circle) (circleC : Circle) (circleD : Circle) (e : ℝ) : Prop :=
  let a := circleA.radius
  let b := circleB.radius
  let c := circleC.radius
  let d := circleD.radius
  (a + c = b + d) ∧ (a + c < e) ∧
  (rect.A = circleA.center) ∧ (rect.B = circleB.center) ∧ (rect.C = circleC.center) ∧ (rect.D = circleD.center)

/-- Theorem: A circle can be inscribed in the quadrilateral formed by outer common tangents -/
theorem inscribed_circle_exists (rect : Rectangle) (circleA : Circle) (circleB : Circle) (circleC : Circle) (circleD : Circle) (e : ℝ) 
  (h : ValidConfiguration rect circleA circleB circleC circleD e) : 
  ∃ (inscribedCircle : Circle), true :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_exists_l2080_208092


namespace NUMINAMATH_CALUDE_function_simplification_and_sum_l2080_208098

noncomputable def f (x : ℝ) : ℝ := (x^3 + 3*x^2 - 4*x - 12) / (x - 3)

theorem function_simplification_and_sum :
  ∃ (A B C D : ℝ),
    (∀ x : ℝ, x ≠ D → f x = A * x^2 + B * x + C) ∧
    (∀ x : ℝ, f x = A * x^2 + B * x + C ↔ x ≠ D) ∧
    A + B + C + D = 24 := by
  sorry

end NUMINAMATH_CALUDE_function_simplification_and_sum_l2080_208098


namespace NUMINAMATH_CALUDE_natural_number_representation_l2080_208039

theorem natural_number_representation (A : ℕ) :
  ∃ n : ℕ, A = 3 * n ∨ A = 3 * n + 1 ∨ A = 3 * n + 2 := by
  sorry

end NUMINAMATH_CALUDE_natural_number_representation_l2080_208039


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l2080_208094

theorem quadratic_roots_range (k : ℝ) : 
  (∀ x : ℝ, x^2 - 11*x + (30 + k) = 0 → x > 5) → 
  0 < k ∧ k ≤ 1/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l2080_208094


namespace NUMINAMATH_CALUDE_junior_score_l2080_208010

theorem junior_score (n : ℝ) (junior_proportion : ℝ) (senior_proportion : ℝ) 
  (overall_average : ℝ) (senior_average : ℝ) :
  junior_proportion = 0.3 →
  senior_proportion = 0.7 →
  overall_average = 79 →
  senior_average = 75 →
  junior_proportion + senior_proportion = 1 →
  let junior_score := (overall_average - senior_average * senior_proportion) / junior_proportion
  junior_score = 88 := by
  sorry

end NUMINAMATH_CALUDE_junior_score_l2080_208010


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l2080_208047

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)
  (h2 : a = (2, 0))
  (h3 : Real.sqrt ((Prod.fst b)^2 + (Prod.snd b)^2) = 1) :
  Real.sqrt ((Prod.fst (a + 2 • b))^2 + (Prod.snd (a + 2 • b))^2) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l2080_208047


namespace NUMINAMATH_CALUDE_array_exists_iff_even_l2080_208057

/-- A type representing the possible entries in the array -/
inductive Entry
  | neg : Entry
  | zero : Entry
  | pos : Entry

/-- Definition of a valid array -/
def ValidArray (n : ℕ) (arr : Matrix (Fin n) (Fin n) Entry) : Prop :=
  ∀ (i j : Fin n), arr i j ∈ [Entry.neg, Entry.zero, Entry.pos]

/-- Definition of row sum -/
def RowSum (n : ℕ) (arr : Matrix (Fin n) (Fin n) Entry) (i : Fin n) : ℤ :=
  (Finset.univ.sum fun j => match arr i j with
    | Entry.neg => -1
    | Entry.zero => 0
    | Entry.pos => 1)

/-- Definition of column sum -/
def ColSum (n : ℕ) (arr : Matrix (Fin n) (Fin n) Entry) (j : Fin n) : ℤ :=
  (Finset.univ.sum fun i => match arr i j with
    | Entry.neg => -1
    | Entry.zero => 0
    | Entry.pos => 1)

/-- All sums are different -/
def AllSumsDifferent (n : ℕ) (arr : Matrix (Fin n) (Fin n) Entry) : Prop :=
  ∀ (i j i' j' : Fin n), 
    (RowSum n arr i = RowSum n arr i' → i = i') ∧
    (ColSum n arr j = ColSum n arr j' → j = j') ∧
    (RowSum n arr i ≠ ColSum n arr j)

/-- Main theorem: The array with described properties exists if and only if n is even -/
theorem array_exists_iff_even (n : ℕ) :
  (∃ (arr : Matrix (Fin n) (Fin n) Entry), 
    ValidArray n arr ∧ AllSumsDifferent n arr) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_array_exists_iff_even_l2080_208057


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l2080_208036

theorem adult_tickets_sold (adult_price child_price total_tickets total_receipts : ℕ) 
  (h1 : adult_price = 12)
  (h2 : child_price = 4)
  (h3 : total_tickets = 130)
  (h4 : total_receipts = 840) :
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_receipts ∧
    adult_tickets = 40 := by
  sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l2080_208036


namespace NUMINAMATH_CALUDE_crazy_silly_school_difference_l2080_208004

/-- The number of books in the 'crazy silly school' series -/
def num_books : ℕ := 15

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := 14

/-- Theorem: The difference between the number of books and movies in the 'crazy silly school' series is 1 -/
theorem crazy_silly_school_difference : num_books - num_movies = 1 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_difference_l2080_208004


namespace NUMINAMATH_CALUDE_triangle_inequality_possible_third_side_length_l2080_208023

theorem triangle_inequality (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 → a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

theorem possible_third_side_length : 
  ∃ (x : ℝ), x > 0 ∧ 3 + 6 > x ∧ 6 + x > 3 ∧ x + 3 > 6 ∧ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_possible_third_side_length_l2080_208023


namespace NUMINAMATH_CALUDE_clubsuit_ratio_l2080_208051

-- Define the ♣ operation
def clubsuit (n m : ℕ) : ℕ := n^2 * m^3

-- Theorem statement
theorem clubsuit_ratio : (clubsuit 3 5) / (clubsuit 5 3) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_clubsuit_ratio_l2080_208051


namespace NUMINAMATH_CALUDE_three_integer_chords_l2080_208082

/-- Represents a circle with a given radius and a point at a given distance from its center -/
structure CircleWithPoint where
  radius : ℝ
  distanceFromCenter : ℝ

/-- Counts the number of integer-length chords containing the given point -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

theorem three_integer_chords :
  let c := CircleWithPoint.mk 13 5
  countIntegerChords c = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_integer_chords_l2080_208082


namespace NUMINAMATH_CALUDE_reading_time_difference_l2080_208073

theorem reading_time_difference 
  (xanthia_speed : ℝ) 
  (molly_speed : ℝ) 
  (book_pages : ℝ) 
  (h1 : xanthia_speed = 120) 
  (h2 : molly_speed = 60) 
  (h3 : book_pages = 300) : 
  (book_pages / molly_speed - book_pages / xanthia_speed) * 60 = 150 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_l2080_208073


namespace NUMINAMATH_CALUDE_square_roots_problem_l2080_208090

theorem square_roots_problem (x a : ℝ) : 
  x > 0 ∧ (2*a - 3)^2 = x ∧ (5 - a)^2 = x → a = -2 ∧ x = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l2080_208090


namespace NUMINAMATH_CALUDE_circle_radius_l2080_208054

theorem circle_radius (A : ℝ) (h : A = 81 * Real.pi) : 
  ∃ r : ℝ, r > 0 ∧ A = Real.pi * r^2 ∧ r = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l2080_208054
