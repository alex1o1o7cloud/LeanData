import Mathlib

namespace NUMINAMATH_CALUDE_odd_function_value_l2618_261882

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_value (f : ℝ → ℝ) (h1 : odd_function f) (h2 : f 3 = 7) :
  f (-3) = -7 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l2618_261882


namespace NUMINAMATH_CALUDE_right_triangle_trig_l2618_261830

theorem right_triangle_trig (D E F : ℝ) (h1 : D = 90) (h2 : E = 8) (h3 : F = 17) :
  let cosF := E / F
  let sinF := Real.sqrt (F^2 - E^2) / F
  cosF = 8 / 17 ∧ sinF = 15 / 17 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_trig_l2618_261830


namespace NUMINAMATH_CALUDE_pyramid_properties_l2618_261865

/-- Pyramid structure with given properties -/
structure Pyramid where
  -- Base is a rhombus
  base_is_rhombus : Prop
  -- Height of the pyramid
  height : ℝ
  -- K lies on diagonal AC
  k_on_diagonal : Prop
  -- KC = KA + AC
  kc_eq_ka_plus_ac : Prop
  -- Length of lateral edge TC
  tc_length : ℝ
  -- Angles of lateral faces to base
  angle1 : ℝ
  angle2 : ℝ

/-- Theorem stating the properties of the pyramid -/
theorem pyramid_properties (p : Pyramid)
  (h_height : p.height = 1)
  (h_tc : p.tc_length = 2 * Real.sqrt 2)
  (h_angles : p.angle1 = π/6 ∧ p.angle2 = π/3) :
  ∃ (base_side angle_ta_tcd : ℝ),
    base_side = 7/6 ∧
    angle_ta_tcd = Real.arcsin (Real.sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_properties_l2618_261865


namespace NUMINAMATH_CALUDE_parallel_lines_angle_condition_l2618_261868

-- Define the concept of lines and planes
variable (Line Plane : Type)

-- Define the concept of parallel lines
variable (parallel : Line → Line → Prop)

-- Define the concept of a line forming an angle with a plane
variable (angle_with_plane : Line → Plane → ℝ)

-- State the theorem
theorem parallel_lines_angle_condition 
  (a b : Line) (α : Plane) :
  (parallel a b → angle_with_plane a α = angle_with_plane b α) ∧
  ¬(angle_with_plane a α = angle_with_plane b α → parallel a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_angle_condition_l2618_261868


namespace NUMINAMATH_CALUDE_problem_statement_l2618_261847

theorem problem_statement (a b c : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : c > 0) :
  (a^2 - b*c > b^2 - a*c) ∧ (a^3 > b^2) ∧ (a + 1/a > b + 1/b) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2618_261847


namespace NUMINAMATH_CALUDE_equal_side_length_l2618_261809

/-- An isosceles right-angled triangle with side lengths a, a, and c, where the sum of squares of sides is 725 --/
structure IsoscelesRightTriangle where
  a : ℝ
  c : ℝ
  isosceles : c^2 = 2 * a^2
  sum_of_squares : a^2 + a^2 + c^2 = 725

/-- The length of each equal side in the isosceles right-angled triangle is 13.5 --/
theorem equal_side_length (t : IsoscelesRightTriangle) : t.a = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_equal_side_length_l2618_261809


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2618_261879

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 3) = 7 → x = 46 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2618_261879


namespace NUMINAMATH_CALUDE_circumference_difference_l2618_261808

/-- Given two circles A and B with areas and π as specified, 
    prove that the difference between their circumferences is 6.2 cm -/
theorem circumference_difference (π : ℝ) (area_A area_B : ℝ) :
  π = 3.1 →
  area_A = 198.4 →
  area_B = 251.1 →
  let radius_A := Real.sqrt (area_A / π)
  let radius_B := Real.sqrt (area_B / π)
  let circumference_A := 2 * π * radius_A
  let circumference_B := 2 * π * radius_B
  circumference_B - circumference_A = 6.2 := by
  sorry

end NUMINAMATH_CALUDE_circumference_difference_l2618_261808


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2618_261813

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2618_261813


namespace NUMINAMATH_CALUDE_tan_C_value_triangle_area_l2618_261875

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def satisfies_condition_1 (t : Triangle) : Prop :=
  (Real.sin t.A / t.a) + (Real.sin t.B / t.b) = (Real.cos t.C / t.c)

def satisfies_condition_2 (t : Triangle) : Prop :=
  t.a^2 + t.b^2 - t.c^2 = 8

-- Theorem 1
theorem tan_C_value (t : Triangle) (h : satisfies_condition_1 t) :
  Real.tan t.C = 1/2 := by sorry

-- Theorem 2
theorem triangle_area (t : Triangle) (h1 : satisfies_condition_1 t) (h2 : satisfies_condition_2 t) :
  (1/2) * t.a * t.b * Real.sin t.C = 1 := by sorry

end NUMINAMATH_CALUDE_tan_C_value_triangle_area_l2618_261875


namespace NUMINAMATH_CALUDE_max_parts_two_planes_l2618_261892

-- Define a plane in 3D space
def Plane3D : Type := Unit

-- Define the possible relationships between two planes
inductive PlaneRelationship
| Parallel
| Intersecting

-- Function to calculate the number of parts based on the relationship
def numParts (rel : PlaneRelationship) : ℕ :=
  match rel with
  | PlaneRelationship.Parallel => 3
  | PlaneRelationship.Intersecting => 4

-- Theorem statement
theorem max_parts_two_planes (p1 p2 : Plane3D) :
  ∃ (rel : PlaneRelationship), ∀ (r : PlaneRelationship), numParts rel ≥ numParts r :=
sorry

end NUMINAMATH_CALUDE_max_parts_two_planes_l2618_261892


namespace NUMINAMATH_CALUDE_solve_equation_l2618_261810

theorem solve_equation : ∃ x : ℝ, (2 * x + 7) / 5 = 17 ∧ x = 39 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2618_261810


namespace NUMINAMATH_CALUDE_positive_difference_of_roots_l2618_261815

theorem positive_difference_of_roots (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 2*x^2 - 10*x + 18 - (2*x + 34)
  let roots := {x : ℝ | f x = 0}
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ r₁ ≠ r₂ ∧ |r₁ - r₂| = 2 * Real.sqrt 17 :=
sorry

end NUMINAMATH_CALUDE_positive_difference_of_roots_l2618_261815


namespace NUMINAMATH_CALUDE_class_mean_calculation_l2618_261860

theorem class_mean_calculation (total_students : ℕ) (first_group : ℕ) (second_group : ℕ)
  (first_mean : ℚ) (second_mean : ℚ) :
  total_students = first_group + second_group →
  first_group = 40 →
  second_group = 10 →
  first_mean = 68 / 100 →
  second_mean = 74 / 100 →
  (first_group * first_mean + second_group * second_mean) / total_students = 692 / 1000 := by
sorry

#eval (40 * (68 : ℚ) / 100 + 10 * (74 : ℚ) / 100) / 50

end NUMINAMATH_CALUDE_class_mean_calculation_l2618_261860


namespace NUMINAMATH_CALUDE_expected_count_in_sample_l2618_261805

/-- 
Given a population where 1/4 of the members have a certain characteristic,
prove that the expected number of individuals with that characteristic
in a random sample of 300 is 75.
-/
theorem expected_count_in_sample 
  (population_probability : ℚ) 
  (sample_size : ℕ) 
  (h1 : population_probability = 1 / 4) 
  (h2 : sample_size = 300) : 
  population_probability * sample_size = 75 := by
sorry

end NUMINAMATH_CALUDE_expected_count_in_sample_l2618_261805


namespace NUMINAMATH_CALUDE_fourth_pentagon_dots_l2618_261824

/-- Represents the number of dots in the nth pentagon of the sequence -/
def dots (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else if n = 2 then 6
  else if n = 3 then 16
  else dots (n - 1) + 5 * (n - 1)

/-- The theorem stating that the fourth pentagon has 31 dots -/
theorem fourth_pentagon_dots : dots 4 = 31 := by
  sorry


end NUMINAMATH_CALUDE_fourth_pentagon_dots_l2618_261824


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2618_261893

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : ∃ (M N Q : ℝ × ℝ),
  let F1 := (-c, 0)
  let F2 := (c, 0)
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  M ∈ C ∧ N ∈ C ∧
  (N.1 - M.1) * 2 * c = 0 ∧
  (N.2 - M.2) * (F2.1 - F1.1) = 0 ∧
  (F2.1 - F1.1) = 4 * Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) ∧
  Q ∈ C ∧
  (Q.1 - F1.1)^2 + (Q.2 - F1.2)^2 = (N.1 - Q.1)^2 + (N.2 - Q.2)^2 →
  c^2 / a^2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2618_261893


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2618_261823

/-- Hyperbola with given properties and intersecting circle -/
structure HyperbolaWithCircle where
  b : ℝ
  h_b_pos : b > 0
  hyperbola : ℝ × ℝ → Prop := fun (x, y) ↦ x^2 - y^2/b^2 = 1
  asymptote : ℝ → ℝ := fun x ↦ b * x
  circle : ℝ × ℝ → Prop := fun (x, y) ↦ x^2 + y^2 = 1
  intersection_area : ℝ := b

/-- The eccentricity of a hyperbola with the given properties is √3 -/
theorem hyperbola_eccentricity (h : HyperbolaWithCircle) : 
  ∃ (e : ℝ), e = Real.sqrt 3 ∧ e^2 = 1 + 1/h.b^2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2618_261823


namespace NUMINAMATH_CALUDE_probability_of_black_piece_l2618_261878

/-- Given a set of items with two types, this function calculates the probability of selecting an item of a specific type. -/
def probability_of_selection (total : ℕ) (type_a : ℕ) : ℚ :=
  type_a / total

/-- The probability of selecting a black piece from a set of Go pieces -/
theorem probability_of_black_piece : probability_of_selection 7 4 = 4 / 7 := by
  sorry

#eval probability_of_selection 7 4

end NUMINAMATH_CALUDE_probability_of_black_piece_l2618_261878


namespace NUMINAMATH_CALUDE_only_D_is_symmetric_l2618_261883

-- Define the type for shapes
inductive Shape
| A
| B
| C
| D
| E

-- Define a function to check if a shape is horizontally symmetric
def isHorizontallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.D => True
  | _ => False

-- Theorem statement
theorem only_D_is_symmetric :
  ∀ s : Shape, isHorizontallySymmetric s ↔ s = Shape.D :=
by
  sorry

end NUMINAMATH_CALUDE_only_D_is_symmetric_l2618_261883


namespace NUMINAMATH_CALUDE_bobbys_remaining_candy_l2618_261825

/-- Given Bobby's initial candy count and the amounts eaten, prove that the remaining candy count is 8. -/
theorem bobbys_remaining_candy (initial_candy : ℕ) (first_eaten : ℕ) (second_eaten : ℕ)
  (h1 : initial_candy = 22)
  (h2 : first_eaten = 9)
  (h3 : second_eaten = 5) :
  initial_candy - first_eaten - second_eaten = 8 := by
  sorry

end NUMINAMATH_CALUDE_bobbys_remaining_candy_l2618_261825


namespace NUMINAMATH_CALUDE_trapezoid_triangle_area_l2618_261816

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a trapezoid ABCD -/
structure Trapezoid :=
  (A B C D : Point)

/-- Checks if two line segments are perpendicular -/
def isPerpendicular (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Checks if a point is on a line segment -/
def isOnSegment (p : Point) (p1 p2 : Point) : Prop := sorry

/-- Checks if two line segments are parallel -/
def isParallel (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Main theorem -/
theorem trapezoid_triangle_area
  (ABCD : Trapezoid)
  (E : Point)
  (h1 : isPerpendicular ABCD.A ABCD.D ABCD.D ABCD.C)
  (h2 : ABCD.A.x - ABCD.D.x = 5)
  (h3 : ABCD.A.y - ABCD.B.y = 5)
  (h4 : ABCD.D.x - ABCD.C.x = 10)
  (h5 : isOnSegment E ABCD.D ABCD.C)
  (h6 : E.x - ABCD.D.x = 4)
  (h7 : isParallel ABCD.B E ABCD.A ABCD.D)
  : triangleArea ABCD.A ABCD.D E = 10 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_triangle_area_l2618_261816


namespace NUMINAMATH_CALUDE_number_equation_l2618_261802

theorem number_equation (x : ℝ) : x - (105 / 21) = 5995 ↔ x = 6000 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l2618_261802


namespace NUMINAMATH_CALUDE_beth_sheep_count_l2618_261854

-- Define the number of sheep Beth has
def beth_sheep : ℕ := 76

-- Define Aaron's sheep in terms of Beth's
def aaron_sheep : ℕ := 7 * beth_sheep

-- State the theorem
theorem beth_sheep_count : 
  beth_sheep + aaron_sheep = 608 ∧ aaron_sheep = 7 * beth_sheep → beth_sheep = 76 := by
  sorry

end NUMINAMATH_CALUDE_beth_sheep_count_l2618_261854


namespace NUMINAMATH_CALUDE_part1_solution_set_part2_m_range_l2618_261861

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| + |x + 3|

-- Part 1
theorem part1_solution_set :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} := by sorry

-- Part 2
theorem part2_m_range :
  {m : ℝ | ∃ x, f m x ≤ 2*m - 5} = {m : ℝ | m ≥ 8} := by sorry

end NUMINAMATH_CALUDE_part1_solution_set_part2_m_range_l2618_261861


namespace NUMINAMATH_CALUDE_green_green_pairs_l2618_261880

/-- Represents the distribution of shirt colors and pairs in a classroom --/
structure Classroom where
  total_students : ℕ
  red_students : ℕ
  green_students : ℕ
  total_pairs : ℕ
  red_red_pairs : ℕ

/-- The theorem states that given the classroom conditions, 
    the number of pairs where both students wear green is 35 --/
theorem green_green_pairs (c : Classroom) 
  (h1 : c.total_students = 144)
  (h2 : c.red_students = 63)
  (h3 : c.green_students = 81)
  (h4 : c.total_pairs = 72)
  (h5 : c.red_red_pairs = 26)
  (h6 : c.total_students = c.red_students + c.green_students) :
  c.total_pairs - c.red_red_pairs - (c.red_students - 2 * c.red_red_pairs) = 35 := by
  sorry

#check green_green_pairs

end NUMINAMATH_CALUDE_green_green_pairs_l2618_261880


namespace NUMINAMATH_CALUDE_part_one_part_two_l2618_261827

-- Define the sets A, B, and C
def A (a : ℝ) := {x : ℝ | x^2 - a*x + a^2 - 19 = 0}
def B := {x : ℝ | x^2 - 5*x + 6 = 0}
def C := {x : ℝ | x^2 + 2*x - 8 = 0}

-- Theorem for part (1)
theorem part_one (a : ℝ) : A a ∩ B = A a ∪ B → a = 5 := by sorry

-- Theorem for part (2)
theorem part_two (a : ℝ) : (∅ ⊂ A a ∩ B) ∧ (A a ∩ C = ∅) → a = -2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2618_261827


namespace NUMINAMATH_CALUDE_square_difference_equals_product_l2618_261807

theorem square_difference_equals_product : (15 + 7)^2 - (7^2 + 15^2) = 210 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_product_l2618_261807


namespace NUMINAMATH_CALUDE_complex_absolute_value_l2618_261801

theorem complex_absolute_value (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l2618_261801


namespace NUMINAMATH_CALUDE_winnie_the_pooh_honey_l2618_261842

def honey_pot (initial_weight : ℝ) (empty_pot_weight : ℝ) : Prop :=
  ∃ (w1 w2 w3 w4 : ℝ),
    w1 = initial_weight / 2 ∧
    w2 = w1 / 2 ∧
    w3 = w2 / 2 ∧
    w4 = w3 / 2 ∧
    w4 = empty_pot_weight

theorem winnie_the_pooh_honey (empty_pot_weight : ℝ) 
  (h1 : empty_pot_weight = 200) : 
  ∃ (initial_weight : ℝ), 
    honey_pot initial_weight empty_pot_weight ∧ 
    initial_weight - empty_pot_weight = 3000 := by
  sorry

end NUMINAMATH_CALUDE_winnie_the_pooh_honey_l2618_261842


namespace NUMINAMATH_CALUDE_marble_probability_l2618_261829

/-- The probability of drawing a red, blue, or green marble from a bag -/
theorem marble_probability (red blue green yellow : ℕ) : 
  red = 5 → blue = 4 → green = 3 → yellow = 6 →
  (red + blue + green : ℚ) / (red + blue + green + yellow) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l2618_261829


namespace NUMINAMATH_CALUDE_expression_equals_one_l2618_261832

theorem expression_equals_one :
  (150^2 - 13^2) / (90^2 - 17^2) * ((90 - 17) * (90 + 17)) / ((150 - 13) * (150 + 13)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2618_261832


namespace NUMINAMATH_CALUDE_min_additional_flights_for_40_percent_rate_l2618_261885

/-- Calculates the on-time rate given the number of on-time flights and total flights -/
def onTimeRate (onTime : ℕ) (total : ℕ) : ℚ :=
  onTime / total

/-- Represents the airport's flight departure scenario -/
structure AirportScenario where
  lateFlights : ℕ
  initialOnTimeFlights : ℕ
  additionalOnTimeFlights : ℕ

/-- Theorem: At least 1 additional on-time flight is needed for the on-time rate to exceed 40% -/
theorem min_additional_flights_for_40_percent_rate 
  (scenario : AirportScenario) 
  (h1 : scenario.lateFlights = 1)
  (h2 : scenario.initialOnTimeFlights = 3) :
  (∀ x : ℕ, x < 1 → 
    onTimeRate (scenario.initialOnTimeFlights + x) 
               (scenario.lateFlights + scenario.initialOnTimeFlights + x) ≤ 2/5) ∧
  (onTimeRate (scenario.initialOnTimeFlights + 1) 
              (scenario.lateFlights + scenario.initialOnTimeFlights + 1) > 2/5) :=
by sorry

end NUMINAMATH_CALUDE_min_additional_flights_for_40_percent_rate_l2618_261885


namespace NUMINAMATH_CALUDE_cylinder_sphere_cone_volume_ratio_l2618_261891

/-- Given a cylinder with volume 128π cm³, the ratio of the volume of a sphere 
(with radius equal to the base radius of the cylinder) to the volume of a cone 
(with the same radius and height as the cylinder) is 2. -/
theorem cylinder_sphere_cone_volume_ratio : 
  ∀ (r h : ℝ), 
  r > 0 → h > 0 →
  π * r^2 * h = 128 * π →
  (4/3 * π * r^3) / (1/3 * π * r^2 * h) = 2 := by
sorry

end NUMINAMATH_CALUDE_cylinder_sphere_cone_volume_ratio_l2618_261891


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l2618_261890

theorem lcm_gcd_product (a b : ℕ) (h1 : a = 15) (h2 : b = 9) :
  Nat.lcm a b * Nat.gcd a b = 135 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l2618_261890


namespace NUMINAMATH_CALUDE_f_composition_equal_range_l2618_261852

/-- The function f(x) = x^2 + ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

/-- The theorem stating the range of a -/
theorem f_composition_equal_range (a : ℝ) :
  ({y | ∃ x, y = f a (f a x)} = {y | ∃ x, y = f a x}) →
  (a ≥ 4 ∨ a ≤ -2) :=
by sorry

end NUMINAMATH_CALUDE_f_composition_equal_range_l2618_261852


namespace NUMINAMATH_CALUDE_base_2_representation_315_l2618_261899

/-- Given a natural number n, returns the number of zeros in its binary representation -/
def count_zeros (n : ℕ) : ℕ := sorry

/-- Given a natural number n, returns the number of ones in its binary representation -/
def count_ones (n : ℕ) : ℕ := sorry

theorem base_2_representation_315 : 
  let x := count_zeros 315
  let y := count_ones 315
  y - x = 5 := by sorry

end NUMINAMATH_CALUDE_base_2_representation_315_l2618_261899


namespace NUMINAMATH_CALUDE_inequality_proof_l2618_261896

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a - b)^2 / (2 * (a + b)) ≤ Real.sqrt ((a^2 + b^2) / 2) - Real.sqrt (a * b) ∧
  Real.sqrt ((a^2 + b^2) / 2) - Real.sqrt (a * b) ≤ (a - b)^2 / (a + b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2618_261896


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2618_261897

/-- Given a hyperbola with center at the origin, one focus at (0, √3), and the distance between
    this focus and the nearest vertex being √3 - 1, prove that its equation is y² - x²/2 = 1 -/
theorem hyperbola_equation (F : ℝ × ℝ) (d : ℝ) :
  F = (0, Real.sqrt 3) →
  d = Real.sqrt 3 - 1 →
  ∀ (x y : ℝ), (y^2 : ℝ) - (x^2 / 2 : ℝ) = 1 ↔ 
    ((x, y) ∈ {p : ℝ × ℝ | ∃ (t : ℝ), 
      (p.1 - 0)^2 / ((Real.sqrt 3)^2 - 1) + 
      (p.2 - 0)^2 / ((Real.sqrt 3)^2 - 1 - 1) = 1}) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2618_261897


namespace NUMINAMATH_CALUDE_divisible_by_24_l2618_261858

theorem divisible_by_24 (n : ℕ+) : ∃ k : ℤ, (n + 7)^2 - (n - 5)^2 = 24 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_24_l2618_261858


namespace NUMINAMATH_CALUDE_area_of_triangle_ABF_l2618_261888

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the focus F
def F : ℝ × ℝ := (-2, 0)

-- Define the intersection line
def intersection_line (x : ℝ) : Prop := x = 2

-- Define the points A and B
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (2, -3)

-- Theorem statement
theorem area_of_triangle_ABF :
  hyperbola A.1 A.2 ∧
  hyperbola B.1 B.2 ∧
  intersection_line A.1 ∧
  intersection_line B.1 →
  (1/2 : ℝ) * |A.2 - B.2| * |A.1 - F.1| = 12 :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABF_l2618_261888


namespace NUMINAMATH_CALUDE_flower_planting_cost_l2618_261866

/-- The cost of planting and maintaining flowers with given items -/
theorem flower_planting_cost (flower_cost : ℚ) (h1 : flower_cost = 9) : ∃ total_cost : ℚ,
  let clay_pot_cost := flower_cost + 20
  let soil_cost := flower_cost - 2
  let fertilizer_cost := flower_cost * (1 + 1/2)
  let tools_cost := clay_pot_cost * (1 - 1/4)
  total_cost = flower_cost + clay_pot_cost + soil_cost + fertilizer_cost + tools_cost ∧ 
  total_cost = 80.25 := by
  sorry

end NUMINAMATH_CALUDE_flower_planting_cost_l2618_261866


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2618_261874

/-- Given a quadratic equation (k-1)x^2 + 6x + 9 = 0 with two equal real roots, prove that k = 2 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 6 * x + 9 = 0 ∧ 
   ∀ y : ℝ, (k - 1) * y^2 + 6 * y + 9 = 0 → y = x) → 
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2618_261874


namespace NUMINAMATH_CALUDE_simplify_power_product_l2618_261881

theorem simplify_power_product (x : ℝ) : (x^5 * x^3)^2 = x^16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_product_l2618_261881


namespace NUMINAMATH_CALUDE_volleyball_team_lineup_l2618_261864

/-- The number of players in the volleyball team -/
def total_players : ℕ := 16

/-- The number of triplets -/
def num_triplets : ℕ := 3

/-- The number of twins -/
def num_twins : ℕ := 2

/-- The number of starters to be chosen -/
def num_starters : ℕ := 7

/-- The number of valid starting lineups -/
def valid_lineups : ℕ := 9778

theorem volleyball_team_lineup :
  (Nat.choose total_players num_starters) -
  (Nat.choose (total_players - num_triplets) (num_starters - num_triplets)) -
  (Nat.choose (total_players - num_twins) (num_starters - num_twins)) +
  (Nat.choose (total_players - num_triplets - num_twins) (num_starters - num_triplets - num_twins)) =
  valid_lineups :=
sorry

end NUMINAMATH_CALUDE_volleyball_team_lineup_l2618_261864


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2618_261898

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x - 1)^2 + 2

-- Theorem statement
theorem quadratic_function_properties :
  (∃ (x : ℝ), f x = 2 ∧ ∀ (y : ℝ), f y ≤ f x) ∧ -- Maximum value is 2
  (∃ (x : ℝ), f x = x + 1) ∧ -- Vertex lies on y = x + 1
  (f 3 = -2) ∧ -- Passes through (3, -2)
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 →
    f x ≤ 2 ∧ -- Maximum value in [0, 3] is 2
    f x ≥ -2 ∧ -- Minimum value in [0, 3] is -2
    (f x = 2 → x = 1) ∧ -- Maximum occurs at x = 1
    (f x = -2 → x = 3)) -- Minimum occurs at x = 3
  := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2618_261898


namespace NUMINAMATH_CALUDE_triangle_angle_sum_max_l2618_261850

theorem triangle_angle_sum_max (A C : Real) (h1 : 0 < A) (h2 : A < 2 * π / 3) (h3 : A + C = 2 * π / 3) :
  let S := (Real.sqrt 3 / 3) * Real.sin A * Real.sin C
  ∃ (max_S : Real), ∀ (A' C' : Real), 
    0 < A' → A' < 2 * π / 3 → A' + C' = 2 * π / 3 → 
    (Real.sqrt 3 / 3) * Real.sin A' * Real.sin C' ≤ max_S ∧
    max_S = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_max_l2618_261850


namespace NUMINAMATH_CALUDE_number_calculation_l2618_261841

theorem number_calculation (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 10 → (40/100 : ℝ) * N = 120 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l2618_261841


namespace NUMINAMATH_CALUDE_car_count_on_river_road_l2618_261887

theorem car_count_on_river_road (buses cars bikes : ℕ) : 
  (3 : ℕ) * cars = 7 * buses →
  (3 : ℕ) * bikes = 10 * buses →
  cars = buses + 90 →
  bikes = buses + 140 →
  cars = 150 := by
sorry

end NUMINAMATH_CALUDE_car_count_on_river_road_l2618_261887


namespace NUMINAMATH_CALUDE_xiaozhao_journey_l2618_261814

def movements : List Int := [1000, -900, 700, -1200, 1200, 100, -1100, -200]

def calorie_per_km : Nat := 7000

def final_position (moves : List Int) : Int :=
  moves.sum

def total_distance (moves : List Int) : Nat :=
  moves.map (Int.natAbs) |>.sum

theorem xiaozhao_journey :
  let pos := final_position movements
  let dist := total_distance movements
  (pos < 0 ∧ pos.natAbs = 400) ∧
  (dist * calorie_per_km / 1000 = 44800) := by
  sorry

end NUMINAMATH_CALUDE_xiaozhao_journey_l2618_261814


namespace NUMINAMATH_CALUDE_largest_angle_is_right_angle_l2618_261845

/-- Given a triangle ABC with sides a, b, c and corresponding altitudes ha, hb, hc -/
theorem largest_angle_is_right_angle 
  (a b c ha hb hc : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ ha > 0 ∧ hb > 0 ∧ hc > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_volumes : (1 : ℝ) / (ha^2 * a)^2 = 1 / (hb^2 * b)^2 + 1 / (hc^2 * c)^2) :
  ∃ θ : ℝ, θ = Real.pi / 2 ∧ 
    θ = max (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) 
            (max (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) 
                 (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)))) := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_is_right_angle_l2618_261845


namespace NUMINAMATH_CALUDE_unique_original_message_exists_l2618_261800

/-- Represents a cryptogram as a list of characters -/
def Cryptogram := List Char

/-- Represents a bijective letter substitution -/
def Substitution := Char → Char

/-- The first cryptogram -/
def cryptogram1 : Cryptogram := 
  ['М', 'И', 'М', 'О', 'П', 'Р', 'А', 'С', 'Т', 'Е', 'Т', 'И', 'Р', 'А', 'С', 'И', 'С', 'П', 'Д', 'А', 'И', 'С', 'А', 'Ф', 'Е', 'И', 'И', 'Б', 'О', 'Е', 'Т', 'К', 'Ж', 'Р', 'Г', 'Л', 'Е', 'О', 'Л', 'О', 'И', 'Ш', 'И', 'С', 'А', 'Н', 'Н', 'С', 'Й', 'С', 'А', 'О', 'О', 'Л', 'Т', 'Л', 'Е', 'Я', 'Т', 'У', 'И', 'Ц', 'В', 'Ы', 'И', 'П', 'И', 'Я', 'Д', 'П', 'И', 'Щ', 'П', 'Ь', 'П', 'С', 'Е', 'Ю', 'Я', 'Я']

/-- The second cryptogram -/
def cryptogram2 : Cryptogram := 
  ['У', 'Щ', 'Ф', 'М', 'Ш', 'П', 'Д', 'Р', 'Е', 'Ц', 'Ч', 'Е', 'Ш', 'Ю', 'Ч', 'Д', 'А', 'К', 'Е', 'Ч', 'М', 'Д', 'В', 'К', 'Ш', 'Б', 'Е', 'Е', 'Ч', 'Д', 'Ф', 'Э', 'П', 'Й', 'Щ', 'Г', 'Ш', 'Ф', 'Щ', 'Ц', 'Е', 'Ю', 'Щ', 'Ф', 'П', 'М', 'Е', 'Ч', 'П', 'М', 'Р', 'Р', 'М', 'Е', 'О', 'Ч', 'Х', 'Е', 'Ш', 'Р', 'Т', 'Г', 'И', 'Ф', 'Р', 'С', 'Я', 'Ы', 'Л', 'К', 'Д', 'Ф', 'Ф', 'Е', 'Е']

/-- The original message -/
def original_message : Cryptogram := 
  ['Ш', 'Е', 'С', 'Т', 'А', 'Я', 'О', 'Л', 'И', 'М', 'П', 'И', 'А', 'Д', 'А', 'П', 'О', 'К', 'Р', 'И', 'П', 'Т', 'О', 'Г', 'Р', 'А', 'Ф', 'И', 'И', 'П', 'О', 'С', 'В', 'Я', 'Щ', 'Е', 'Н', 'А', 'С', 'Е', 'М', 'И', 'Д', 'Е', 'С', 'Я', 'Т', 'И', 'П', 'Я', 'Т', 'И', 'Л', 'Е', 'Т', 'И', 'Ю', 'С', 'П', 'Е', 'Ц', 'И', 'А', 'Л', 'Ь', 'Н', 'О', 'Й', 'С', 'Л', 'У', 'Ж', 'Б', 'Ы', 'Р', 'О', 'С', 'С', 'И', 'И']

/-- Predicate to check if a list is a permutation of another list -/
def is_permutation (l1 l2 : List α) : Prop := sorry

/-- Predicate to check if a function is bijective on a given list -/
def is_bijective_on (f : α → β) (l : List α) : Prop := sorry

/-- Main theorem: There exists a unique original message that satisfies the cryptogram conditions -/
theorem unique_original_message_exists : 
  ∃! (msg : Cryptogram), 
    (is_permutation msg cryptogram1) ∧ 
    (∃ (subst : Substitution), 
      (is_bijective_on subst msg) ∧ 
      (cryptogram2 = msg.map subst)) :=
sorry

end NUMINAMATH_CALUDE_unique_original_message_exists_l2618_261800


namespace NUMINAMATH_CALUDE_multiple_of_a_share_l2618_261826

def total_sum : ℚ := 427
def c_share : ℚ := 84

theorem multiple_of_a_share : ∃ (a b : ℚ) (x : ℚ), 
  a + b + c_share = total_sum ∧ 
  x * a = 4 * b ∧ 
  x * a = 7 * c_share ∧
  x = 3 := by sorry

end NUMINAMATH_CALUDE_multiple_of_a_share_l2618_261826


namespace NUMINAMATH_CALUDE_angle_in_third_quadrant_l2618_261849

theorem angle_in_third_quadrant (α : Real) : 
  (π / 2 < α ∧ α < π) → (π < π / 2 + α ∧ π / 2 + α < 3 * π / 2) := by
  sorry

end NUMINAMATH_CALUDE_angle_in_third_quadrant_l2618_261849


namespace NUMINAMATH_CALUDE_circle_m_range_l2618_261876

/-- A circle in the xy-plane can be represented by the equation x² + y² + dx + ey + f = 0,
    where d, e, and f are real constants, and d² + e² - 4f > 0 -/
def is_circle (d e f : ℝ) : Prop := d^2 + e^2 - 4*f > 0

/-- The equation x² + y² - 2x - 4y + m = 0 represents a circle -/
def represents_circle (m : ℝ) : Prop := is_circle (-2) (-4) m

theorem circle_m_range :
  ∀ m : ℝ, represents_circle m → m < 5 := by sorry

end NUMINAMATH_CALUDE_circle_m_range_l2618_261876


namespace NUMINAMATH_CALUDE_vacation_cost_l2618_261867

theorem vacation_cost (C : ℝ) :
  (C / 4 - C / 5 = 50) → C = 1000 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_l2618_261867


namespace NUMINAMATH_CALUDE_magazine_publication_theorem_l2618_261820

/-- Represents a magazine issue -/
structure Issue :=
  (year : ℕ)
  (month : ℕ)
  (exercisePosition : ℕ)
  (problemPosition : ℕ)

/-- The publication schedule of the magazine -/
def publicationSchedule : 
  (exercisesPerIssue : ℕ) → 
  (problemsPerIssue : ℕ) → 
  (issuesPerYear : ℕ) → 
  (startYear : ℕ) → 
  (lastExerciseNumber : ℕ) → 
  (lastProblemNumber : ℕ) → 
  (Prop) :=
  λ exercisesPerIssue problemsPerIssue issuesPerYear startYear lastExerciseNumber lastProblemNumber =>
    ∃ (exerciseIssue problemIssue : Issue),
      -- The exercise issue is in 1979, 3rd month, 2nd exercise
      exerciseIssue.year = 1979 ∧
      exerciseIssue.month = 3 ∧
      exerciseIssue.exercisePosition = 2 ∧
      -- The problem issue is in 1973, 5th month, 5th problem
      problemIssue.year = 1973 ∧
      problemIssue.month = 5 ∧
      problemIssue.problemPosition = 5 ∧
      -- The serial numbers match the respective years
      (lastExerciseNumber + (exerciseIssue.year - startYear) * exercisesPerIssue * issuesPerYear + 
       (exerciseIssue.month - 1) * exercisesPerIssue + exerciseIssue.exercisePosition = exerciseIssue.year) ∧
      (lastProblemNumber + (problemIssue.year - startYear) * problemsPerIssue * issuesPerYear + 
       (problemIssue.month - 1) * problemsPerIssue + problemIssue.problemPosition = problemIssue.year)

theorem magazine_publication_theorem :
  publicationSchedule 8 8 9 1967 1169 1576 :=
by
  sorry


end NUMINAMATH_CALUDE_magazine_publication_theorem_l2618_261820


namespace NUMINAMATH_CALUDE_smallest_value_w3_plus_z3_l2618_261840

theorem smallest_value_w3_plus_z3 (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2)
  (h2 : Complex.abs (w^2 + z^2) = 18) :
  Complex.abs (w^3 + z^3) ≥ 50 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_w3_plus_z3_l2618_261840


namespace NUMINAMATH_CALUDE_perpendicular_slope_l2618_261803

/-- Given a line with equation 4x - 5y = 20, the slope of the perpendicular line is -5/4 -/
theorem perpendicular_slope (x y : ℝ) :
  (4 * x - 5 * y = 20) → 
  ∃ (m : ℝ), m = -5/4 ∧ m * (4/5) = -1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l2618_261803


namespace NUMINAMATH_CALUDE_complex_multiplication_l2618_261812

def i : ℂ := Complex.I

theorem complex_multiplication :
  (1 + i) * (3 - i) = 4 + 2*i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2618_261812


namespace NUMINAMATH_CALUDE_tan_triple_angle_l2618_261873

theorem tan_triple_angle (α : Real) (P : ℝ × ℝ) :
  α > 0 ∧ α < π / 2 →  -- α is acute
  P.1 = 2 * (Real.cos (280 * π / 180))^2 →  -- x-coordinate of P
  P.2 = Real.sin (20 * π / 180) →  -- y-coordinate of P
  Real.tan (3 * α) = Real.sqrt 3 := by
    sorry

end NUMINAMATH_CALUDE_tan_triple_angle_l2618_261873


namespace NUMINAMATH_CALUDE_max_distance_trig_points_l2618_261886

theorem max_distance_trig_points (α β : ℝ) : 
  let P := (Real.cos α, Real.sin α)
  let Q := (Real.cos β, Real.sin β)
  ∃ (max_dist : ℝ), max_dist = 2 ∧ ∀ (α' β' : ℝ), 
    let P' := (Real.cos α', Real.sin α')
    let Q' := (Real.cos β', Real.sin β')
    Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_trig_points_l2618_261886


namespace NUMINAMATH_CALUDE_red_pens_per_student_red_pens_calculation_l2618_261895

theorem red_pens_per_student (students : ℕ) (black_pens_per_student : ℕ) 
  (pens_taken_first_month : ℕ) (pens_taken_second_month : ℕ) 
  (remaining_pens_per_student : ℕ) : ℕ :=
  let total_black_pens := students * black_pens_per_student
  let total_pens_taken := pens_taken_first_month + pens_taken_second_month
  let total_remaining_pens := students * remaining_pens_per_student
  let initial_total_pens := total_pens_taken + total_remaining_pens
  let total_red_pens := initial_total_pens - total_black_pens
  total_red_pens / students

theorem red_pens_calculation :
  red_pens_per_student 3 43 37 41 79 = 62 := by
  sorry

end NUMINAMATH_CALUDE_red_pens_per_student_red_pens_calculation_l2618_261895


namespace NUMINAMATH_CALUDE_mayoral_election_vote_ratio_l2618_261831

theorem mayoral_election_vote_ratio :
  let votes_Z : ℕ := 25000
  let votes_X : ℕ := 22500
  let votes_Y : ℕ := (3 * votes_Z) / 5
  (votes_X - votes_Y) * 2 = votes_Y := by
  sorry

end NUMINAMATH_CALUDE_mayoral_election_vote_ratio_l2618_261831


namespace NUMINAMATH_CALUDE_marias_number_problem_l2618_261844

theorem marias_number_problem (n : ℚ) : 
  (((n + 3) * 3 - 2) / 3 = 10) → (n = 23 / 3) := by
  sorry

end NUMINAMATH_CALUDE_marias_number_problem_l2618_261844


namespace NUMINAMATH_CALUDE_percent_to_decimal_three_percent_to_decimal_l2618_261804

theorem percent_to_decimal (p : ℚ) : p / 100 = p * (1 / 100) := by sorry

theorem three_percent_to_decimal : (3 : ℚ) / 100 = 0.03 := by sorry

end NUMINAMATH_CALUDE_percent_to_decimal_three_percent_to_decimal_l2618_261804


namespace NUMINAMATH_CALUDE_y_power_neg_x_value_l2618_261859

theorem y_power_neg_x_value (x y : ℝ) (h : |y - 2*x| + (x + y - 3)^2 = 0) : y^(-x) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_y_power_neg_x_value_l2618_261859


namespace NUMINAMATH_CALUDE_no_discriminant_for_quartic_l2618_261894

theorem no_discriminant_for_quartic (P : ℝ → ℝ → ℝ → ℝ → ℝ) :
  ∃ (a b c d : ℝ),
    (∃ (r₁ r₂ r₃ r₄ : ℝ), ∀ (x : ℝ),
      x^4 + a*x^3 + b*x^2 + c*x + d = (x - r₁) * (x - r₂) * (x - r₃) * (x - r₄) ∧
      P a b c d < 0) ∨
    ((¬ ∃ (r₁ r₂ r₃ r₄ : ℝ), ∀ (x : ℝ),
      x^4 + a*x^3 + b*x^2 + c*x + d = (x - r₁) * (x - r₂) * (x - r₃) * (x - r₄)) ∧
      P a b c d ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_discriminant_for_quartic_l2618_261894


namespace NUMINAMATH_CALUDE_bus_capacity_is_198_l2618_261877

/-- Represents the capacity of a double-decker bus -/
def BusCapacity : ℕ :=
  let lower_left := 15 * 3
  let lower_right := (15 - 3) * 3
  let lower_back := 11
  let lower_standing := 12
  let upper_left := 20 * 2
  let upper_right_regular := (18 - 5) * 2
  let upper_right_reserved := 5 * 4
  let upper_standing := 8
  lower_left + lower_right + lower_back + lower_standing +
  upper_left + upper_right_regular + upper_right_reserved + upper_standing

/-- Theorem stating that the bus capacity is 198 people -/
theorem bus_capacity_is_198 : BusCapacity = 198 := by
  sorry

#eval BusCapacity

end NUMINAMATH_CALUDE_bus_capacity_is_198_l2618_261877


namespace NUMINAMATH_CALUDE_sin_2010th_derivative_l2618_261853

open Real

-- Define the recursive function for the nth derivative of sin x
noncomputable def f (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => sin
  | n + 1 => deriv (f n)

-- State the theorem
theorem sin_2010th_derivative :
  ∀ x, f 2010 x = -sin x :=
by
  sorry

end NUMINAMATH_CALUDE_sin_2010th_derivative_l2618_261853


namespace NUMINAMATH_CALUDE_new_rectangle_area_l2618_261843

/-- Given a rectangle with sides 3 and 4, prove that a new rectangle
    formed with one side equal to the diagonal of the original rectangle
    and the other side equal to the sum of the original sides has an area of 35. -/
theorem new_rectangle_area (a b : ℝ) (ha : a = 3) (hb : b = 4) :
  let d := Real.sqrt (a^2 + b^2)
  let new_side_sum := a + b
  d * new_side_sum = 35 := by sorry

end NUMINAMATH_CALUDE_new_rectangle_area_l2618_261843


namespace NUMINAMATH_CALUDE_minimize_z_l2618_261828

-- Define the function z
def z (x a b c d : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + c*(x - a) + d*(x - b)

-- Theorem statement
theorem minimize_z (a b c d : ℝ) :
  ∃ x : ℝ, ∀ y : ℝ, z x a b c d ≤ z y a b c d ∧ x = (2*(a+b) - (c+d)) / 4 :=
sorry

end NUMINAMATH_CALUDE_minimize_z_l2618_261828


namespace NUMINAMATH_CALUDE_original_lines_per_sheet_l2618_261836

/-- Represents the number of lines on each sheet in the original report -/
def L : ℕ := 56

/-- The number of sheets in the original report -/
def original_sheets : ℕ := 20

/-- The number of characters per line in the original report -/
def original_chars_per_line : ℕ := 65

/-- The number of lines per sheet in the retyped report -/
def new_lines_per_sheet : ℕ := 65

/-- The number of characters per line in the retyped report -/
def new_chars_per_line : ℕ := 70

/-- The percentage reduction in the number of sheets -/
def reduction_percentage : ℚ := 20 / 100

theorem original_lines_per_sheet :
  L = 56 ∧
  original_sheets * L * original_chars_per_line = 
    (original_sheets * (1 - reduction_percentage)).floor * new_lines_per_sheet * new_chars_per_line :=
by sorry

end NUMINAMATH_CALUDE_original_lines_per_sheet_l2618_261836


namespace NUMINAMATH_CALUDE_flyers_left_proof_l2618_261848

/-- The number of flyers left after Jack and Rose hand out some flyers -/
def flyers_left (initial : ℕ) (jack_handed : ℕ) (rose_handed : ℕ) : ℕ :=
  initial - (jack_handed + rose_handed)

/-- Proof that given 1,236 initial flyers, with Jack handing out 120 flyers
    and Rose handing out 320 flyers, the number of flyers left is 796 -/
theorem flyers_left_proof :
  flyers_left 1236 120 320 = 796 := by
  sorry

end NUMINAMATH_CALUDE_flyers_left_proof_l2618_261848


namespace NUMINAMATH_CALUDE_courtyard_tile_cost_l2618_261818

/-- Calculates the total cost of tiles for a courtyard --/
def total_tile_cost (length width : ℝ) (tiles_per_sqft : ℝ) 
  (green_tile_percentage : ℝ) (green_tile_cost red_tile_cost : ℝ) : ℝ :=
  let area := length * width
  let total_tiles := area * tiles_per_sqft
  let green_tiles := total_tiles * green_tile_percentage
  let red_tiles := total_tiles - green_tiles
  (green_tiles * green_tile_cost) + (red_tiles * red_tile_cost)

/-- Theorem stating the total cost of tiles for the given courtyard specifications --/
theorem courtyard_tile_cost :
  total_tile_cost 10 25 4 0.4 3 1.5 = 2100 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_tile_cost_l2618_261818


namespace NUMINAMATH_CALUDE_john_total_spend_l2618_261851

-- Define the prices and quantities
def tshirt_price : ℝ := 20
def tshirt_quantity : ℕ := 3
def pants_price : ℝ := 50
def pants_quantity : ℕ := 2
def jacket_original_price : ℝ := 80
def jacket_discount : ℝ := 0.25
def hat_price : ℝ := 15
def shoes_original_price : ℝ := 60
def shoes_discount : ℝ := 0.10

-- Define the total cost function
def total_cost : ℝ :=
  (tshirt_price * tshirt_quantity) +
  (pants_price * pants_quantity) +
  (jacket_original_price * (1 - jacket_discount)) +
  hat_price +
  (shoes_original_price * (1 - shoes_discount))

-- Theorem to prove
theorem john_total_spend : total_cost = 289 := by
  sorry

end NUMINAMATH_CALUDE_john_total_spend_l2618_261851


namespace NUMINAMATH_CALUDE_janet_movie_cost_l2618_261834

/-- Calculates the total cost of filming Janet's newest movie given the following conditions:
  * Janet's previous movie was 2 hours long
  * The new movie is 60% longer than the previous movie
  * The previous movie cost $50 per minute to film
  * The new movie cost twice as much per minute to film as the previous movie
-/
def total_cost_newest_movie (previous_movie_length : Real) 
                            (length_increase_percent : Real)
                            (previous_cost_per_minute : Real)
                            (new_cost_multiplier : Real) : Real :=
  let new_movie_length := previous_movie_length * (1 + length_increase_percent)
  let new_movie_length_minutes := new_movie_length * 60
  let new_cost_per_minute := previous_cost_per_minute * new_cost_multiplier
  new_movie_length_minutes * new_cost_per_minute

theorem janet_movie_cost :
  total_cost_newest_movie 2 0.6 50 2 = 19200 := by
  sorry

end NUMINAMATH_CALUDE_janet_movie_cost_l2618_261834


namespace NUMINAMATH_CALUDE_triangle_angle_side_inequality_l2618_261835

/-- Theorem: For any triangle, the weighted sum of angles divided by the sum of sides 
    is bounded between π/3 and π/2 -/
theorem triangle_angle_side_inequality (A B C a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- sides are positive
  a + b > c ∧ b + c > a ∧ c + a > b →  -- triangle inequality
  A + B + C = π →  -- sum of angles
  0 < A ∧ 0 < B ∧ 0 < C →  -- angles are positive
  π / 3 ≤ (A * a + B * b + C * c) / (a + b + c) ∧ 
  (A * a + B * b + C * c) / (a + b + c) < π / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_side_inequality_l2618_261835


namespace NUMINAMATH_CALUDE_max_sum_given_product_l2618_261871

theorem max_sum_given_product (a b : ℤ) (h : a * b = -72) : 
  (∀ (x y : ℤ), x * y = -72 → x + y ≤ a + b) → a + b = 71 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_product_l2618_261871


namespace NUMINAMATH_CALUDE_soda_lasts_40_days_l2618_261838

/-- The number of days soda bottles last given the initial quantity and daily consumption rate -/
def soda_duration (total_bottles : ℕ) (daily_consumption : ℕ) : ℕ :=
  total_bottles / daily_consumption

theorem soda_lasts_40_days :
  soda_duration 360 9 = 40 := by
  sorry

end NUMINAMATH_CALUDE_soda_lasts_40_days_l2618_261838


namespace NUMINAMATH_CALUDE_hawks_score_l2618_261889

/-- Calculates the total score for a team given the number of touchdowns and points per touchdown -/
def totalScore (touchdowns : ℕ) (pointsPerTouchdown : ℕ) : ℕ :=
  touchdowns * pointsPerTouchdown

/-- Theorem: If a team scores 3 touchdowns, and each touchdown is worth 7 points, then the team's total score is 21 points -/
theorem hawks_score :
  totalScore 3 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_hawks_score_l2618_261889


namespace NUMINAMATH_CALUDE_polynomial_equation_sum_l2618_261863

theorem polynomial_equation_sum (a b c d : ℤ) : 
  (∀ x, (x^2 + a*x + b) * (x + c) = x^3 + 5*x^2 - 6*x - 4) → 
  a + b + c + d = 11 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equation_sum_l2618_261863


namespace NUMINAMATH_CALUDE_single_element_condition_intersection_condition_l2618_261856

-- Define set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * x + 3 = 0}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 2 * x - 3 = 0}

-- Theorem for the first part of the problem
theorem single_element_condition (a : ℝ) :
  (∃! x, x ∈ A a) ↔ (a = 0 ∨ a = 1/3) := by sorry

-- Theorem for the second part of the problem
theorem intersection_condition (a : ℝ) :
  A a ∩ B = A a ↔ (a > 1/3 ∨ a = -1) := by sorry

end NUMINAMATH_CALUDE_single_element_condition_intersection_condition_l2618_261856


namespace NUMINAMATH_CALUDE_minor_arc_circumference_l2618_261821

theorem minor_arc_circumference (r : ℝ) (θ : ℝ) (h_r : r = 12) (h_θ : θ = 110 * π / 180) :
  let circle_circumference := 2 * π * r
  let arc_length := circle_circumference * θ / (2 * π)
  arc_length = 22 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_minor_arc_circumference_l2618_261821


namespace NUMINAMATH_CALUDE_odd_sum_of_squares_implies_odd_product_l2618_261870

theorem odd_sum_of_squares_implies_odd_product (n m : ℤ) 
  (h : Odd (n^2 + m^2)) : Odd (n * m) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_of_squares_implies_odd_product_l2618_261870


namespace NUMINAMATH_CALUDE_candy_distribution_l2618_261811

theorem candy_distribution (total_candies : ℕ) (num_friends : ℕ) 
  (h1 : total_candies = 27) (h2 : num_friends = 5) :
  total_candies % num_friends = 
    (total_candies - (total_candies / num_friends) * num_friends) := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2618_261811


namespace NUMINAMATH_CALUDE_rational_numbers_closed_l2618_261817

-- Define the set of rational numbers
def RationalNumbers : Set ℚ := {x | ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}

-- State the theorem
theorem rational_numbers_closed :
  (∀ (a b : ℚ), a ∈ RationalNumbers → b ∈ RationalNumbers → (a + b) ∈ RationalNumbers) ∧
  (∀ (a b : ℚ), a ∈ RationalNumbers → b ∈ RationalNumbers → (a - b) ∈ RationalNumbers) ∧
  (∀ (a b : ℚ), a ∈ RationalNumbers → b ∈ RationalNumbers → (a * b) ∈ RationalNumbers) ∧
  (∀ (a b : ℚ), a ∈ RationalNumbers → b ∈ RationalNumbers → b ≠ 0 → (a / b) ∈ RationalNumbers) :=
by sorry

end NUMINAMATH_CALUDE_rational_numbers_closed_l2618_261817


namespace NUMINAMATH_CALUDE_function_properties_a_value_l2618_261884

noncomputable section

-- Define the natural exponential function
def exp (x : ℝ) := Real.exp x

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a * exp x - a - x) * exp x

theorem function_properties (h : ∀ x : ℝ, f 1 x ≥ 0) :
  (∃! x₀ : ℝ, ∀ x : ℝ, f 1 x ≤ f 1 x₀) ∧
  (∃ x₀ : ℝ, ∀ x : ℝ, f 1 x ≤ f 1 x₀ ∧ 0 < f 1 x₀ ∧ f 1 x₀ < 1/4) :=
sorry

theorem a_value (h : ∀ a : ℝ, a ≥ 0 → ∀ x : ℝ, f a x ≥ 0) :
  ∃! a : ℝ, a = 1 ∧ ∀ x : ℝ, f a x ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_function_properties_a_value_l2618_261884


namespace NUMINAMATH_CALUDE_triangle_angle_B_l2618_261862

/-- Given a triangle ABC with side lengths a and c, and angle A, proves that angle B has two possible values. -/
theorem triangle_angle_B (a c : ℝ) (A : ℝ) (h1 : a = 5 * Real.sqrt 2) (h2 : c = 10) (h3 : A = π / 6) :
  ∃ (B : ℝ), (B = π * 7 / 12 ∨ B = π / 12) := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_B_l2618_261862


namespace NUMINAMATH_CALUDE_angle_relationship_l2618_261837

theorem angle_relationship (angle1 angle2 angle3 angle4 : Real) :
  (angle1 + angle2 = 90) →  -- angle1 and angle2 are complementary
  (angle3 + angle4 = 180) →  -- angle3 and angle4 are supplementary
  (angle1 = angle3) →  -- angle1 equals angle3
  (angle2 + 90 = angle4) :=  -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_angle_relationship_l2618_261837


namespace NUMINAMATH_CALUDE_prob_blue_or_green_l2618_261869

def cube_prob (blue_faces green_faces red_faces : ℕ) : ℚ :=
  (blue_faces + green_faces : ℚ) / (blue_faces + green_faces + red_faces)

theorem prob_blue_or_green : 
  cube_prob 3 1 2 = 2/3 := by sorry

end NUMINAMATH_CALUDE_prob_blue_or_green_l2618_261869


namespace NUMINAMATH_CALUDE_wage_productivity_relationship_l2618_261839

/-- Represents the regression line equation for worker's wage and labor productivity -/
def regression_line (x : ℝ) : ℝ := 50 + 80 * x

/-- Theorem stating the relationship between changes in labor productivity and worker's wage -/
theorem wage_productivity_relationship :
  ∀ x : ℝ, regression_line (x + 1) - regression_line x = 80 := by
  sorry

end NUMINAMATH_CALUDE_wage_productivity_relationship_l2618_261839


namespace NUMINAMATH_CALUDE_two_a_plus_b_value_l2618_261857

theorem two_a_plus_b_value (a b : ℚ) 
  (eq1 : 3 * a - b = 8) 
  (eq2 : 4 * b + 7 * a = 13) : 
  2 * a + b = 73 / 19 := by
sorry

end NUMINAMATH_CALUDE_two_a_plus_b_value_l2618_261857


namespace NUMINAMATH_CALUDE_plane_equation_from_point_and_normal_specific_plane_equation_l2618_261833

/-- Given a point M and a normal vector N, this theorem states that
    the equation Ax + By + Cz + D = 0 represents a plane passing through M
    and perpendicular to N, where (A, B, C) are the components of N. -/
theorem plane_equation_from_point_and_normal (M : ℝ × ℝ × ℝ) (N : ℝ × ℝ × ℝ) :
  let (x₀, y₀, z₀) := M
  let (A, B, C) := N
  let D := -(A * x₀ + B * y₀ + C * z₀)
  ∀ (x y z : ℝ), A * x + B * y + C * z + D = 0 ↔
    ((x - x₀) * A + (y - y₀) * B + (z - z₀) * C = 0 ∧
     ∃ (t : ℝ), x - x₀ = t * A ∧ y - y₀ = t * B ∧ z - z₀ = t * C) :=
by sorry

/-- The equation 4x + 3y + 2z - 27 = 0 represents a plane that passes through
    the point (2, 3, 5) and is perpendicular to the vector (4, 3, 2). -/
theorem specific_plane_equation :
  let M : ℝ × ℝ × ℝ := (2, 3, 5)
  let N : ℝ × ℝ × ℝ := (4, 3, 2)
  ∀ (x y z : ℝ), 4 * x + 3 * y + 2 * z - 27 = 0 ↔
    ((x - 2) * 4 + (y - 3) * 3 + (z - 5) * 2 = 0 ∧
     ∃ (t : ℝ), x - 2 = t * 4 ∧ y - 3 = t * 3 ∧ z - 5 = t * 2) :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_from_point_and_normal_specific_plane_equation_l2618_261833


namespace NUMINAMATH_CALUDE_g_increasing_g_geq_h_condition_l2618_261819

noncomputable section

-- Define the functions g and h
def g (a : ℝ) (x : ℝ) : ℝ := a * x - a / x - 5 * Real.log x
def h (m : ℝ) (x : ℝ) : ℝ := x^2 - m * x + 4

-- Theorem 1: g(x) is increasing when a > 5/2
theorem g_increasing (a : ℝ) : 
  (∀ x > 0, ∀ y > 0, x < y → g a x < g a y) ↔ a > 5/2 :=
sorry

-- Theorem 2: Condition for g(x₁) ≥ h(x₂) when a = 2
theorem g_geq_h_condition (m : ℝ) :
  (∃ x₁ ∈ Set.Ioo 0 1, ∀ x₂ ∈ Set.Icc 1 2, g 2 x₁ ≥ h m x₂) ↔ 
  m ≥ 8 - 5 * Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_g_increasing_g_geq_h_condition_l2618_261819


namespace NUMINAMATH_CALUDE_handshake_count_l2618_261822

/-- Represents a gathering of married couples -/
structure Gathering where
  couples : ℕ
  no_male_handshakes : Bool

/-- Calculates the total number of handshakes in the gathering -/
def total_handshakes (g : Gathering) : ℕ :=
  let women := g.couples
  let men := g.couples
  let women_handshakes := women.choose 2
  let men_women_handshakes := men * (women - 1)
  women_handshakes + men_women_handshakes

/-- Theorem stating that in a gathering of 15 married couples with the given conditions, 
    the total number of handshakes is 315 -/
theorem handshake_count (g : Gathering) :
  g.couples = 15 ∧ g.no_male_handshakes = true → total_handshakes g = 315 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l2618_261822


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2618_261855

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 2 / a 1

theorem geometric_sequence_property (a : ℕ → ℝ) (h1 : geometric_sequence a) 
  (h2 : a 2 ^ 2 + 6 * a 2 + 4 = 0) (h3 : a 18 ^ 2 + 6 * a 18 + 4 = 0) :
  a 4 * a 16 + a 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2618_261855


namespace NUMINAMATH_CALUDE_line_contains_point_l2618_261806

theorem line_contains_point (j : ℝ) : 
  (∀ x y : ℝ, -2 - 3*j*x = 7*y → x = 1/3 ∧ y = -3) → j = 19 := by
  sorry

end NUMINAMATH_CALUDE_line_contains_point_l2618_261806


namespace NUMINAMATH_CALUDE_ghee_mixture_quantity_l2618_261872

theorem ghee_mixture_quantity (original_ghee_percent : Real) 
                               (original_vanaspati_percent : Real)
                               (original_palm_oil_percent : Real)
                               (added_ghee : Real)
                               (added_palm_oil : Real)
                               (final_vanaspati_percent : Real) :
  original_ghee_percent = 0.55 →
  original_vanaspati_percent = 0.35 →
  original_palm_oil_percent = 0.10 →
  added_ghee = 15 →
  added_palm_oil = 5 →
  final_vanaspati_percent = 0.30 →
  ∃ (original_quantity : Real),
    original_quantity = 120 ∧
    original_vanaspati_percent * original_quantity = 
      final_vanaspati_percent * (original_quantity + added_ghee + added_palm_oil) :=
by sorry

end NUMINAMATH_CALUDE_ghee_mixture_quantity_l2618_261872


namespace NUMINAMATH_CALUDE_relay_race_total_time_l2618_261846

/-- The time taken by four athletes to complete a relay race -/
def relay_race_time (athlete1_time : ℕ) : ℕ :=
  let athlete2_time := athlete1_time + 10
  let athlete3_time := athlete2_time - 15
  let athlete4_time := athlete1_time - 25
  athlete1_time + athlete2_time + athlete3_time + athlete4_time

/-- Theorem stating that the total time for the relay race is 200 seconds -/
theorem relay_race_total_time : relay_race_time 55 = 200 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_total_time_l2618_261846
