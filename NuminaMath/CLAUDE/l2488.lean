import Mathlib

namespace NUMINAMATH_CALUDE_tan_a_pi_third_equals_sqrt_three_l2488_248847

-- Define the function for logarithm with base a
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem tan_a_pi_third_equals_sqrt_three 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : log_base a 16 = 2) : 
  Real.tan (a * π / 3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_a_pi_third_equals_sqrt_three_l2488_248847


namespace NUMINAMATH_CALUDE_decagon_adjacent_probability_l2488_248895

/-- A decagon is a polygon with 10 sides and vertices -/
def Decagon : Type := Unit

/-- Two vertices in a polygon are adjacent if they share an edge -/
def adjacent (v1 v2 : ℕ) (p : Decagon) : Prop := sorry

/-- The probability of an event is the number of favorable outcomes divided by the total number of possible outcomes -/
def probability (event total : ℕ) : ℚ := event / total

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem decagon_adjacent_probability :
  ∀ (d : Decagon),
  probability 
    (10 : ℕ)  -- Number of adjacent vertex pairs
    (choose_two 10)  -- Total number of vertex pairs
  = 2 / 9 := by sorry

end NUMINAMATH_CALUDE_decagon_adjacent_probability_l2488_248895


namespace NUMINAMATH_CALUDE_solve_system_l2488_248894

theorem solve_system (a b : ℚ) (h1 : a + a/4 = 3) (h2 : b - 2*a = 1) : a = 12/5 ∧ b = 29/5 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2488_248894


namespace NUMINAMATH_CALUDE_inequality_solution_l2488_248874

theorem inequality_solution (x : ℝ) : x^2 - x - 5 > 3*x ↔ x > 5 ∨ x < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2488_248874


namespace NUMINAMATH_CALUDE_intersection_when_m_3_intersection_equals_B_l2488_248819

def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}

def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 3 * m - 2}

theorem intersection_when_m_3 : A ∩ B 3 = {x | 2 ≤ x ∧ x ≤ 4} := by sorry

theorem intersection_equals_B (m : ℝ) : A ∩ B m = B m ↔ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_3_intersection_equals_B_l2488_248819


namespace NUMINAMATH_CALUDE_chess_team_arrangements_l2488_248803

/-- Represents the number of boys on the chess team -/
def num_boys : ℕ := 3

/-- Represents the number of girls on the chess team -/
def num_girls : ℕ := 3

/-- Represents the total number of team members -/
def total_members : ℕ := num_boys + num_girls

/-- Represents the number of positions in the middle of the row -/
def middle_positions : ℕ := total_members - 2

/-- Calculates the number of ways to arrange girls at the ends -/
def end_arrangements : ℕ := num_girls * (num_girls - 1)

/-- Calculates the number of ways to arrange the middle positions -/
def middle_arrangements : ℕ := Nat.factorial middle_positions

/-- Theorem: The total number of possible arrangements is 144 -/
theorem chess_team_arrangements :
  end_arrangements * middle_arrangements = 144 := by
  sorry


end NUMINAMATH_CALUDE_chess_team_arrangements_l2488_248803


namespace NUMINAMATH_CALUDE_expand_product_l2488_248851

theorem expand_product (x y : ℝ) : (x + 3) * (x + 2*y + 4) = x^2 + 7*x + 2*x*y + 6*y + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2488_248851


namespace NUMINAMATH_CALUDE_power_of_two_equality_l2488_248802

theorem power_of_two_equality : ∃ y : ℕ, 8^3 + 8^3 + 8^3 + 8^3 = 2^y ∧ y = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l2488_248802


namespace NUMINAMATH_CALUDE_min_value_polynomial_min_value_achieved_l2488_248880

theorem min_value_polynomial (x : ℝ) : 
  (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2164 ≥ 2161.75 :=
sorry

theorem min_value_achieved : 
  ∃ x : ℝ, (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2164 = 2161.75 :=
sorry

end NUMINAMATH_CALUDE_min_value_polynomial_min_value_achieved_l2488_248880


namespace NUMINAMATH_CALUDE_hall_length_is_36_meters_l2488_248844

-- Define the hall dimensions
def hall_width : ℝ := 15

-- Define the stone dimensions in meters
def stone_length : ℝ := 0.6  -- 6 dm = 0.6 m
def stone_width : ℝ := 0.5   -- 5 dm = 0.5 m

-- Define the number of stones
def num_stones : ℕ := 1800

-- Theorem stating the length of the hall
theorem hall_length_is_36_meters :
  let total_area := (↑num_stones : ℝ) * stone_length * stone_width
  let hall_length := total_area / hall_width
  hall_length = 36 := by sorry

end NUMINAMATH_CALUDE_hall_length_is_36_meters_l2488_248844


namespace NUMINAMATH_CALUDE_find_a_and_b_l2488_248898

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2008*x - 2009 > 0}
def N (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem find_a_and_b :
  ∃ (a b : ℝ), 
    (M ∪ N a b = Set.univ) ∧ 
    (M ∩ N a b = Set.Ioc 2009 2010) ∧ 
    a = 2009 ∧ 
    b = -2009 * 2010 := by
  sorry

end NUMINAMATH_CALUDE_find_a_and_b_l2488_248898


namespace NUMINAMATH_CALUDE_lines_concurrent_at_S_l2488_248877

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents a plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Tetrahedron SABC with points A', B', C' on edges SA, SB, SC respectively -/
structure Tetrahedron where
  S : Point3D
  A : Point3D
  B : Point3D
  C : Point3D
  A' : Point3D
  B' : Point3D
  C' : Point3D

/-- The intersection line d of planes ABC and A'B'C' -/
def intersection_line (t : Tetrahedron) : Line3D :=
  sorry

/-- Theorem: Lines AA', BB', CC' are concurrent at S for any rotation of A'B'C' around d -/
theorem lines_concurrent_at_S (t : Tetrahedron) (θ : ℝ) : 
  ∃ (S : Point3D), 
    (Line3D.mk t.A t.A').point = S ∧ 
    (Line3D.mk t.B t.B').point = S ∧ 
    (Line3D.mk t.C t.C').point = S := by
  sorry

end NUMINAMATH_CALUDE_lines_concurrent_at_S_l2488_248877


namespace NUMINAMATH_CALUDE_parking_cost_is_10_l2488_248810

-- Define the given conditions
def saved : ℕ := 28
def entry_cost : ℕ := 55
def meal_pass_cost : ℕ := 25
def distance : ℕ := 165
def fuel_efficiency : ℕ := 30
def gas_price : ℕ := 3
def additional_savings : ℕ := 95

-- Define the function to calculate parking cost
def parking_cost : ℕ :=
  let total_needed := saved + additional_savings
  let round_trip_distance := 2 * distance
  let gas_needed := round_trip_distance / fuel_efficiency
  let gas_cost := gas_needed * gas_price
  let total_cost_without_parking := gas_cost + entry_cost + meal_pass_cost
  total_needed - total_cost_without_parking

-- Theorem to prove
theorem parking_cost_is_10 : parking_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_parking_cost_is_10_l2488_248810


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2488_248863

theorem min_value_quadratic : 
  ∃ (min : ℝ), min = -39 ∧ ∀ (x : ℝ), x^2 + 14*x + 10 ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2488_248863


namespace NUMINAMATH_CALUDE_equation_equivalence_l2488_248893

theorem equation_equivalence (x y : ℝ) (h : y = x + 1/x) :
  x^4 + x^3 - 3*x^2 + x + 2 = 0 ↔ x^2 * (y^2 + y - 5) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2488_248893


namespace NUMINAMATH_CALUDE_blueberry_pie_count_l2488_248830

theorem blueberry_pie_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) 
  (h_total : total_pies = 30)
  (h_ratio : apple_ratio + blueberry_ratio + cherry_ratio = 10)
  (h_apple : apple_ratio = 2)
  (h_blueberry : blueberry_ratio = 3)
  (h_cherry : cherry_ratio = 5) :
  (total_pies / (apple_ratio + blueberry_ratio + cherry_ratio)) * blueberry_ratio = 9 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_pie_count_l2488_248830


namespace NUMINAMATH_CALUDE_carson_age_carson_age_real_l2488_248859

/-- Given the ages of Aunt Anna, Maria, and Carson, prove Carson's age -/
theorem carson_age (anna_age : ℕ) (maria_age : ℕ) (carson_age : ℕ) : 
  anna_age = 60 →
  maria_age = 2 * anna_age / 3 →
  carson_age = maria_age - 7 →
  carson_age = 33 := by
sorry

/-- Alternative formulation using real numbers for more precise calculations -/
theorem carson_age_real (anna_age : ℝ) (maria_age : ℝ) (carson_age : ℝ) : 
  anna_age = 60 →
  maria_age = 2 / 3 * anna_age →
  carson_age = maria_age - 7 →
  carson_age = 33 := by
sorry

end NUMINAMATH_CALUDE_carson_age_carson_age_real_l2488_248859


namespace NUMINAMATH_CALUDE_gcd_of_sum_is_222_l2488_248854

def is_consecutive_even (a b c d : ℕ) : Prop :=
  b = a + 2 ∧ c = a + 4 ∧ d = a + 6 ∧ a % 2 = 0

def e_sum (a d : ℕ) : ℕ := a + d

def abcde (a b c d e : ℕ) : ℕ := 10000 * a + 1000 * b + 100 * c + 10 * d + e

def edcba (a b c d e : ℕ) : ℕ := 10000 * e + 1000 * d + 100 * c + 10 * b + a

theorem gcd_of_sum_is_222 (a b c d : ℕ) (h : is_consecutive_even a b c d) :
  Nat.gcd (abcde a b c d (e_sum a d) + edcba a b c d (e_sum a d))
          (abcde (a + 2) (b + 2) (c + 2) (d + 2) (e_sum (a + 2) (d + 2)) +
           edcba (a + 2) (b + 2) (c + 2) (d + 2) (e_sum (a + 2) (d + 2))) = 222 :=
sorry

end NUMINAMATH_CALUDE_gcd_of_sum_is_222_l2488_248854


namespace NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l2488_248885

/-- An ellipse E with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The equation of an ellipse -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The area of the quadrilateral formed by the four vertices of an ellipse -/
def quadrilateral_area (E : Ellipse) : ℝ :=
  2 * E.a * E.b

theorem ellipse_equation_from_conditions (E : Ellipse) 
  (h_vertex : ellipse_equation E 0 (-2))
  (h_area : quadrilateral_area E = 4 * Real.sqrt 5) :
  ∀ x y, ellipse_equation E x y ↔ x^2 / 5 + y^2 / 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l2488_248885


namespace NUMINAMATH_CALUDE_calculate_expression_l2488_248812

theorem calculate_expression : (1/2)⁻¹ + |Real.sqrt 3 - 2| + Real.sqrt 12 = 4 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2488_248812


namespace NUMINAMATH_CALUDE_male_students_in_sample_l2488_248815

/-- Represents the stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  sample_size : ℕ
  female_count : ℕ

/-- Calculates the number of male students to be drawn in a stratified sample -/
def male_students_drawn (s : StratifiedSample) : ℕ :=
  s.sample_size

/-- Theorem stating the number of male students to be drawn in the given scenario -/
theorem male_students_in_sample (s : StratifiedSample) 
  (h1 : s.total_population = 900)
  (h2 : s.sample_size = 45)
  (h3 : s.female_count = 0) :
  male_students_drawn s = 25 := by
    sorry

#eval male_students_drawn { total_population := 900, sample_size := 45, female_count := 0 }

end NUMINAMATH_CALUDE_male_students_in_sample_l2488_248815


namespace NUMINAMATH_CALUDE_robin_gum_count_l2488_248865

/-- The number of gum packages Robin has -/
def num_packages : ℕ := 12

/-- The number of gum pieces in each package -/
def pieces_per_package : ℕ := 20

/-- The total number of gum pieces Robin has -/
def total_pieces : ℕ := num_packages * pieces_per_package

theorem robin_gum_count : total_pieces = 240 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_count_l2488_248865


namespace NUMINAMATH_CALUDE_range_of_a_l2488_248876

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + a*x + 4 = 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (¬(p a ∧ q a)) ∧ (p a ∨ q a) → (e ≤ a ∧ a < 4) ∨ (a ≤ -4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2488_248876


namespace NUMINAMATH_CALUDE_division_of_decimals_l2488_248805

theorem division_of_decimals : (0.05 : ℝ) / 0.0025 = 20 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l2488_248805


namespace NUMINAMATH_CALUDE_inscribed_semicircle_radius_l2488_248846

/-- An isosceles triangle with base 16 and height 15 -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  isBase16 : base = 16
  isHeight15 : height = 15

/-- A semicircle inscribed in the isosceles triangle -/
structure InscribedSemicircle (t : IsoscelesTriangle) where
  radius : ℝ
  diameterOnBase : radius * 2 ≤ t.base

/-- The radius of the inscribed semicircle is 120/17 -/
theorem inscribed_semicircle_radius (t : IsoscelesTriangle) 
  (s : InscribedSemicircle t) : s.radius = 120 / 17 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_semicircle_radius_l2488_248846


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2488_248848

theorem triangle_angle_calculation (a b c : ℝ) (A B C : ℝ) :
  a = 3 →
  b = Real.sqrt 6 →
  A = 2 * π / 3 →
  (a / Real.sin A = b / Real.sin B) →
  B = π / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l2488_248848


namespace NUMINAMATH_CALUDE_complex_number_location_l2488_248856

theorem complex_number_location (z : ℂ) (h : z * (-1 + 2 * Complex.I) = Complex.abs (1 + 3 * Complex.I)) :
  z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l2488_248856


namespace NUMINAMATH_CALUDE_rotated_square_height_l2488_248896

/-- The distance of point B from the original line when a square is rotated -/
theorem rotated_square_height (side_length : ℝ) (rotation_angle : ℝ) : 
  side_length = 4 →
  rotation_angle = 30 * π / 180 →
  let diagonal := side_length * Real.sqrt 2
  let height := (diagonal / 2) * Real.sin rotation_angle
  height = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rotated_square_height_l2488_248896


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_inequality_condition_l2488_248869

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + 2 * |x - 1|

-- Part 1
theorem min_value_when_a_is_one :
  ∃ m : ℝ, (∀ x : ℝ, f x 1 ≥ m) ∧ (∃ x : ℝ, f x 1 = m) ∧ m = 2 :=
sorry

-- Part 2
theorem inequality_condition :
  ∀ a b : ℝ, a > 0 → b > 0 →
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → f x a > x^2 - b + 1) →
  (a + 1/2)^2 + (b + 1/2)^2 > 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_one_inequality_condition_l2488_248869


namespace NUMINAMATH_CALUDE_root_sum_product_l2488_248887

theorem root_sum_product (p q r : ℝ) : 
  (5 * p^3 - 10 * p^2 + 17 * p - 7 = 0) ∧ 
  (5 * q^3 - 10 * q^2 + 17 * q - 7 = 0) ∧ 
  (5 * r^3 - 10 * r^2 + 17 * r - 7 = 0) → 
  p * q + p * r + q * r = 17 / 5 := by
sorry

end NUMINAMATH_CALUDE_root_sum_product_l2488_248887


namespace NUMINAMATH_CALUDE_all_approximations_valid_l2488_248820

/-- Represents an approximation with its estimated value, absolute error, and relative error. -/
structure Approximation where
  value : ℝ
  absoluteError : ℝ
  relativeError : ℝ

/-- The approximations given in the problem. -/
def classSize : Approximation := ⟨40, 5, 0.125⟩
def hallPeople : Approximation := ⟨1500, 100, 0.067⟩
def itemPrice : Approximation := ⟨100, 5, 0.05⟩
def pageCharacters : Approximation := ⟨40000, 500, 0.0125⟩

/-- Checks if the relative error is correctly calculated from the absolute error and value. -/
def isValidApproximation (a : Approximation) : Prop :=
  a.relativeError = a.absoluteError / a.value

/-- Proves that all given approximations are valid. -/
theorem all_approximations_valid :
  isValidApproximation classSize ∧
  isValidApproximation hallPeople ∧
  isValidApproximation itemPrice ∧
  isValidApproximation pageCharacters :=
sorry

end NUMINAMATH_CALUDE_all_approximations_valid_l2488_248820


namespace NUMINAMATH_CALUDE_arithmetic_mean_sqrt2_l2488_248883

theorem arithmetic_mean_sqrt2 :
  let x := Real.sqrt 2 - 1
  (x + (1 / x)) / 2 = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_sqrt2_l2488_248883


namespace NUMINAMATH_CALUDE_other_bill_value_l2488_248849

theorem other_bill_value (total_bills : ℕ) (total_value : ℕ) (five_dollar_bills : ℕ) :
  total_bills = 12 →
  total_value = 100 →
  five_dollar_bills = 4 →
  ∃ other_value : ℕ, 
    other_value * (total_bills - five_dollar_bills) + 5 * five_dollar_bills = total_value ∧
    other_value = 10 :=
by sorry

end NUMINAMATH_CALUDE_other_bill_value_l2488_248849


namespace NUMINAMATH_CALUDE_max_runs_in_one_day_match_l2488_248837

/-- Represents the number of overs in a cricket one-day match -/
def overs : ℕ := 50

/-- Represents the number of legal deliveries per over -/
def deliveries_per_over : ℕ := 6

/-- Represents the maximum number of runs that can be scored on a single delivery -/
def max_runs_per_delivery : ℕ := 6

/-- Theorem stating the maximum number of runs a batsman can score in an ideal scenario -/
theorem max_runs_in_one_day_match :
  overs * deliveries_per_over * max_runs_per_delivery = 1800 := by
  sorry

end NUMINAMATH_CALUDE_max_runs_in_one_day_match_l2488_248837


namespace NUMINAMATH_CALUDE_proportionality_problem_l2488_248875

/-- Given that x is directly proportional to y² and inversely proportional to z,
    prove that x = 24 when y = 2 and z = 3, given that x = 6 when y = 1 and z = 3. -/
theorem proportionality_problem (k : ℝ) :
  (∀ y z, ∃ x, x = k * y^2 / z) →
  (6 = k * 1^2 / 3) →
  (∃ x, x = k * 2^2 / 3 ∧ x = 24) :=
by sorry

end NUMINAMATH_CALUDE_proportionality_problem_l2488_248875


namespace NUMINAMATH_CALUDE_min_tan_angle_ocular_rays_l2488_248840

def G : Set (ℕ × ℕ) := {p | p.1 ≤ 20 ∧ p.2 ≤ 20 ∧ p.1 > 0 ∧ p.2 > 0}

def isOcularRay (m : ℚ) : Prop := ∃ p ∈ G, m = p.2 / p.1

def tanAngleBetweenRays (m1 m2 : ℚ) : ℚ := |m1 - m2| / (1 + m1 * m2)

def A : Set ℚ := {a | ∃ m1 m2, isOcularRay m1 ∧ isOcularRay m2 ∧ m1 ≠ m2 ∧ a = tanAngleBetweenRays m1 m2}

theorem min_tan_angle_ocular_rays :
  ∃ a ∈ A, a = (1 : ℚ) / 722 ∧ ∀ b ∈ A, (1 : ℚ) / 722 ≤ b :=
sorry

end NUMINAMATH_CALUDE_min_tan_angle_ocular_rays_l2488_248840


namespace NUMINAMATH_CALUDE_albert_additional_laps_l2488_248838

/-- Calculates the number of additional complete laps needed to finish a given distance. -/
def additional_laps (total_distance : ℕ) (track_length : ℕ) (completed_laps : ℕ) : ℕ :=
  ((total_distance - completed_laps * track_length) / track_length : ℕ)

/-- Theorem: Given the specific conditions, the number of additional complete laps is 5. -/
theorem albert_additional_laps :
  additional_laps 99 9 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_albert_additional_laps_l2488_248838


namespace NUMINAMATH_CALUDE_intersection_sum_l2488_248821

theorem intersection_sum (a b : ℝ) : 
  (2 = (1/3) * 4 + a) ∧ (4 = (1/3) * 2 + b) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l2488_248821


namespace NUMINAMATH_CALUDE_tangent_line_at_one_a_lower_bound_local_max_inequality_l2488_248888

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - 2 * a * x

def g (a : ℝ) (x : ℝ) : ℝ := f a x + (1/2) * x^2

def hasTangentLine (f : ℝ → ℝ) (x₀ y₀ m : ℝ) : Prop :=
  ∀ x, f x = m * (x - x₀) + y₀

def hasLocalMax (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - x₀| < ε → f x ≤ f x₀

theorem tangent_line_at_one (a : ℝ) (h : a = 2) :
  hasTangentLine (f a) 1 (-4) (-3) :=
sorry

theorem a_lower_bound (a : ℝ) (h : ∀ x > 0, f a x ≤ 2) :
  a ≥ 1 / (2 * Real.exp 3) :=
sorry

theorem local_max_inequality (a : ℝ) (x₀ : ℝ) (h : hasLocalMax (g a) x₀) :
  x₀ * f a x₀ + 1 + a * x₀^2 > 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_a_lower_bound_local_max_inequality_l2488_248888


namespace NUMINAMATH_CALUDE_sin_equation_holds_ten_degrees_is_acute_l2488_248832

theorem sin_equation_holds : 
  (Real.sin (10 * Real.pi / 180)) * (1 + Real.sqrt 3 * Real.tan (70 * Real.pi / 180)) = 1 := by
  sorry

-- Additional definition to ensure 10° is acute
def is_acute_angle (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

theorem ten_degrees_is_acute : is_acute_angle (10 * Real.pi / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_equation_holds_ten_degrees_is_acute_l2488_248832


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_five_l2488_248811

theorem fraction_zero_implies_x_negative_five (x : ℝ) :
  (x + 5) / (x - 2) = 0 → x = -5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_five_l2488_248811


namespace NUMINAMATH_CALUDE_factors_of_243_l2488_248855

theorem factors_of_243 : Finset.card (Nat.divisors 243) = 6 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_243_l2488_248855


namespace NUMINAMATH_CALUDE_orange_harvest_difference_l2488_248800

/-- Represents the harvest rates for a type of orange --/
structure HarvestRate where
  ripe : ℕ
  unripe : ℕ

/-- Represents the harvest rates for weekdays and weekends --/
structure WeeklyHarvestRate where
  weekday : HarvestRate
  weekend : HarvestRate

/-- Calculates the total difference between ripe and unripe oranges for a week --/
def weeklyDifference (rate : WeeklyHarvestRate) : ℕ :=
  (rate.weekday.ripe * 5 + rate.weekend.ripe * 2) -
  (rate.weekday.unripe * 5 + rate.weekend.unripe * 2)

theorem orange_harvest_difference :
  let valencia := WeeklyHarvestRate.mk (HarvestRate.mk 90 38) (HarvestRate.mk 75 33)
  let navel := WeeklyHarvestRate.mk (HarvestRate.mk 125 65) (HarvestRate.mk 100 57)
  let blood := WeeklyHarvestRate.mk (HarvestRate.mk 60 42) (HarvestRate.mk 45 36)
  weeklyDifference valencia + weeklyDifference navel + weeklyDifference blood = 838 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_difference_l2488_248800


namespace NUMINAMATH_CALUDE_video_distribution_solution_l2488_248886

/-- Represents the problem of distributing video content across discs -/
def VideoDistribution (total_minutes : ℝ) (max_capacity : ℝ) : Prop :=
  ∃ (num_discs : ℕ) (minutes_per_disc : ℝ),
    num_discs > 0 ∧
    num_discs = ⌈total_minutes / max_capacity⌉ ∧
    minutes_per_disc = total_minutes / num_discs ∧
    minutes_per_disc ≤ max_capacity

/-- Theorem stating the solution to the video distribution problem -/
theorem video_distribution_solution :
  VideoDistribution 495 65 →
  ∃ (num_discs : ℕ) (minutes_per_disc : ℝ),
    num_discs = 8 ∧ minutes_per_disc = 61.875 := by
  sorry

#check video_distribution_solution

end NUMINAMATH_CALUDE_video_distribution_solution_l2488_248886


namespace NUMINAMATH_CALUDE_chinese_number_puzzle_l2488_248818

def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem chinese_number_puzzle :
  ∀ (x y z : ℕ),
    x < 100 →
    y < 10000 →
    z < 10000 →
    100 * x + y + z = 2015 →
    z + sum_n 10 = y →
    x ≠ (y / 1000) →
    x ≠ (y / 100 % 10) →
    x ≠ (y / 10 % 10) →
    x ≠ (y % 10) →
    x ≠ (z / 1000) →
    x ≠ (z / 100 % 10) →
    x ≠ (z / 10 % 10) →
    x ≠ (z % 10) →
    (y / 1000) ≠ (y / 100 % 10) →
    (y / 1000) ≠ (y / 10 % 10) →
    (y / 1000) ≠ (y % 10) →
    (y / 100 % 10) ≠ (y / 10 % 10) →
    (y / 100 % 10) ≠ (y % 10) →
    (y / 10 % 10) ≠ (y % 10) →
    (z / 1000) ≠ (z / 100 % 10) →
    (z / 1000) ≠ (z / 10 % 10) →
    (z / 1000) ≠ (z % 10) →
    (z / 100 % 10) ≠ (z / 10 % 10) →
    (z / 100 % 10) ≠ (z % 10) →
    (z / 10 % 10) ≠ (z % 10) →
    100 * x + y = 1985 :=
by
  sorry

#eval sum_n 10  -- This should evaluate to 55

end NUMINAMATH_CALUDE_chinese_number_puzzle_l2488_248818


namespace NUMINAMATH_CALUDE_joes_test_scores_l2488_248861

theorem joes_test_scores (initial_avg : ℝ) (lowest_score : ℝ) (new_avg : ℝ) :
  initial_avg = 70 →
  lowest_score = 55 →
  new_avg = 75 →
  ∃ n : ℕ, n > 1 ∧
    (n : ℝ) * initial_avg - lowest_score = (n - 1 : ℝ) * new_avg ∧
    n = 4 :=
by sorry

end NUMINAMATH_CALUDE_joes_test_scores_l2488_248861


namespace NUMINAMATH_CALUDE_jane_average_score_l2488_248878

def jane_scores : List ℝ := [89, 95, 88, 92, 94, 87]

theorem jane_average_score : 
  (jane_scores.sum / jane_scores.length : ℝ) = 90.8333 := by
  sorry

end NUMINAMATH_CALUDE_jane_average_score_l2488_248878


namespace NUMINAMATH_CALUDE_cyclist_distance_l2488_248881

theorem cyclist_distance (travel_time : ℝ) (car_distance : ℝ) (speed_difference : ℝ) :
  travel_time = 8 →
  car_distance = 48 →
  speed_difference = 5 →
  let car_speed := car_distance / travel_time
  let cyclist_speed := car_speed - speed_difference
  cyclist_speed * travel_time = 8 := by sorry

end NUMINAMATH_CALUDE_cyclist_distance_l2488_248881


namespace NUMINAMATH_CALUDE_smallest_x_value_l2488_248813

theorem smallest_x_value (x y : ℤ) (h : x * y + 7 * x + 6 * y = -8) : 
  (∀ z : ℤ, ∃ w : ℤ, z * w + 7 * z + 6 * w = -8 → z ≥ -40) ∧ 
  (∃ w : ℤ, -40 * w + 7 * (-40) + 6 * w = -8) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l2488_248813


namespace NUMINAMATH_CALUDE_handshake_arrangement_count_l2488_248822

/-- Represents a handshaking arrangement for a group of people -/
structure HandshakeArrangement (n : ℕ) :=
  (shakes : Fin n → Finset (Fin n))
  (shake_count : ∀ i, (shakes i).card = 3)
  (symmetry : ∀ i j, j ∈ shakes i ↔ i ∈ shakes j)

/-- The number of valid handshaking arrangements for 12 people -/
def M : ℕ := sorry

/-- Theorem stating that the number of handshaking arrangements is congruent to 340 modulo 1000 -/
theorem handshake_arrangement_count :
  M ≡ 340 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_handshake_arrangement_count_l2488_248822


namespace NUMINAMATH_CALUDE_largest_multiple_of_nine_l2488_248807

theorem largest_multiple_of_nine (n : ℤ) : 
  (n % 9 = 0 ∧ -n > -100) → n ≤ 99 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_nine_l2488_248807


namespace NUMINAMATH_CALUDE_probability_sum_5_is_one_ninth_l2488_248831

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of ways to roll a sum of 5 with two dice -/
def favorable_outcomes : ℕ := 4

/-- The probability of rolling a sum of 5 with two dice -/
def probability_sum_5 : ℚ := favorable_outcomes / total_outcomes

theorem probability_sum_5_is_one_ninth : probability_sum_5 = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_5_is_one_ninth_l2488_248831


namespace NUMINAMATH_CALUDE_cloud_same_color_tangents_iff_l2488_248891

/-- A configuration of n points on a line with circumferences painted in k colors -/
structure Cloud (n k : ℕ) where
  n_ge_two : n ≥ 2
  colors : Fin k → Type
  circumferences : Fin n → Fin n → Option (Fin k)
  different_points : ∀ i j : Fin n, i ≠ j → circumferences i j ≠ none

/-- Two circumferences are mutually exterior tangent -/
def mutually_exterior_tangent (c : Cloud n k) (i j m l : Fin n) : Prop :=
  (i ≠ j ∧ m ≠ l) ∧ (i = m ∨ i = l ∨ j = m ∨ j = l)

/-- A cloud has two mutually exterior tangent circumferences of the same color -/
def has_same_color_tangents (c : Cloud n k) : Prop :=
  ∃ i j m l : Fin n, ∃ color : Fin k,
    mutually_exterior_tangent c i j m l ∧
    c.circumferences i j = some color ∧
    c.circumferences m l = some color

/-- Main theorem: characterization of n for which all (n,k)-clouds have same color tangents -/
theorem cloud_same_color_tangents_iff (k : ℕ) :
  (∀ n : ℕ, ∀ c : Cloud n k, has_same_color_tangents c) ↔ n ≥ 2^k + 1 :=
sorry

end NUMINAMATH_CALUDE_cloud_same_color_tangents_iff_l2488_248891


namespace NUMINAMATH_CALUDE_range_of_a_l2488_248890

theorem range_of_a (p : Prop) (h : p) : 
  (∀ x : ℝ, x ∈ Set.Ioo 1 2 → Real.exp x - a ≤ 0) → a ∈ Set.Ici (Real.exp 2) :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2488_248890


namespace NUMINAMATH_CALUDE_log_relation_l2488_248817

theorem log_relation (p q : ℝ) (hp : 0 < p) : 
  (Real.log 5 / Real.log 8 = p) → (Real.log 125 / Real.log 2 = q * p) → q = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_relation_l2488_248817


namespace NUMINAMATH_CALUDE_sixth_sample_is_98_l2488_248826

/-- Systematic sampling function -/
def systematicSample (total : ℕ) (sampleSize : ℕ) (start : ℕ) (k : ℕ) : ℕ :=
  start + (k - 1) * (total / sampleSize)

theorem sixth_sample_is_98 :
  systematicSample 900 50 8 6 = 98 := by
  sorry

end NUMINAMATH_CALUDE_sixth_sample_is_98_l2488_248826


namespace NUMINAMATH_CALUDE_pi_estimation_l2488_248833

theorem pi_estimation (m n : ℕ) (h : m > 0) : 
  ∃ (ε : ℝ), ε > 0 ∧ |4 * (n : ℝ) / (m : ℝ) - π| < ε :=
sorry

end NUMINAMATH_CALUDE_pi_estimation_l2488_248833


namespace NUMINAMATH_CALUDE_class_book_count_l2488_248836

/-- Calculates the total number of books a class has from the library --/
def totalBooks (initial borrowed₁ returned borrowed₂ : ℕ) : ℕ :=
  initial + borrowed₁ - returned + borrowed₂

/-- Theorem: The class currently has 80 books from the library --/
theorem class_book_count :
  totalBooks 54 23 12 15 = 80 := by
  sorry

end NUMINAMATH_CALUDE_class_book_count_l2488_248836


namespace NUMINAMATH_CALUDE_intersection_sequence_correct_l2488_248839

def A : Set ℕ := {n | ∃ m : ℕ+, n = m * (m + 1)}
def B : Set ℕ := {n | ∃ m : ℕ+, n = 3 * m - 1}

def intersection_sequence (k : ℕ+) : ℕ := 9 * k^2 - 9 * k + 2

theorem intersection_sequence_correct :
  ∀ k : ℕ+, (intersection_sequence k) ∈ A ∩ B ∧
  (∀ n ∈ A ∩ B, n < intersection_sequence k → 
    ∃ j : ℕ+, j < k ∧ n = intersection_sequence j) :=
sorry

end NUMINAMATH_CALUDE_intersection_sequence_correct_l2488_248839


namespace NUMINAMATH_CALUDE_value_of_n_l2488_248853

theorem value_of_n : ∃ n : ℕ, 5^3 - 7 = 2^2 + n ∧ n = 114 := by sorry

end NUMINAMATH_CALUDE_value_of_n_l2488_248853


namespace NUMINAMATH_CALUDE_robins_full_pages_l2488_248871

/-- The number of full pages in a photo album -/
def full_pages (total_photos : ℕ) (photos_per_page : ℕ) : ℕ :=
  total_photos / photos_per_page

/-- Theorem: Robin's photo album has 181 full pages -/
theorem robins_full_pages :
  full_pages 2176 12 = 181 := by
  sorry

end NUMINAMATH_CALUDE_robins_full_pages_l2488_248871


namespace NUMINAMATH_CALUDE_profit_starts_in_fourth_year_option_one_more_profitable_l2488_248867

/-- Represents the financial data for the real estate investment --/
structure RealEstateInvestment where
  initialInvestment : ℝ
  firstYearRenovationCost : ℝ
  yearlyRenovationIncrease : ℝ
  annualRentalIncome : ℝ

/-- Calculates the renovation cost for a given year --/
def renovationCost (investment : RealEstateInvestment) (year : ℕ) : ℝ :=
  investment.firstYearRenovationCost + investment.yearlyRenovationIncrease * (year - 1)

/-- Calculates the cumulative profit up to a given year --/
def cumulativeProfit (investment : RealEstateInvestment) (year : ℕ) : ℝ :=
  year * investment.annualRentalIncome - investment.initialInvestment - 
    (Finset.range year).sum (fun i => renovationCost investment (i + 1))

/-- Theorem stating that the developer starts making a net profit in the 4th year --/
theorem profit_starts_in_fourth_year (investment : RealEstateInvestment) 
  (h1 : investment.initialInvestment = 810000)
  (h2 : investment.firstYearRenovationCost = 10000)
  (h3 : investment.yearlyRenovationIncrease = 20000)
  (h4 : investment.annualRentalIncome = 300000) :
  (cumulativeProfit investment 3 < 0) ∧ (cumulativeProfit investment 4 > 0) := by
  sorry

/-- Represents the two selling options --/
inductive SellingOption
  | OptionOne : SellingOption
  | OptionTwo : SellingOption

/-- Calculates the profit for a given selling option --/
def profitForOption (investment : RealEstateInvestment) (option : SellingOption) : ℝ :=
  match option with
  | SellingOption.OptionOne => 460000 -- Simplified for the sake of the statement
  | SellingOption.OptionTwo => 100000 -- Simplified for the sake of the statement

/-- Theorem stating that Option 1 is more profitable --/
theorem option_one_more_profitable (investment : RealEstateInvestment) :
  profitForOption investment SellingOption.OptionOne > profitForOption investment SellingOption.OptionTwo := by
  sorry

end NUMINAMATH_CALUDE_profit_starts_in_fourth_year_option_one_more_profitable_l2488_248867


namespace NUMINAMATH_CALUDE_cyclist_time_is_two_hours_l2488_248825

/-- Represents the scenario of two cyclists traveling between two points --/
structure CyclistScenario where
  s : ℝ  -- Base speed of cyclists without wind
  t : ℝ  -- Time taken by Cyclist 1 to travel from A to B
  wind_speed : ℝ := 3  -- Wind speed affecting both cyclists

/-- Conditions of the cyclist problem --/
def cyclist_problem (scenario : CyclistScenario) : Prop :=
  let total_time := 4  -- Total time after which they meet
  -- Distance covered by Cyclist 1 in total_time
  let dist_cyclist1 := scenario.s * total_time + scenario.wind_speed * (2 * scenario.t - total_time)
  -- Distance covered by Cyclist 2 in total_time
  let dist_cyclist2 := (scenario.s - scenario.wind_speed) * total_time
  -- They meet halfway of the total distance
  dist_cyclist1 = dist_cyclist2 + (scenario.s + scenario.wind_speed) * scenario.t

/-- The theorem stating that the time taken by Cyclist 1 to travel from A to B is 2 hours --/
theorem cyclist_time_is_two_hours (scenario : CyclistScenario) :
  cyclist_problem scenario → scenario.t = 2 := by
  sorry

#check cyclist_time_is_two_hours

end NUMINAMATH_CALUDE_cyclist_time_is_two_hours_l2488_248825


namespace NUMINAMATH_CALUDE_bicycle_price_calculation_l2488_248841

theorem bicycle_price_calculation (original_price : ℝ) 
  (initial_discount_rate : ℝ) (additional_discount : ℝ) (sales_tax_rate : ℝ) :
  original_price = 200 →
  initial_discount_rate = 0.25 →
  additional_discount = 10 →
  sales_tax_rate = 0.05 →
  (original_price * (1 - initial_discount_rate) - additional_discount) * (1 + sales_tax_rate) = 147 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_calculation_l2488_248841


namespace NUMINAMATH_CALUDE_rectangular_plot_ratio_l2488_248823

/-- A rectangular plot with given area and breadth -/
structure RectangularPlot where
  area : ℝ
  breadth : ℝ
  length_multiple : ℕ
  area_eq : area = breadth * (breadth * length_multiple)

/-- The ratio of length to breadth for a rectangular plot -/
def length_breadth_ratio (plot : RectangularPlot) : ℚ :=
  plot.length_multiple

theorem rectangular_plot_ratio (plot : RectangularPlot) 
  (h1 : plot.area = 432)
  (h2 : plot.breadth = 12) :
  length_breadth_ratio plot = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_ratio_l2488_248823


namespace NUMINAMATH_CALUDE_multiply_98_by_98_l2488_248828

theorem multiply_98_by_98 : 98 * 98 = 9604 := by
  sorry

end NUMINAMATH_CALUDE_multiply_98_by_98_l2488_248828


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l2488_248814

theorem quadratic_distinct_roots_range (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + k = 0 ∧ y^2 - 2*y + k = 0) ↔ k < 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l2488_248814


namespace NUMINAMATH_CALUDE_function_composition_equality_l2488_248806

theorem function_composition_equality (A B : ℝ) (h : B ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ A * x^2 - 3 * B^3
  let g : ℝ → ℝ := λ x ↦ 2 * B * x + B^2
  f (g 2) = 0 → A = 3 / (16 / B + 8 * B + B^3) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l2488_248806


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2488_248872

theorem geometric_sequence_product (a b : ℝ) : 
  (5 < a) → (a < b) → (b < 40) → 
  (b / a = a / 5) → (40 / b = b / a) → 
  a * b = 200 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2488_248872


namespace NUMINAMATH_CALUDE_graduate_ratio_l2488_248801

theorem graduate_ratio (N : ℝ) (G : ℝ) (C : ℝ) 
  (h1 : C = (2/3) * N) 
  (h2 : G / (G + C) = 0.15789473684210525) : 
  G = (1/8) * N := by
sorry

end NUMINAMATH_CALUDE_graduate_ratio_l2488_248801


namespace NUMINAMATH_CALUDE_inequality_abc_l2488_248862

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^a * b^b * c^c ≥ (a*b*c)^((a+b+c)/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_abc_l2488_248862


namespace NUMINAMATH_CALUDE_mean_increases_median_may_unchanged_variance_increases_l2488_248882

variable (n : ℕ)
variable (x : Fin n → ℝ)
variable (x_n_plus_1 : ℝ)

-- Assume n ≥ 3
axiom n_ge_3 : n ≥ 3

-- Define the median, mean, and variance of the original dataset
def median : ℝ := sorry
def mean : ℝ := sorry
def variance : ℝ := sorry

-- Assume x_n_plus_1 is much greater than any value in x
axiom x_n_plus_1_much_greater : ∀ i : Fin n, x_n_plus_1 > x i

-- Define the new dataset including x_n_plus_1
def new_dataset : Fin (n + 1) → ℝ :=
  λ i => if h : i.val < n then x ⟨i.val, h⟩ else x_n_plus_1

-- Define the new median, mean, and variance
def new_median : ℝ := sorry
def new_mean : ℝ := sorry
def new_variance : ℝ := sorry

-- Theorem statements
theorem mean_increases : new_mean > mean := sorry
theorem median_may_unchanged : new_median = median ∨ new_median > median := sorry
theorem variance_increases : new_variance > variance := sorry

end NUMINAMATH_CALUDE_mean_increases_median_may_unchanged_variance_increases_l2488_248882


namespace NUMINAMATH_CALUDE_factorization_equality_l2488_248879

theorem factorization_equality (x y : ℝ) : -2*x^2 + 4*x*y - 2*y^2 = -2*(x-y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2488_248879


namespace NUMINAMATH_CALUDE_bankers_gain_calculation_l2488_248808

/-- Banker's gain calculation -/
theorem bankers_gain_calculation (true_discount : ℝ) (interest_rate : ℝ) (time_period : ℝ) :
  true_discount = 60.00000000000001 →
  interest_rate = 0.12 →
  time_period = 1 →
  let face_value := (true_discount * (1 + interest_rate * time_period)) / (interest_rate * time_period)
  let bankers_discount := face_value * interest_rate * time_period
  bankers_discount - true_discount = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_bankers_gain_calculation_l2488_248808


namespace NUMINAMATH_CALUDE_x_y_equation_l2488_248889

theorem x_y_equation (x y : ℚ) (hx : x = 2/3) (hy : y = 9/2) : (1/3) * x^4 * y^5 = 121.5 := by
  sorry

end NUMINAMATH_CALUDE_x_y_equation_l2488_248889


namespace NUMINAMATH_CALUDE_harmonic_sum_identity_l2488_248860

def h (n : ℕ) : ℚ :=
  (Finset.range n).sum (fun i => 1 / (i + 1 : ℚ))

theorem harmonic_sum_identity (n : ℕ) (hn : n ≥ 2) :
  n + (Finset.range (n - 1)).sum h = n * h n := by
  sorry

end NUMINAMATH_CALUDE_harmonic_sum_identity_l2488_248860


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l2488_248834

/-- The surface area of a cuboid with given dimensions. -/
def cuboidSurfaceArea (length breadth height : ℝ) : ℝ :=
  2 * (length * height + breadth * height + length * breadth)

/-- Theorem: The surface area of a cuboid with length 4 cm, breadth 6 cm, and height 5 cm is 148 cm². -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 4 6 5 = 148 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l2488_248834


namespace NUMINAMATH_CALUDE_peter_distance_l2488_248816

/-- The total distance Peter covers -/
def D : ℝ := sorry

/-- The time Peter takes to cover the distance in hours -/
def total_time : ℝ := 1.4

/-- The speed at which Peter covers two-thirds of the distance -/
def speed1 : ℝ := 4

/-- The speed at which Peter covers one-third of the distance -/
def speed2 : ℝ := 5

theorem peter_distance : 
  (2/3 * D) / speed1 + (1/3 * D) / speed2 = total_time ∧ D = 6 := by sorry

end NUMINAMATH_CALUDE_peter_distance_l2488_248816


namespace NUMINAMATH_CALUDE_three_fourths_of_forty_l2488_248809

theorem three_fourths_of_forty : (3 / 4 : ℚ) * 40 = 30 := by
  sorry

end NUMINAMATH_CALUDE_three_fourths_of_forty_l2488_248809


namespace NUMINAMATH_CALUDE_fraction_is_composite_l2488_248899

theorem fraction_is_composite : ¬ Nat.Prime ((5^125 - 1) / (5^25 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_is_composite_l2488_248899


namespace NUMINAMATH_CALUDE_rachel_problem_solving_time_l2488_248827

/-- The number of minutes Rachel spent solving math problems before bed -/
def minutes_before_bed : ℕ := 12

/-- The number of problems Rachel solved per minute before bed -/
def problems_per_minute : ℕ := 5

/-- The number of problems Rachel solved the next day -/
def problems_next_day : ℕ := 16

/-- The total number of problems Rachel solved -/
def total_problems : ℕ := 76

/-- Theorem stating that Rachel spent 12 minutes solving problems before bed -/
theorem rachel_problem_solving_time :
  minutes_before_bed * problems_per_minute + problems_next_day = total_problems :=
by sorry

end NUMINAMATH_CALUDE_rachel_problem_solving_time_l2488_248827


namespace NUMINAMATH_CALUDE_proposition_logic_proof_l2488_248804

theorem proposition_logic_proof (p q : Prop) 
  (hp : p ↔ (3 ≥ 3)) 
  (hq : q ↔ (3 > 4)) : 
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p) := by
  sorry

end NUMINAMATH_CALUDE_proposition_logic_proof_l2488_248804


namespace NUMINAMATH_CALUDE_remainder_of_power_divided_by_polynomial_l2488_248850

theorem remainder_of_power_divided_by_polynomial (x : ℤ) :
  (x + 1)^2010 ≡ 1 [ZMOD (x^2 + x + 1)] := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_power_divided_by_polynomial_l2488_248850


namespace NUMINAMATH_CALUDE_josh_remaining_money_l2488_248873

def initial_amount : ℝ := 100

def transactions : List ℝ := [12.67, 25.39, 14.25, 4.32, 27.50]

def remaining_money : ℝ := initial_amount - transactions.sum

theorem josh_remaining_money :
  remaining_money = 15.87 := by sorry

end NUMINAMATH_CALUDE_josh_remaining_money_l2488_248873


namespace NUMINAMATH_CALUDE_red_bus_length_l2488_248870

theorem red_bus_length 
  (red_bus orange_car yellow_bus : ℝ) 
  (h1 : red_bus = 4 * orange_car) 
  (h2 : orange_car = yellow_bus / 3.5) 
  (h3 : red_bus = yellow_bus + 6) : 
  red_bus = 48 := by
sorry

end NUMINAMATH_CALUDE_red_bus_length_l2488_248870


namespace NUMINAMATH_CALUDE_middle_number_in_ratio_l2488_248835

theorem middle_number_in_ratio (a b c : ℝ) : 
  a / b = 3 / 2 ∧ b / c = 2 / 5 ∧ a^2 + b^2 + c^2 = 1862 → b = 14 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_in_ratio_l2488_248835


namespace NUMINAMATH_CALUDE_probability_red_ball_l2488_248866

/-- The probability of drawing a red ball from a box with red and black balls -/
theorem probability_red_ball (red_balls black_balls : ℕ) :
  red_balls > 0 →
  black_balls ≥ 0 →
  (red_balls : ℚ) / (red_balls + black_balls : ℚ) = 7 / 10 ↔
  red_balls = 7 ∧ black_balls = 3 :=
by sorry

end NUMINAMATH_CALUDE_probability_red_ball_l2488_248866


namespace NUMINAMATH_CALUDE_banana_arrangements_l2488_248829

def word : String := "BANANA"

/-- The number of unique arrangements of letters in the word -/
def num_arrangements (w : String) : ℕ := sorry

theorem banana_arrangements :
  num_arrangements word = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l2488_248829


namespace NUMINAMATH_CALUDE_fifth_color_marbles_l2488_248897

/-- The number of marbles of each color in a box --/
structure MarbleCount where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ
  fifth : ℕ

/-- The properties of the marble counts --/
def valid_marble_count (m : MarbleCount) : Prop :=
  m.red = 25 ∧
  m.green = 3 * m.red ∧
  m.yellow = m.green / 5 ∧
  m.blue = 2 * m.yellow ∧
  m.red + m.green + m.yellow + m.blue + m.fifth = 4 * m.green

theorem fifth_color_marbles (m : MarbleCount) 
  (h : valid_marble_count m) : m.fifth = 155 := by
  sorry

end NUMINAMATH_CALUDE_fifth_color_marbles_l2488_248897


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_minus_z_squared_l2488_248884

theorem x_squared_minus_y_squared_minus_z_squared (x y z : ℝ) 
  (sum_eq : x + y + z = 12)
  (diff_eq : x - y = 4)
  (yz_sum : y + z = 7) :
  x^2 - y^2 - z^2 = -12 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_minus_z_squared_l2488_248884


namespace NUMINAMATH_CALUDE_seven_digit_sum_theorem_l2488_248824

theorem seven_digit_sum_theorem :
  ∀ a b : ℕ,
  (a ≤ 9 ∧ a > 0) →
  (b ≤ 9) →
  (7 * a = 10 * a + b) →
  (a + b = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_seven_digit_sum_theorem_l2488_248824


namespace NUMINAMATH_CALUDE_inequality_proof_l2488_248857

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) : ¬(1/a > 1/b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2488_248857


namespace NUMINAMATH_CALUDE_log_zero_nonexistent_l2488_248842

theorem log_zero_nonexistent : ¬ ∃ x : ℝ, Real.log x = 0 := by sorry

end NUMINAMATH_CALUDE_log_zero_nonexistent_l2488_248842


namespace NUMINAMATH_CALUDE_triangle_inequality_l2488_248843

theorem triangle_inequality (A B C : Real) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : C = π - A - B) 
  (h : 1 / Real.sin A + 2 / Real.sin B = 3 * (1 / Real.tan A + 1 / Real.tan B)) :
  Real.cos C ≥ (2 * Real.sqrt 10 - 2) / 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2488_248843


namespace NUMINAMATH_CALUDE_binomial_sum_modulo_1000_l2488_248864

theorem binomial_sum_modulo_1000 : 
  (Finset.sum (Finset.range 503) (fun k => Nat.choose 2011 (4 * k))) % 1000 = 15 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_modulo_1000_l2488_248864


namespace NUMINAMATH_CALUDE_smallest_class_size_l2488_248845

theorem smallest_class_size (n : ℕ) (scores : Fin n → ℕ) : 
  (∀ i, scores i ≤ 120) →  -- Each student took a 120-point test
  (∃ s : Finset (Fin n), s.card = 8 ∧ ∀ i ∈ s, scores i = 120) →  -- Eight students scored 120
  (∀ i, scores i ≥ 72) →  -- Each student scored at least 72
  (Finset.sum Finset.univ scores / n = 84) →  -- The mean score was 84
  (n ≥ 32 ∧ ∀ m : ℕ, m < 32 → ¬ (∃ scores' : Fin m → ℕ, 
    (∀ i, scores' i ≤ 120) ∧ 
    (∃ s : Finset (Fin m), s.card = 8 ∧ ∀ i ∈ s, scores' i = 120) ∧ 
    (∀ i, scores' i ≥ 72) ∧ 
    (Finset.sum Finset.univ scores' / m = 84))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_class_size_l2488_248845


namespace NUMINAMATH_CALUDE_stone_145_is_1_l2488_248892

/-- Represents the counting pattern for stones -/
def stone_count (n : ℕ) : ℕ :=
  if n ≤ 10 then n
  else if n ≤ 19 then 20 - n
  else stone_count ((n - 1) % 18 + 1)

/-- The theorem stating that the 145th count corresponds to the first stone -/
theorem stone_145_is_1 : stone_count 145 = 1 := by
  sorry

end NUMINAMATH_CALUDE_stone_145_is_1_l2488_248892


namespace NUMINAMATH_CALUDE_kittens_given_to_jessica_l2488_248852

theorem kittens_given_to_jessica (initial_kittens : ℕ) (received_kittens : ℕ) (final_kittens : ℕ) :
  initial_kittens = 6 →
  received_kittens = 9 →
  final_kittens = 12 →
  initial_kittens + received_kittens - final_kittens = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_kittens_given_to_jessica_l2488_248852


namespace NUMINAMATH_CALUDE_salt_mixture_proof_l2488_248868

/-- Proves that adding 70 ounces of a 60% salt solution to 70 ounces of a 20% salt solution
    results in a mixture that is 40% salt. -/
theorem salt_mixture_proof :
  let initial_amount : ℝ := 70
  let initial_concentration : ℝ := 0.20
  let added_amount : ℝ := 70
  let added_concentration : ℝ := 0.60
  let final_concentration : ℝ := 0.40
  let total_amount : ℝ := initial_amount + added_amount
  let total_salt : ℝ := initial_amount * initial_concentration + added_amount * added_concentration
  total_salt / total_amount = final_concentration := by
  sorry

#check salt_mixture_proof

end NUMINAMATH_CALUDE_salt_mixture_proof_l2488_248868


namespace NUMINAMATH_CALUDE_series_sum_equals_negative_four_l2488_248858

/-- The sum of the infinite series $\sum_{n=1}^\infty \frac{2n^2 - 3n + 2}{n(n+1)(n+2)}$ equals -4. -/
theorem series_sum_equals_negative_four :
  ∑' n : ℕ+, (2 * n^2 - 3 * n + 2 : ℝ) / (n * (n + 1) * (n + 2)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_negative_four_l2488_248858
