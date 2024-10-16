import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_l1355_135514

theorem simplify_expression : -(-3) - 4 + (-5) = 3 - 4 - 5 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1355_135514


namespace NUMINAMATH_CALUDE_residue_mod_13_l1355_135591

theorem residue_mod_13 : (156 + 3 * 52 + 4 * 182 + 6 * 26) % 13 = 0 := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_13_l1355_135591


namespace NUMINAMATH_CALUDE_total_marbles_count_l1355_135506

/-- The number of marbles Connie has -/
def connie_marbles : ℕ := 39

/-- The number of marbles Juan has -/
def juan_marbles : ℕ := connie_marbles + 25

/-- The number of marbles Maria has -/
def maria_marbles : ℕ := 2 * juan_marbles

/-- The total number of marbles for all three people -/
def total_marbles : ℕ := connie_marbles + juan_marbles + maria_marbles

theorem total_marbles_count : total_marbles = 231 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_count_l1355_135506


namespace NUMINAMATH_CALUDE_monotonic_power_function_l1355_135528

theorem monotonic_power_function (m : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → (m^2 - 5*m + 7) * x₁^(m^2 - 6) < (m^2 - 5*m + 7) * x₂^(m^2 - 6)) →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_monotonic_power_function_l1355_135528


namespace NUMINAMATH_CALUDE_triangle_side_length_l1355_135521

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c - t.b = 2 ∧
  Real.cos (t.A / 2) = Real.sqrt 3 / 3 ∧
  1/2 * t.b * t.c * Real.sin t.A = 5 * Real.sqrt 2

-- Theorem statement
theorem triangle_side_length (t : Triangle) :
  triangle_conditions t → t.a = 2 * Real.sqrt 11 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1355_135521


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l1355_135525

structure Rectangle where
  width : ℝ
  height : ℝ

def similar (r1 r2 : Rectangle) : Prop :=
  r1.width / r2.width = r1.height / r2.height

theorem rectangle_area_ratio 
  (ABCD EFGH : Rectangle) 
  (h1 : similar ABCD EFGH) 
  (h2 : ∃ (K : ℝ), K > 0 ∧ K < ABCD.width ∧ (ABCD.width - K) / K = 2 / 3) : 
  (ABCD.width * ABCD.height) / (EFGH.width * EFGH.height) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l1355_135525


namespace NUMINAMATH_CALUDE_nested_sqrt_value_l1355_135515

theorem nested_sqrt_value (y : ℝ) (h : y = Real.sqrt (2 + y)) : y = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_value_l1355_135515


namespace NUMINAMATH_CALUDE_angle_D_measure_l1355_135541

-- Define the hexagon and its angles
structure Hexagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ

-- Define the properties of the hexagon
def is_convex_hexagon_with_properties (h : Hexagon) : Prop :=
  -- Angles A, B, and C are congruent
  h.A = h.B ∧ h.B = h.C
  -- Angles D and E are congruent
  ∧ h.D = h.E
  -- Angle A is 50 degrees less than angle D
  ∧ h.A + 50 = h.D
  -- Sum of angles in a hexagon is 720 degrees
  ∧ h.A + h.B + h.C + h.D + h.E + h.F = 720

-- Theorem statement
theorem angle_D_measure (h : Hexagon) 
  (h_props : is_convex_hexagon_with_properties h) : 
  h.D = 153.33 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l1355_135541


namespace NUMINAMATH_CALUDE_expression_simplification_l1355_135562

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x^3 * y^2 - 3 * x^2 * y^3) / ((1/2 * x * y)^2) = 8*x - 12*y := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1355_135562


namespace NUMINAMATH_CALUDE_player_one_points_l1355_135537

/-- Represents the sectors on the rotating table -/
def sectors : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]

/-- The number of players -/
def num_players : ℕ := 16

/-- The number of rotations -/
def num_rotations : ℕ := 13

/-- Calculate the points for a player after given number of rotations -/
def player_points (player : ℕ) (rotations : ℕ) : ℕ := sorry

theorem player_one_points :
  player_points 5 num_rotations = 72 →
  player_points 9 num_rotations = 84 →
  player_points 1 num_rotations = 20 := by sorry

end NUMINAMATH_CALUDE_player_one_points_l1355_135537


namespace NUMINAMATH_CALUDE_exists_number_with_sum_and_count_of_factors_l1355_135597

open Nat

def sumOfDivisors (n : ℕ) : ℕ := sorry

def numberOfDivisors (n : ℕ) : ℕ := sorry

theorem exists_number_with_sum_and_count_of_factors :
  ∃ n : ℕ, n > 0 ∧ sumOfDivisors n + numberOfDivisors n = 1767 := by sorry

end NUMINAMATH_CALUDE_exists_number_with_sum_and_count_of_factors_l1355_135597


namespace NUMINAMATH_CALUDE_rod_and_rope_problem_l1355_135559

/-- 
Given a rod and a rope with the following properties:
1. The rope is 5 feet longer than the rod
2. When the rope is folded in half, it is 5 feet shorter than the rod

Prove that the system of equations x = y + 5 and 1/2 * x = y - 5 holds true,
where x is the length of the rope in feet and y is the length of the rod in feet.
-/
theorem rod_and_rope_problem (x y : ℝ) 
  (h1 : x = y + 5)
  (h2 : x / 2 = y - 5) : 
  x = y + 5 ∧ x / 2 = y - 5 := by
  sorry

end NUMINAMATH_CALUDE_rod_and_rope_problem_l1355_135559


namespace NUMINAMATH_CALUDE_jason_retirement_age_l1355_135520

def military_career (join_age : ℕ) (years_to_chief : ℕ) (years_after_master_chief : ℕ) : ℕ → Prop :=
  fun retirement_age =>
    ∃ (years_to_master_chief : ℕ),
      years_to_master_chief = years_to_chief + (years_to_chief * 25 / 100) ∧
      retirement_age = join_age + years_to_chief + years_to_master_chief + years_after_master_chief

theorem jason_retirement_age :
  military_career 18 8 10 46 := by
  sorry

end NUMINAMATH_CALUDE_jason_retirement_age_l1355_135520


namespace NUMINAMATH_CALUDE_lowest_score_within_two_std_dev_l1355_135561

/-- Represents the lowest score within a given number of standard deviations from the mean. -/
def lowestScore (mean standardDeviation : ℝ) (numStdDev : ℝ) : ℝ :=
  mean - numStdDev * standardDeviation

/-- Theorem stating that given a mean of 60 and standard deviation of 10,
    the lowest score within 2 standard deviations is 40. -/
theorem lowest_score_within_two_std_dev :
  lowestScore 60 10 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_lowest_score_within_two_std_dev_l1355_135561


namespace NUMINAMATH_CALUDE_base8_532_equals_base7_1006_l1355_135533

/-- Converts a number from base 8 to base 10 --/
def base8ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 --/
def decimalToBase7 (n : ℕ) : ℕ := sorry

/-- Theorem stating that 532 in base 8 is equal to 1006 in base 7 --/
theorem base8_532_equals_base7_1006 : 
  decimalToBase7 (base8ToDecimal 532) = 1006 := by sorry

end NUMINAMATH_CALUDE_base8_532_equals_base7_1006_l1355_135533


namespace NUMINAMATH_CALUDE_line_moved_upwards_l1355_135554

/-- Given a line with equation y = -x + 1, prove that moving it 5 units upwards
    results in the equation y = -x + 6 -/
theorem line_moved_upwards (x y : ℝ) :
  (y = -x + 1) → (y + 5 = -x + 6) :=
by sorry

end NUMINAMATH_CALUDE_line_moved_upwards_l1355_135554


namespace NUMINAMATH_CALUDE_sixth_quiz_score_l1355_135570

theorem sixth_quiz_score (scores : List ℕ) (target_mean : ℕ) : 
  scores = [86, 90, 82, 84, 95] →
  target_mean = 95 →
  ∃ (sixth_score : ℕ), 
    sixth_score = 133 ∧ 
    (scores.sum + sixth_score) / 6 = target_mean :=
by sorry

end NUMINAMATH_CALUDE_sixth_quiz_score_l1355_135570


namespace NUMINAMATH_CALUDE_one_eighth_of_2_36_l1355_135531

theorem one_eighth_of_2_36 (y : ℤ) : (1 / 8 : ℚ) * (2 ^ 36) = 2 ^ y → y = 33 := by
  sorry

end NUMINAMATH_CALUDE_one_eighth_of_2_36_l1355_135531


namespace NUMINAMATH_CALUDE_escalator_length_l1355_135549

/-- The length of an escalator given specific conditions -/
theorem escalator_length : 
  ∀ (escalator_speed person_speed time : ℝ) (length : ℝ),
  escalator_speed = 12 →
  person_speed = 2 →
  time = 14 →
  length = (escalator_speed + person_speed) * time →
  length = 196 := by
sorry

end NUMINAMATH_CALUDE_escalator_length_l1355_135549


namespace NUMINAMATH_CALUDE_quadratic_function_range_l1355_135585

theorem quadratic_function_range (a b : ℝ) :
  let f := fun x => a * x^2 + b * x
  (1 ≤ f (-1) ∧ f (-1) ≤ 2) →
  (2 ≤ f 1 ∧ f 1 ≤ 4) →
  (5 ≤ f (-2) ∧ f (-2) ≤ 10) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l1355_135585


namespace NUMINAMATH_CALUDE_oil_purchase_calculation_l1355_135563

theorem oil_purchase_calculation (tank_capacity : ℕ) (tanks_needed : ℕ) (total_oil : ℕ) : 
  tank_capacity = 32 → tanks_needed = 23 → total_oil = tank_capacity * tanks_needed → total_oil = 736 :=
by sorry

end NUMINAMATH_CALUDE_oil_purchase_calculation_l1355_135563


namespace NUMINAMATH_CALUDE_expression_value_l1355_135581

theorem expression_value : 
  let x : ℤ := -1
  let y : ℤ := 2
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1355_135581


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l1355_135546

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The condition that the polynomial satisfies the given equation -/
def SatisfiesEquation (P : RealPolynomial) : Prop :=
  ∀ (x y z : ℝ), x ≠ 0 → y ≠ 0 → z ≠ 0 → 2*x*y*z = x + y + z →
    P x / (y*z) + P y / (z*x) + P z / (x*y) = P (x - y) + P (y - z) + P (z - x)

/-- The theorem stating that any polynomial satisfying the equation must be of the form c(x^2 + 3) -/
theorem polynomial_equation_solution (P : RealPolynomial) 
    (h : SatisfiesEquation P) : 
    ∃ (c : ℝ), ∀ x, P x = c * (x^2 + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l1355_135546


namespace NUMINAMATH_CALUDE_problem_solution_l1355_135519

def A : Set ℝ := {x | (x - 1/2) * (x - 3) = 0}

def B (a : ℝ) : Set ℝ := {x | Real.log (x^2 + a*x + a + 9/4) = 0}

theorem problem_solution :
  (∀ a : ℝ, (∃! x : ℝ, x ∈ B a) → (a = 5 ∨ a = -1)) ∧
  (∀ a : ℝ, B a ⊂ A → a ∈ Set.Icc (-1) 5) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1355_135519


namespace NUMINAMATH_CALUDE_spending_vs_earning_difference_l1355_135536

def initial_amount : Int := 153
def part_time_earnings : Int := 65
def atm_collection : Int := 195
def supermarket_spending : Int := 87
def electronics_spending : Int := 134
def clothes_spending : Int := 78

theorem spending_vs_earning_difference :
  (supermarket_spending + electronics_spending + clothes_spending) -
  (part_time_earnings + atm_collection) = -39 :=
by sorry

end NUMINAMATH_CALUDE_spending_vs_earning_difference_l1355_135536


namespace NUMINAMATH_CALUDE_probability_both_boys_given_one_boy_l1355_135548

/-- Represents the gender of a child -/
inductive Gender
  | Boy
  | Girl

/-- Represents a family with two children -/
structure Family :=
  (child1 : Gender)
  (child2 : Gender)

/-- The set of all possible families with two children -/
def allFamilies : Finset Family :=
  sorry

/-- The set of families with at least one boy -/
def familiesWithBoy : Finset Family :=
  sorry

/-- The set of families with two boys -/
def familiesWithTwoBoys : Finset Family :=
  sorry

theorem probability_both_boys_given_one_boy :
    (familiesWithTwoBoys.card : ℚ) / familiesWithBoy.card = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_boys_given_one_boy_l1355_135548


namespace NUMINAMATH_CALUDE_company_assets_and_price_l1355_135504

theorem company_assets_and_price (A B P : ℝ) 
  (h1 : P = 1.5 * A) 
  (h2 : P = 0.8571428571428571 * (A + B)) : 
  P = 2 * B := by
sorry

end NUMINAMATH_CALUDE_company_assets_and_price_l1355_135504


namespace NUMINAMATH_CALUDE_maria_car_trip_l1355_135584

theorem maria_car_trip (total_distance : ℝ) (first_stop_fraction : ℝ) (second_stop_fraction : ℝ) :
  total_distance = 560 ∧ 
  first_stop_fraction = 1/2 ∧ 
  second_stop_fraction = 1/4 →
  total_distance - (first_stop_fraction * total_distance) - 
    (second_stop_fraction * (total_distance - first_stop_fraction * total_distance)) = 210 := by
  sorry

end NUMINAMATH_CALUDE_maria_car_trip_l1355_135584


namespace NUMINAMATH_CALUDE_book_arrangements_count_l1355_135564

/-- The number of ways to arrange 8 different books (3 math, 3 foreign language, 2 literature)
    such that all math books are together and all foreign language books are together. -/
def book_arrangements : ℕ :=
  let total_books : ℕ := 8
  let math_books : ℕ := 3
  let foreign_books : ℕ := 3
  let literature_books : ℕ := 2
  sorry

/-- Theorem stating that the number of book arrangements is 864. -/
theorem book_arrangements_count : book_arrangements = 864 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangements_count_l1355_135564


namespace NUMINAMATH_CALUDE_inequality_and_existence_l1355_135510

theorem inequality_and_existence : 
  (∀ x y z : ℝ, x^2 + 2*y^2 + 3*z^2 ≥ Real.sqrt 3 * (x*y + y*z + z*x)) ∧ 
  (∃ k : ℝ, k > Real.sqrt 3 ∧ (∀ x y z : ℝ, x^2 + 2*y^2 + 3*z^2 ≥ k * (x*y + y*z + z*x))) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_existence_l1355_135510


namespace NUMINAMATH_CALUDE_remainder_theorem_l1355_135508

theorem remainder_theorem (x : ℝ) : ∃ (Q : ℝ → ℝ) (R : ℝ → ℝ),
  (∀ x, x^100 = (x^2 - 3*x + 2) * Q x + R x) ∧
  (∃ a b, R = fun x ↦ a * x + b) ∧
  R = fun x ↦ 2^100 * (x - 1) - (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1355_135508


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1355_135568

theorem decimal_to_fraction :
  (3.75 : ℚ) = 15 / 4 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1355_135568


namespace NUMINAMATH_CALUDE_truncated_pyramid_edge_count_l1355_135539

/-- A square-based pyramid with truncated vertices -/
structure TruncatedPyramid where
  /-- The number of vertices in the original square-based pyramid -/
  original_vertices : Nat
  /-- The number of edges in the original square-based pyramid -/
  original_edges : Nat
  /-- The number of new edges created by each truncation -/
  new_edges_per_truncation : Nat
  /-- Assertion that the original shape is a square-based pyramid -/
  is_square_based_pyramid : original_vertices = 5 ∧ original_edges = 8
  /-- Assertion that each truncation creates a triangular face -/
  truncation_creates_triangle : new_edges_per_truncation = 3

/-- Theorem stating that a truncated square-based pyramid has 23 edges -/
theorem truncated_pyramid_edge_count (p : TruncatedPyramid) :
  p.original_edges + p.original_vertices * p.new_edges_per_truncation = 23 :=
by sorry

end NUMINAMATH_CALUDE_truncated_pyramid_edge_count_l1355_135539


namespace NUMINAMATH_CALUDE_fourth_grade_student_count_l1355_135529

/-- The number of students at the end of the year in fourth grade -/
def final_student_count (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Theorem: Given the initial conditions, the final student count is 29 -/
theorem fourth_grade_student_count :
  final_student_count 33 18 14 = 29 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_student_count_l1355_135529


namespace NUMINAMATH_CALUDE_complex_power_difference_l1355_135565

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : (1 + i)^16 - (1 - i)^16 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l1355_135565


namespace NUMINAMATH_CALUDE_complex_addition_l1355_135560

theorem complex_addition : (2 : ℂ) + 5*I + (3 : ℂ) - 7*I = (5 : ℂ) - 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_addition_l1355_135560


namespace NUMINAMATH_CALUDE_sum_abs_diff_second_quadrant_l1355_135578

/-- A point in the second quadrant of the Cartesian coordinate system -/
structure SecondQuadrantPoint where
  a : ℝ
  b : ℝ
  h1 : a < 0
  h2 : b > 0

/-- The sum of absolute differences for a point in the second quadrant -/
def sumAbsDiff (p : SecondQuadrantPoint) : ℝ :=
  |p.a - p.b| + |p.b - p.a|

theorem sum_abs_diff_second_quadrant (p : SecondQuadrantPoint) :
  sumAbsDiff p = -2 * p.a + 2 * p.b := by sorry

end NUMINAMATH_CALUDE_sum_abs_diff_second_quadrant_l1355_135578


namespace NUMINAMATH_CALUDE_power_difference_equals_one_ninth_l1355_135580

theorem power_difference_equals_one_ninth (x y : ℕ) : 
  (2^x : ℕ) ∣ 360 ∧ 
  ∀ k > x, ¬((2^k : ℕ) ∣ 360) ∧ 
  (5^y : ℕ) ∣ 360 ∧ 
  ∀ m > y, ¬((5^m : ℕ) ∣ 360) → 
  (1/3 : ℚ)^(x - y) = 1/9 := by
sorry

end NUMINAMATH_CALUDE_power_difference_equals_one_ninth_l1355_135580


namespace NUMINAMATH_CALUDE_parallel_vectors_x_coord_l1355_135505

/-- Given vectors a and b in ℝ², if a + b is parallel to a - 2b, then the x-coordinate of b is 4. -/
theorem parallel_vectors_x_coord (a b : ℝ × ℝ) (h : a.1 = 2 ∧ a.2 = 1 ∧ b.2 = 2) :
  (∃ k : ℝ, (a.1 + b.1, a.2 + b.2) = k • (a.1 - 2 * b.1, a.2 - 2 * b.2)) →
  b.1 = 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_coord_l1355_135505


namespace NUMINAMATH_CALUDE_sum_product_bounds_l1355_135509

theorem sum_product_bounds (x y z : ℝ) (h : x + y + z = 1) :
  ∃ (min max : ℝ), min = -1/4 ∧ max = 1/2 ∧
  (xy + xz + yz ≥ min ∧ xy + xz + yz ≤ max) ∧
  ∀ t, min ≤ t ∧ t ≤ max → ∃ (a b c : ℝ), a + b + c = 1 ∧ ab + ac + bc = t :=
sorry

end NUMINAMATH_CALUDE_sum_product_bounds_l1355_135509


namespace NUMINAMATH_CALUDE_line_intersection_theorem_l1355_135500

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of lines being skew
variable (skew : Line → Line → Prop)

-- Define the property of a line being contained in a plane
variable (contained_in : Line → Plane → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the property of a line intersecting another line
variable (intersects : Line → Line → Prop)

theorem line_intersection_theorem 
  (a b m : Line) (α β : Plane)
  (h1 : skew a b)
  (h2 : contained_in a α)
  (h3 : contained_in b β)
  (h4 : intersect α β = m) :
  intersects m a ∨ intersects m b :=
sorry

end NUMINAMATH_CALUDE_line_intersection_theorem_l1355_135500


namespace NUMINAMATH_CALUDE_license_plate_count_l1355_135598

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The total number of characters in a license plate -/
def total_chars : ℕ := 7

/-- The number of digits in a license plate -/
def num_digits_in_plate : ℕ := 5

/-- The number of letters in a license plate -/
def num_letters_in_plate : ℕ := 2

/-- The number of distinct license plates -/
def num_distinct_plates : ℕ := 1420200000

theorem license_plate_count :
  (total_chars.choose num_letters_in_plate) * 
  (num_letters ^ num_letters_in_plate) * 
  (num_digits ^ num_digits_in_plate) = num_distinct_plates :=
sorry

end NUMINAMATH_CALUDE_license_plate_count_l1355_135598


namespace NUMINAMATH_CALUDE_inequality_solution_and_a_range_l1355_135588

def f (x : ℝ) := |3*x + 2|

theorem inequality_solution_and_a_range :
  (∀ x : ℝ, f x < 6 - |x - 2| ↔ -3/2 < x ∧ x < 1) ∧
  (∀ m n : ℝ, m > 0 → n > 0 → m + n = 4 →
    (∀ a : ℝ, a > 0 →
      (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) →
        0 < a ∧ a ≤ 1/3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_and_a_range_l1355_135588


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1355_135527

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 2/y ≥ 1/a + 2/b) →
  1/a + 2/b = 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1355_135527


namespace NUMINAMATH_CALUDE_inverse_proposition_correct_l1355_135576

/-- The statement of a geometric proposition -/
structure GeometricProposition :=
  (hypothesis : String)
  (conclusion : String)

/-- The inverse of a geometric proposition -/
def inverse_proposition (p : GeometricProposition) : GeometricProposition :=
  { hypothesis := p.conclusion,
    conclusion := p.hypothesis }

/-- The original proposition -/
def original_prop : GeometricProposition :=
  { hypothesis := "Triangles are congruent",
    conclusion := "Corresponding angles are equal" }

/-- Theorem stating that the inverse proposition is correct -/
theorem inverse_proposition_correct : 
  inverse_proposition original_prop = 
  { hypothesis := "Corresponding angles are equal",
    conclusion := "Triangles are congruent" } := by
  sorry


end NUMINAMATH_CALUDE_inverse_proposition_correct_l1355_135576


namespace NUMINAMATH_CALUDE_set_intersection_problem_l1355_135522

theorem set_intersection_problem (m : ℝ) : 
  let A : Set ℝ := {-1, 3, m}
  let B : Set ℝ := {3, 4}
  B ∩ A = B → m = 4 := by
sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l1355_135522


namespace NUMINAMATH_CALUDE_certain_number_problem_l1355_135542

theorem certain_number_problem : ∃ x : ℕ, 3*15 + 3*16 + 3*19 + x = 161 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1355_135542


namespace NUMINAMATH_CALUDE_square_root_problem_l1355_135582

theorem square_root_problem (a b : ℝ) : 
  (∀ x : ℝ, x^2 = a + 11 → x = 1 ∨ x = -1) → 
  ((1 - b).sqrt = 4) → 
  (a = -10 ∧ b = -15 ∧ (2*a + 7*b)^(1/3 : ℝ) = -5) := by
sorry

end NUMINAMATH_CALUDE_square_root_problem_l1355_135582


namespace NUMINAMATH_CALUDE_converse_x_squared_greater_than_one_l1355_135567

theorem converse_x_squared_greater_than_one (x : ℝ) :
  x^2 > 1 → (x < -1 ∨ x > 1) :=
sorry

end NUMINAMATH_CALUDE_converse_x_squared_greater_than_one_l1355_135567


namespace NUMINAMATH_CALUDE_seventh_root_of_unity_product_l1355_135507

theorem seventh_root_of_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^5 - 1) * (r^6 - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_of_unity_product_l1355_135507


namespace NUMINAMATH_CALUDE_ramanujan_number_l1355_135596

theorem ramanujan_number (hardy_number ramanujan_number : ℂ) : 
  hardy_number * ramanujan_number = 48 - 24 * I ∧ 
  hardy_number = 6 + I → 
  ramanujan_number = (312 - 432 * I) / 37 := by
  sorry

end NUMINAMATH_CALUDE_ramanujan_number_l1355_135596


namespace NUMINAMATH_CALUDE_combined_earnings_proof_l1355_135523

/-- Given Dwayne's annual earnings and the difference between Brady's and Dwayne's earnings,
    calculate their combined annual earnings. -/
def combinedEarnings (dwayneEarnings : ℕ) (earningsDifference : ℕ) : ℕ :=
  dwayneEarnings + (dwayneEarnings + earningsDifference)

/-- Theorem stating that given the specific values from the problem,
    the combined earnings of Brady and Dwayne are $3450. -/
theorem combined_earnings_proof :
  combinedEarnings 1500 450 = 3450 := by
  sorry

end NUMINAMATH_CALUDE_combined_earnings_proof_l1355_135523


namespace NUMINAMATH_CALUDE_factorial_divisibility_l1355_135516

theorem factorial_divisibility (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 100) :
  ∃ k : ℕ, (Nat.factorial (n^2 + 1)) = k * (Nat.factorial n)^(n + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l1355_135516


namespace NUMINAMATH_CALUDE_function_value_at_negative_five_l1355_135557

/-- Given a function f(x) = ax + b * sin(x) + 1 where f(5) = 7, prove that f(-5) = -5 -/
theorem function_value_at_negative_five 
  (a b : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x + b * Real.sin x + 1)
  (h2 : f 5 = 7) : 
  f (-5) = -5 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_five_l1355_135557


namespace NUMINAMATH_CALUDE_recurrence_relation_solution_l1355_135501

def a (n : ℕ) : ℤ := 2 * 4^n - 2*n + 2
def b (n : ℕ) : ℤ := 2 * 4^n + 2*n - 2

theorem recurrence_relation_solution :
  (∀ n : ℕ, a (n + 1) = 3 * a n + b n - 4) ∧
  (∀ n : ℕ, b (n + 1) = 2 * a n + 2 * b n + 2) ∧
  a 0 = 4 ∧
  b 0 = 0 := by sorry

end NUMINAMATH_CALUDE_recurrence_relation_solution_l1355_135501


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l1355_135543

/- Define the concept of opposite for integers -/
def opposite (n : Int) : Int := -n

/- Theorem: The opposite of -2023 is 2023 -/
theorem opposite_of_negative_2023 : opposite (-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l1355_135543


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1355_135538

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1355_135538


namespace NUMINAMATH_CALUDE_sum_four_digit_numbers_eq_179982_l1355_135512

/-- The sum of all four-digit numbers created using digits 1, 2, and 3 with repetition -/
def sum_four_digit_numbers : ℕ :=
  let digits : List ℕ := [1, 2, 3]
  let total_numbers : ℕ := digits.length ^ 4
  let sum_per_position : ℕ := (digits.sum * total_numbers) / digits.length
  sum_per_position * 1000 + sum_per_position * 100 + sum_per_position * 10 + sum_per_position

theorem sum_four_digit_numbers_eq_179982 :
  sum_four_digit_numbers = 179982 := by
  sorry

#eval sum_four_digit_numbers

end NUMINAMATH_CALUDE_sum_four_digit_numbers_eq_179982_l1355_135512


namespace NUMINAMATH_CALUDE_polygon_sides_l1355_135545

theorem polygon_sides (n : ℕ) : n = 8 ↔ 
  (n - 2) * 180 = 3 * 360 := by sorry

end NUMINAMATH_CALUDE_polygon_sides_l1355_135545


namespace NUMINAMATH_CALUDE_square_area_increase_l1355_135553

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let new_side := 1.6 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 1.56 :=
by sorry

end NUMINAMATH_CALUDE_square_area_increase_l1355_135553


namespace NUMINAMATH_CALUDE_mans_upstream_rate_l1355_135574

/-- Prove that given a man's downstream rate, his rate in still water, and the current rate, his upstream rate can be calculated. -/
theorem mans_upstream_rate (downstream_rate still_water_rate current_rate : ℝ) 
  (h1 : downstream_rate = 32)
  (h2 : still_water_rate = 24.5)
  (h3 : current_rate = 7.5) :
  still_water_rate - current_rate = 17 := by
  sorry

end NUMINAMATH_CALUDE_mans_upstream_rate_l1355_135574


namespace NUMINAMATH_CALUDE_book_pricing_deduction_percentage_l1355_135577

theorem book_pricing_deduction_percentage
  (cost_price : ℝ)
  (profit_percentage : ℝ)
  (list_price : ℝ)
  (h1 : cost_price = 47.50)
  (h2 : profit_percentage = 25)
  (h3 : list_price = 69.85) :
  let selling_price := cost_price * (1 + profit_percentage / 100)
  let deduction_percentage := (list_price - selling_price) / list_price * 100
  deduction_percentage = 15 := by
sorry

end NUMINAMATH_CALUDE_book_pricing_deduction_percentage_l1355_135577


namespace NUMINAMATH_CALUDE_bathroom_length_l1355_135552

theorem bathroom_length (area : ℝ) (width : ℝ) (length : ℝ) : 
  area = 8 → width = 2 → area = length * width → length = 4 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_length_l1355_135552


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1355_135587

theorem fraction_evaluation : 
  let numerator := (12^4 + 288) * (24^4 + 288) * (36^4 + 288) * (48^4 + 288) * (60^4 + 288)
  let denominator := (6^4 + 288) * (18^4 + 288) * (30^4 + 288) * (42^4 + 288) * (54^4 + 288)
  numerator / denominator = -332 := by
sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1355_135587


namespace NUMINAMATH_CALUDE_disk_arrangement_area_l1355_135524

/-- The total area of eight congruent disks arranged around a square -/
theorem disk_arrangement_area (s : ℝ) (h : s = 2) : 
  let r := s / 2
  8 * π * r^2 = 4 * π := by
  sorry

end NUMINAMATH_CALUDE_disk_arrangement_area_l1355_135524


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1355_135594

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b + 1 → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(a > b + 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1355_135594


namespace NUMINAMATH_CALUDE_alternating_sum_fraction_l1355_135555

theorem alternating_sum_fraction :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) /
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sum_fraction_l1355_135555


namespace NUMINAMATH_CALUDE_max_value_three_power_minus_nine_power_l1355_135573

theorem max_value_three_power_minus_nine_power (x : ℝ) :
  ∃ (max : ℝ), max = (1 : ℝ) / 4 ∧ ∀ y : ℝ, 3^y - 9^y ≤ max :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_three_power_minus_nine_power_l1355_135573


namespace NUMINAMATH_CALUDE_age_problem_l1355_135590

theorem age_problem (age_older age_younger : ℕ) : 
  age_older = age_younger + 2 →
  age_older + age_younger = 74 →
  age_older = 38 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l1355_135590


namespace NUMINAMATH_CALUDE_fractional_exponent_simplification_l1355_135540

theorem fractional_exponent_simplification :
  (2^2024 + 2^2020) / (2^2024 - 2^2020) = 17 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fractional_exponent_simplification_l1355_135540


namespace NUMINAMATH_CALUDE_fraction_value_l1355_135599

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 3 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  a * c / (b * d) = 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1355_135599


namespace NUMINAMATH_CALUDE_robins_gum_problem_l1355_135535

theorem robins_gum_problem (initial_gum : ℕ) (final_gum : ℕ) (h1 : initial_gum = 18) (h2 : final_gum = 44) :
  final_gum - initial_gum = 26 := by
  sorry

end NUMINAMATH_CALUDE_robins_gum_problem_l1355_135535


namespace NUMINAMATH_CALUDE_circle_equation_sum_l1355_135586

/-- Given a circle equation, prove the sum of center coordinates and radius -/
theorem circle_equation_sum (x y : ℝ) :
  (∀ x y, x^2 + 14*y + 65 = -y^2 - 8*x) →
  ∃ a b r : ℝ,
    (∀ x y, (x - a)^2 + (y - b)^2 = r^2) ∧
    a + b + r = -11 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_sum_l1355_135586


namespace NUMINAMATH_CALUDE_no_valid_solution_l1355_135544

/-- Represents a mapping from letters to digits -/
def LetterDigitMap := Char → Fin 10

/-- Checks if a LetterDigitMap assigns unique digits to different letters -/
def is_valid_map (m : LetterDigitMap) : Prop :=
  ∀ c₁ c₂ : Char, c₁ ≠ c₂ → m c₁ ≠ m c₂

/-- Evaluates the left-hand side of the equation -/
def lhs (m : LetterDigitMap) : ℕ :=
  (m 'Ш') * (m 'Е') * (m 'С') * (m 'Т') * (m 'Ь') + 1

/-- Evaluates the right-hand side of the equation -/
def rhs (m : LetterDigitMap) : ℕ :=
  (m 'C') * (m 'E') * (m 'M') * (m 'b')

/-- The main theorem stating that no valid mapping exists to satisfy the equation -/
theorem no_valid_solution : ¬∃ m : LetterDigitMap, is_valid_map m ∧ lhs m = rhs m := by
  sorry

end NUMINAMATH_CALUDE_no_valid_solution_l1355_135544


namespace NUMINAMATH_CALUDE_constant_ratio_sum_l1355_135503

theorem constant_ratio_sum (x₁ x₂ x₃ x₄ : ℝ) (k : ℝ) 
  (h_not_all_equal : ¬(x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄))
  (h_ratio_12_34 : (x₁ + x₂) / (x₃ + x₄) = k)
  (h_ratio_13_24 : (x₁ + x₃) / (x₂ + x₄) = k)
  (h_ratio_14_23 : (x₁ + x₄) / (x₂ + x₃) = k)
  (h_ratio_34_12 : (x₃ + x₄) / (x₁ + x₂) = k)
  (h_ratio_24_13 : (x₂ + x₄) / (x₁ + x₃) = k)
  (h_ratio_23_14 : (x₂ + x₃) / (x₁ + x₄) = k) :
  k = -1 := by sorry

end NUMINAMATH_CALUDE_constant_ratio_sum_l1355_135503


namespace NUMINAMATH_CALUDE_division_problem_l1355_135558

theorem division_problem : ∃ (n : ℕ), n ≠ 0 ∧ 45 = 11 * n + 1 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1355_135558


namespace NUMINAMATH_CALUDE_octagon_diagonals_l1355_135550

/-- The number of diagonals in a polygon with n vertices -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 vertices -/
def octagon_vertices : ℕ := 8

theorem octagon_diagonals : num_diagonals octagon_vertices = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l1355_135550


namespace NUMINAMATH_CALUDE_square_sum_of_product_and_sum_l1355_135579

theorem square_sum_of_product_and_sum (m n : ℝ) 
  (h1 : m * n = 12) 
  (h2 : m + n = 8) : 
  m^2 + n^2 = 40 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_product_and_sum_l1355_135579


namespace NUMINAMATH_CALUDE_quadratic_properties_l1355_135572

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a ≠ 0
  hpos : ∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0
  hsym : -b / (2 * a) = 2
  hintercept : ∃ x, x > 0 ∧ a * x^2 + b * x + c = 0 ∧ |c| = x

/-- Theorem stating properties of the quadratic function -/
theorem quadratic_properties (f : QuadraticFunction) :
  f.c > -1 ∧ f.a * (-f.c)^2 + f.b * (-f.c) + f.c = 0 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_properties_l1355_135572


namespace NUMINAMATH_CALUDE_product_of_fractions_l1355_135592

theorem product_of_fractions : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1355_135592


namespace NUMINAMATH_CALUDE_equation_solution_l1355_135556

theorem equation_solution : ∃ x : ℝ, x > 0 ∧ 5 * (x^(1/4))^2 - (3*x)/(x^(3/4)) = 10 + 2 * x^(1/4) ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1355_135556


namespace NUMINAMATH_CALUDE_eight_triangle_positions_l1355_135534

/-- A point on a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- The area of a triangle given three grid points -/
def triangleArea (a b c : GridPoint) : ℚ :=
  (1/2) * |a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)|

/-- Theorem: There are exactly 8 points C on the grid such that triangle ABC has area 4.5 -/
theorem eight_triangle_positions (a b : GridPoint) : 
  ∃! (s : Finset GridPoint), s.card = 8 ∧ 
    (∀ c ∈ s, triangleArea a b c = 4.5) ∧
    (∀ c : GridPoint, c ∉ s → triangleArea a b c ≠ 4.5) :=
sorry

end NUMINAMATH_CALUDE_eight_triangle_positions_l1355_135534


namespace NUMINAMATH_CALUDE_perpendicular_points_coplanar_l1355_135566

-- Define the types for points and spheres
variable (Point Sphere : Type)

-- Define the property of a point being on a sphere
variable (onSphere : Point → Sphere → Prop)

-- Define the property of points being distinct
variable (distinct : List Point → Prop)

-- Define the property of points being coplanar
variable (coplanar : List Point → Prop)

-- Define the property of a point being on a line
variable (onLine : Point → Point → Point → Prop)

-- Define the property of a line being perpendicular to another line
variable (perpendicular : Point → Point → Point → Point → Prop)

-- Define the quadrilateral pyramid inscribed in a sphere
variable (S A B C D : Point) (sphere1 : Sphere)
variable (inscribed : onSphere S sphere1 ∧ onSphere A sphere1 ∧ onSphere B sphere1 ∧ onSphere C sphere1 ∧ onSphere D sphere1)

-- Define the perpendicular points
variable (A1 B1 C1 D1 : Point)
variable (perp : perpendicular A A1 S C ∧ perpendicular B B1 S D ∧ perpendicular C C1 S A ∧ perpendicular D D1 S B)

-- Define the property of perpendicular points being on the respective lines
variable (onLines : onLine A1 S C ∧ onLine B1 S D ∧ onLine C1 S A ∧ onLine D1 S B)

-- Define the property of S, A1, B1, C1, D1 being distinct and on another sphere
variable (sphere2 : Sphere)
variable (distinctOnSphere : distinct [S, A1, B1, C1, D1] ∧ 
                             onSphere S sphere2 ∧ onSphere A1 sphere2 ∧ onSphere B1 sphere2 ∧ onSphere C1 sphere2 ∧ onSphere D1 sphere2)

-- Theorem statement
theorem perpendicular_points_coplanar : 
  coplanar [A1, B1, C1, D1] :=
sorry

end NUMINAMATH_CALUDE_perpendicular_points_coplanar_l1355_135566


namespace NUMINAMATH_CALUDE_min_coefficient_value_l1355_135569

theorem min_coefficient_value (c d box : ℤ) : 
  (∀ x : ℝ, (c * x + d) * (d * x + c) = 29 * x^2 + box * x + 29) →
  c ≠ d ∧ c ≠ box ∧ d ≠ box →
  ∀ b : ℤ, (∀ x : ℝ, (c * x + d) * (d * x + c) = 29 * x^2 + b * x + 29) → box ≤ b →
  box = 842 :=
by sorry

end NUMINAMATH_CALUDE_min_coefficient_value_l1355_135569


namespace NUMINAMATH_CALUDE_price_decrease_sales_increase_ratio_l1355_135575

theorem price_decrease_sales_increase_ratio (P U : ℝ) (h_positive : P > 0 ∧ U > 0) :
  let new_price := 0.8 * P
  let new_units := U / 0.8
  let revenue_unchanged := P * U = new_price * new_units
  let percent_decrease_price := 20
  let percent_increase_units := (new_units - U) / U * 100
  revenue_unchanged →
  percent_increase_units / percent_decrease_price = 1.25 := by
sorry

end NUMINAMATH_CALUDE_price_decrease_sales_increase_ratio_l1355_135575


namespace NUMINAMATH_CALUDE_tree_distance_l1355_135583

/-- The distance between consecutive trees in a yard with an obstacle -/
theorem tree_distance (yard_length : ℝ) (num_trees : ℕ) (obstacle_gap : ℝ) :
  yard_length = 600 →
  num_trees = 36 →
  obstacle_gap = 10 →
  (yard_length - obstacle_gap) / (num_trees - 1 : ℝ) = 590 / 35 := by
  sorry

end NUMINAMATH_CALUDE_tree_distance_l1355_135583


namespace NUMINAMATH_CALUDE_vector_BC_calculation_l1355_135593

-- Define the points and vectors
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (3, 2)
def AC : ℝ × ℝ := (-4, -3)

-- Define the theorem
theorem vector_BC_calculation (A B : ℝ × ℝ) (AC : ℝ × ℝ) : 
  A = (0, 1) → B = (3, 2) → AC = (-4, -3) → 
  (B.1 - (A.1 + AC.1), B.2 - (A.2 + AC.2)) = (-7, -4) := by
  sorry

end NUMINAMATH_CALUDE_vector_BC_calculation_l1355_135593


namespace NUMINAMATH_CALUDE_advance_ticket_cost_l1355_135511

/-- The cost of advance tickets is $20, given the specified conditions. -/
theorem advance_ticket_cost (same_day_cost : ℕ) (total_tickets : ℕ) (total_receipts : ℕ) (advance_tickets_sold : ℕ) :
  same_day_cost = 30 →
  total_tickets = 60 →
  total_receipts = 1600 →
  advance_tickets_sold = 20 →
  ∃ (advance_cost : ℕ), advance_cost * advance_tickets_sold + same_day_cost * (total_tickets - advance_tickets_sold) = total_receipts ∧ advance_cost = 20 :=
by sorry

end NUMINAMATH_CALUDE_advance_ticket_cost_l1355_135511


namespace NUMINAMATH_CALUDE_health_risk_factors_l1355_135589

theorem health_risk_factors (total_population : ℝ) 
  (prob_single : ℝ) (prob_pair : ℝ) (prob_all_given_two : ℝ) :
  prob_single = 0.08 →
  prob_pair = 0.15 →
  prob_all_given_two = 1/4 →
  ∃ (prob_none_given_not_one : ℝ),
    prob_none_given_not_one = 26/57 := by
  sorry

end NUMINAMATH_CALUDE_health_risk_factors_l1355_135589


namespace NUMINAMATH_CALUDE_three_hour_therapy_charge_l1355_135526

/-- Represents the pricing structure and total charges for a psychologist's therapy sessions. -/
structure TherapyPricing where
  firstHourCharge : ℕ
  additionalHourCharge : ℕ
  hoursForStandardSession : ℕ
  totalChargeForStandardSession : ℕ

/-- Calculates the total charge for a given number of therapy hours. -/
def totalCharge (pricing : TherapyPricing) (hours : ℕ) : ℕ :=
  pricing.firstHourCharge + (hours - 1) * pricing.additionalHourCharge

/-- Theorem stating that given the conditions, the total charge for 3 hours of therapy is $188. -/
theorem three_hour_therapy_charge 
  (pricing : TherapyPricing) 
  (h1 : pricing.firstHourCharge = pricing.additionalHourCharge + 20)
  (h2 : pricing.hoursForStandardSession = 5)
  (h3 : pricing.totalChargeForStandardSession = 300)
  (h4 : totalCharge pricing pricing.hoursForStandardSession = pricing.totalChargeForStandardSession) :
  totalCharge pricing 3 = 188 :=
by
  sorry


end NUMINAMATH_CALUDE_three_hour_therapy_charge_l1355_135526


namespace NUMINAMATH_CALUDE_nickel_count_is_three_l1355_135518

/-- Represents the number of coins of each type -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- The total value of coins in cents -/
def totalValue (c : CoinCount) : ℕ :=
  c.pennies * 1 + c.nickels * 5 + c.dimes * 10

/-- The total number of coins -/
def totalCoins (c : CoinCount) : ℕ :=
  c.pennies + c.nickels + c.dimes

theorem nickel_count_is_three :
  ∃ (c : CoinCount),
    totalCoins c = 8 ∧
    totalValue c = 47 ∧
    c.pennies ≥ 1 ∧
    c.nickels ≥ 1 ∧
    c.dimes ≥ 1 ∧
    (∀ (c' : CoinCount),
      totalCoins c' = 8 →
      totalValue c' = 47 →
      c'.pennies ≥ 1 →
      c'.nickels ≥ 1 →
      c'.dimes ≥ 1 →
      c'.nickels = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_nickel_count_is_three_l1355_135518


namespace NUMINAMATH_CALUDE_tea_box_theorem_l1355_135513

/-- The amount of tea leaves in a box, given daily consumption and duration -/
def tea_box_amount (daily_consumption : ℚ) (weeks : ℕ) : ℚ :=
  daily_consumption * 7 * weeks

/-- Theorem: A box of tea leaves containing 28 ounces lasts 20 weeks with 1/5 ounce daily consumption -/
theorem tea_box_theorem :
  tea_box_amount (1/5) 20 = 28 := by
  sorry

#eval tea_box_amount (1/5) 20

end NUMINAMATH_CALUDE_tea_box_theorem_l1355_135513


namespace NUMINAMATH_CALUDE_wall_width_l1355_135571

/-- Given a rectangular wall with specific proportions and volume, prove its width --/
theorem wall_width (w h l : ℝ) (h_height : h = 6 * w) (h_length : l = 7 * h) 
  (h_volume : w * h * l = 86436) : w = 7 := by
  sorry

end NUMINAMATH_CALUDE_wall_width_l1355_135571


namespace NUMINAMATH_CALUDE_food_boxes_l1355_135517

theorem food_boxes (total_food : ℝ) (food_per_box : ℝ) (h1 : total_food = 777.5) (h2 : food_per_box = 2.25) :
  ⌊total_food / food_per_box⌋ = 345 := by
  sorry

end NUMINAMATH_CALUDE_food_boxes_l1355_135517


namespace NUMINAMATH_CALUDE_pant_price_before_discount_l1355_135532

/-- The cost of a wardrobe given specific items and prices --/
def wardrobe_cost (skirt_price blouse_price pant_price : ℚ) : ℚ :=
  3 * skirt_price + 5 * blouse_price + (pant_price + pant_price / 2)

/-- Theorem stating the cost of pants before discount --/
theorem pant_price_before_discount :
  ∃ (pant_price : ℚ),
    wardrobe_cost 20 15 pant_price = 180 ∧
    pant_price = 30 :=
by
  sorry

#check pant_price_before_discount

end NUMINAMATH_CALUDE_pant_price_before_discount_l1355_135532


namespace NUMINAMATH_CALUDE_perpendicular_slope_l1355_135502

theorem perpendicular_slope (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let original_slope := -a / b
  let perpendicular_slope := -1 / original_slope
  perpendicular_slope = b / a :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l1355_135502


namespace NUMINAMATH_CALUDE_max_z_value_l1355_135551

theorem max_z_value (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x*y + y*z + z*x = 3) :
  z ≤ 13/3 := by
sorry

end NUMINAMATH_CALUDE_max_z_value_l1355_135551


namespace NUMINAMATH_CALUDE_harry_average_sleep_time_l1355_135595

/-- Harry's sleep schedule for a week --/
structure SleepSchedule where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculate the average sleep time --/
def averageSleepTime (schedule : SleepSchedule) : ℚ :=
  (schedule.monday + schedule.tuesday + schedule.wednesday + schedule.thursday + schedule.friday) / 5

/-- Harry's actual sleep schedule --/
def harrySleepSchedule : SleepSchedule := {
  monday := 8,
  tuesday := 7,
  wednesday := 8,
  thursday := 10,
  friday := 7
}

/-- Theorem: Harry's average sleep time is 8 hours --/
theorem harry_average_sleep_time :
  averageSleepTime harrySleepSchedule = 8 := by
  sorry

end NUMINAMATH_CALUDE_harry_average_sleep_time_l1355_135595


namespace NUMINAMATH_CALUDE_mechanics_billing_problem_l1355_135530

/-- A mechanic's billing problem -/
theorem mechanics_billing_problem 
  (total_bill : ℝ) 
  (parts_cost : ℝ) 
  (job_duration : ℝ) 
  (h1 : total_bill = 450)
  (h2 : parts_cost = 225)
  (h3 : job_duration = 5) :
  (total_bill - parts_cost) / job_duration = 45 := by
sorry

end NUMINAMATH_CALUDE_mechanics_billing_problem_l1355_135530


namespace NUMINAMATH_CALUDE_james_streaming_income_l1355_135547

/-- James' streaming income calculation --/
theorem james_streaming_income 
  (initial_subscribers : ℕ) 
  (gifted_subscribers : ℕ) 
  (income_per_subscriber : ℕ) : ℕ :=
  by
  have total_subscribers : ℕ := initial_subscribers + gifted_subscribers
  have monthly_income : ℕ := total_subscribers * income_per_subscriber
  exact monthly_income

#check james_streaming_income 150 50 9

end NUMINAMATH_CALUDE_james_streaming_income_l1355_135547
