import Mathlib

namespace NUMINAMATH_CALUDE_susan_spending_l2748_274893

def carnival_spending (initial_amount food_cost : ℕ) : ℕ :=
  let ride_cost := 2 * food_cost
  let total_spent := food_cost + ride_cost
  initial_amount - total_spent

theorem susan_spending :
  carnival_spending 50 12 = 14 := by
  sorry

end NUMINAMATH_CALUDE_susan_spending_l2748_274893


namespace NUMINAMATH_CALUDE_omega_squared_plus_7omega_plus_40_abs_l2748_274834

def ω : ℂ := 4 + 3 * Complex.I

theorem omega_squared_plus_7omega_plus_40_abs : 
  Complex.abs (ω^2 + 7*ω + 40) = 15 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_omega_squared_plus_7omega_plus_40_abs_l2748_274834


namespace NUMINAMATH_CALUDE_coin_toss_probability_l2748_274808

theorem coin_toss_probability (p_heads : ℚ) (h1 : p_heads = 1/4) :
  1 - p_heads = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l2748_274808


namespace NUMINAMATH_CALUDE_exponent_sum_zero_polynomial_simplification_l2748_274800

-- Problem 1
theorem exponent_sum_zero (m : ℝ) : m^3 * m^6 + (-m^3)^3 = 0 := by sorry

-- Problem 2
theorem polynomial_simplification (a : ℝ) : a*(a-2) - 2*a*(1-3*a) = 7*a^2 - 4*a := by sorry

end NUMINAMATH_CALUDE_exponent_sum_zero_polynomial_simplification_l2748_274800


namespace NUMINAMATH_CALUDE_x_minus_y_equals_three_l2748_274859

theorem x_minus_y_equals_three (x y : ℝ) 
  (eq1 : 3 * x - 5 * y = 5)
  (eq2 : x / (x + y) = 5 / 7) :
  x - y = 3 := by sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_three_l2748_274859


namespace NUMINAMATH_CALUDE_octagon_triangulations_n_gon_triangulations_l2748_274837

/-- Number of ways to triangulate a convex n-gon --/
def triangulations (n : ℕ) : ℕ :=
  if n ≤ 2 then 1 else (2 * n - 4).factorial / (n - 1).factorial / (n - 2).factorial

/-- Theorem: The number of ways to triangulate an octagon is 132 --/
theorem octagon_triangulations : triangulations 8 = 132 := by sorry

/-- Theorem: General formula for triangulating an n-gon --/
theorem n_gon_triangulations (n : ℕ) (h : n ≥ 3) : 
  triangulations n = (2 * n - 4).factorial / (n - 1).factorial / (n - 2).factorial := by sorry

end NUMINAMATH_CALUDE_octagon_triangulations_n_gon_triangulations_l2748_274837


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l2748_274809

/-- Given three positive real numbers that form a geometric sequence,
    their sum is 21, and subtracting 9 from the third number results
    in an arithmetic sequence, prove that the numbers are either
    (1, 4, 16) or (16, 4, 1). -/
theorem geometric_arithmetic_sequence (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive numbers
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- geometric sequence
  a + b + c = 21 →  -- sum is 21
  ∃ d : ℝ, b - a = d ∧ (c - 9) - b = d →  -- arithmetic sequence after subtracting 9
  ((a = 1 ∧ b = 4 ∧ c = 16) ∨ (a = 16 ∧ b = 4 ∧ c = 1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l2748_274809


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2748_274824

/-- Given a hyperbola with equation x²/25 - y²/16 = 1, prove that the positive value m
    such that y = ±mx represents the asymptotes is 4/5 -/
theorem hyperbola_asymptote_slope (x y : ℝ) :
  x^2 / 25 - y^2 / 16 = 1 →
  ∃ (m : ℝ), m > 0 ∧ (y = m * x ∨ y = -m * x) ∧ m = 4/5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2748_274824


namespace NUMINAMATH_CALUDE_walking_speed_calculation_l2748_274882

/-- Proves that the walking speed is 4 km/hr given the conditions of the problem -/
theorem walking_speed_calculation (run_speed : ℝ) (total_distance : ℝ) (total_time : ℝ)
  (h1 : run_speed = 8)
  (h2 : total_distance = 20)
  (h3 : total_time = 3.75) :
  ∃ (walk_speed : ℝ),
    walk_speed = 4 ∧
    (total_distance / 2) / walk_speed + (total_distance / 2) / run_speed = total_time :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_calculation_l2748_274882


namespace NUMINAMATH_CALUDE_min_value_problem_equality_condition_l2748_274810

theorem min_value_problem (x : ℝ) (h : x > 0) : 3 * x + 4 / x ≥ 4 * Real.sqrt 3 := by
  sorry

theorem equality_condition : ∃ x : ℝ, x > 0 ∧ 3 * x + 4 / x = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_equality_condition_l2748_274810


namespace NUMINAMATH_CALUDE_dice_product_divisible_by_8_l2748_274897

/-- The number of dice rolled simultaneously -/
def num_dice : ℕ := 8

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The probability that a single die roll is divisible by 2 -/
def prob_divisible_by_2 : ℚ := 1/2

/-- The probability that the product of dice rolls is divisible by 8 -/
def prob_product_divisible_by_8 : ℚ := 247/256

/-- Theorem: The probability that the product of 8 standard 6-sided dice rolls is divisible by 8 is 247/256 -/
theorem dice_product_divisible_by_8 :
  prob_product_divisible_by_8 = 247/256 :=
sorry

end NUMINAMATH_CALUDE_dice_product_divisible_by_8_l2748_274897


namespace NUMINAMATH_CALUDE_min_buses_for_field_trip_l2748_274849

def min_buses (total_students : ℕ) (bus_capacity_1 : ℕ) (bus_capacity_2 : ℕ) : ℕ :=
  let large_buses := total_students / bus_capacity_1
  let remaining_students := total_students % bus_capacity_1
  if remaining_students = 0 then
    large_buses
  else
    large_buses + 1

theorem min_buses_for_field_trip :
  min_buses 530 45 40 = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_buses_for_field_trip_l2748_274849


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2748_274805

/-- A hyperbola with the given properties -/
def hyperbola (x y : ℝ) : Prop :=
  y^2 / 4 - x^2 = 1

theorem hyperbola_properties :
  ∃ (x y : ℝ),
    -- The hyperbola is centered at the origin
    hyperbola 0 0 ∧
    -- One of its asymptotes is x - 2y = 0
    (∃ (t : ℝ), x = 2*t ∧ y = t) ∧
    -- The hyperbola passes through the point P(√(5/2), 3)
    hyperbola (Real.sqrt (5/2)) 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2748_274805


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_m_l2748_274873

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |1 - x|

-- Theorem for the first part of the problem
theorem solution_set_of_inequality (x : ℝ) :
  f x ≤ 3*x + 4 ↔ x ≥ -1/3 :=
sorry

-- Theorem for the second part of the problem
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, f x ≥ (m^2 - 3*m + 3) * |x|) ↔ 1 ≤ m ∧ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_m_l2748_274873


namespace NUMINAMATH_CALUDE_arithmetic_progression_product_perfect_power_l2748_274855

theorem arithmetic_progression_product_perfect_power :
  ∃ (a d : ℕ+) (b : ℕ+),
    (∀ i j : Fin 5, i ≠ j → a + i.val * d ≠ a + j.val * d) ∧
    (∃ (c : ℕ+), (a * (a + d) * (a + 2 * d) * (a + 3 * d) * (a + 4 * d) : ℕ) = c ^ 2008) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_product_perfect_power_l2748_274855


namespace NUMINAMATH_CALUDE_parabola_transformation_l2748_274892

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = -2x^2 + 1 -/
def original_parabola : Parabola := { a := -2, b := 0, c := 1 }

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a, b := -2 * p.a * h, c := p.a * h^2 + p.b * h + p.c }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + v }

/-- The resulting parabola after transformations -/
def transformed_parabola : Parabola :=
  shift_vertical (shift_horizontal original_parabola 3) (-1)

theorem parabola_transformation :
  transformed_parabola = { a := -2, b := 12, c := -18 } :=
sorry

end NUMINAMATH_CALUDE_parabola_transformation_l2748_274892


namespace NUMINAMATH_CALUDE_parentheses_removal_equivalence_l2748_274825

theorem parentheses_removal_equivalence (x : ℝ) : 
  (3 * x + 2) - 2 * (2 * x - 1) = 3 * x + 2 - 4 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_equivalence_l2748_274825


namespace NUMINAMATH_CALUDE_abs_five_minus_e_l2748_274877

-- Define e as a real number approximately equal to 2.718
def e : ℝ := 2.718

-- State the theorem
theorem abs_five_minus_e : |5 - e| = 2.282 := by
  sorry

end NUMINAMATH_CALUDE_abs_five_minus_e_l2748_274877


namespace NUMINAMATH_CALUDE_layla_earnings_correct_l2748_274891

/-- Calculates the babysitting earnings for a given family -/
def family_earnings (base_rate : ℚ) (hours : ℚ) (bonus_threshold : ℚ) (bonus_amount : ℚ) 
  (discount_rate : ℚ) (flat_rate : ℚ) (is_weekend : Bool) (past_midnight : Bool) : ℚ :=
  sorry

/-- Calculates Layla's total babysitting earnings -/
def layla_total_earnings : ℚ :=
  let donaldsons := family_earnings 15 7 5 5 0 0 false false
  let merck := family_earnings 18 6 3 0 0.1 0 false false
  let hille := family_earnings 20 3 0 10 0 0 false true
  let johnson := family_earnings 22 4 4 0 0 80 false false
  let ramos := family_earnings 25 2 0 20 0 0 true false
  donaldsons + merck + hille + johnson + ramos

theorem layla_earnings_correct : layla_total_earnings = 435.2 := by
  sorry

end NUMINAMATH_CALUDE_layla_earnings_correct_l2748_274891


namespace NUMINAMATH_CALUDE_base_of_first_term_l2748_274890

theorem base_of_first_term (x s : ℝ) (h : (x^16) * (25^s) = 5 * (10^16)) : x = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_base_of_first_term_l2748_274890


namespace NUMINAMATH_CALUDE_intercept_sum_l2748_274899

/-- Given a line with equation y - 3 = -3(x - 5), prove that the sum of its x-intercept and y-intercept is 24 -/
theorem intercept_sum (x y : ℝ) : 
  (y - 3 = -3 * (x - 5)) → 
  ∃ (x_int y_int : ℝ), 
    (0 - 3 = -3 * (x_int - 5)) ∧ 
    (y_int - 3 = -3 * (0 - 5)) ∧ 
    (x_int + y_int = 24) := by
  sorry

end NUMINAMATH_CALUDE_intercept_sum_l2748_274899


namespace NUMINAMATH_CALUDE_kim_average_increase_l2748_274851

theorem kim_average_increase : 
  let scores : List ℝ := [85, 89, 90, 92, 95]
  let average4 := (scores.take 4).sum / 4
  let average5 := scores.sum / 5
  average5 - average4 = 1.2 := by
sorry

end NUMINAMATH_CALUDE_kim_average_increase_l2748_274851


namespace NUMINAMATH_CALUDE_number_proportion_l2748_274847

theorem number_proportion (x : ℚ) : 
  (x / 5 = 30 / (10 * 60)) → x = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_number_proportion_l2748_274847


namespace NUMINAMATH_CALUDE_cone_height_equals_six_l2748_274804

/-- Proves that given a cylinder M with base radius 2 and height 6, and a cone N whose base diameter equals its slant height, if their volumes are equal, then the height of cone N is 6. -/
theorem cone_height_equals_six (r : ℝ) (h : ℝ) :
  (2 : ℝ) ^ 2 * 6 = (1 / 3) * r ^ 2 * h ∧ 
  h = r * Real.sqrt 3 →
  h = 6 := by sorry

end NUMINAMATH_CALUDE_cone_height_equals_six_l2748_274804


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2748_274871

theorem solution_set_quadratic_inequality :
  {x : ℝ | 2 * x^2 - x - 3 > 0} = {x : ℝ | x < -1 ∨ x > 3/2} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2748_274871


namespace NUMINAMATH_CALUDE_total_pencils_l2748_274838

/-- Given 11 children, with each child having 2 pencils, the total number of pencils is 22. -/
theorem total_pencils (num_children : Nat) (pencils_per_child : Nat) (total_pencils : Nat) : 
  num_children = 11 → pencils_per_child = 2 → total_pencils = num_children * pencils_per_child →
  total_pencils = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l2748_274838


namespace NUMINAMATH_CALUDE_square_difference_one_l2748_274801

theorem square_difference_one : (726 : ℕ) * 726 - 725 * 727 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_one_l2748_274801


namespace NUMINAMATH_CALUDE_replaced_student_weight_l2748_274880

/-- Given 5 students, if replacing one student with a 72 kg student causes
    the average weight to decrease by 4 kg, then the replaced student's weight was 92 kg. -/
theorem replaced_student_weight
  (n : ℕ) -- number of students
  (old_avg : ℝ) -- original average weight
  (new_avg : ℝ) -- new average weight after replacement
  (new_student_weight : ℝ) -- weight of the new student
  (h1 : n = 5) -- there are 5 students
  (h2 : new_avg = old_avg - 4) -- average weight decreases by 4 kg
  (h3 : new_student_weight = 72) -- new student weighs 72 kg
  : n * old_avg - (n * new_avg + new_student_weight) = 92 := by
  sorry

end NUMINAMATH_CALUDE_replaced_student_weight_l2748_274880


namespace NUMINAMATH_CALUDE_parallel_vectors_acute_angle_l2748_274823

def parallelVectors (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_acute_angle (x : ℝ) 
  (h1 : 0 < x) (h2 : x < π/2) 
  (h3 : parallelVectors (Real.sin x, 1) (1/2, Real.cos x)) : 
  x = π/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_acute_angle_l2748_274823


namespace NUMINAMATH_CALUDE_power_of_five_divides_l2748_274879

/-- Sequence of positive integers defined recursively -/
def x : ℕ → ℕ
  | 0 => 2  -- We use 0-based indexing in Lean
  | n + 1 => 2 * (x n)^3 + x n

/-- The statement to be proved -/
theorem power_of_five_divides (n : ℕ) : 
  ∃ k : ℕ, x n^2 + 1 = 5^(n+1) * k ∧ ¬(5 ∣ k) := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_divides_l2748_274879


namespace NUMINAMATH_CALUDE_unique_valid_set_l2748_274831

/-- The sum of n consecutive integers starting from a -/
def consecutiveSum (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- A predicate that checks if a set of n consecutive integers starting from a sums to 30 -/
def isValidSet (a n : ℕ) : Prop :=
  a ≥ 3 ∧ n ≥ 2 ∧ consecutiveSum a n = 30

/-- The main theorem stating there is exactly one valid set -/
theorem unique_valid_set : ∃! p : ℕ × ℕ, isValidSet p.1 p.2 := by sorry

end NUMINAMATH_CALUDE_unique_valid_set_l2748_274831


namespace NUMINAMATH_CALUDE_square_area_decrease_l2748_274819

theorem square_area_decrease (s : ℝ) (h : s = 12) :
  let new_s := s * (1 - 0.125)
  (s^2 - new_s^2) / s^2 = 0.25 := by sorry

end NUMINAMATH_CALUDE_square_area_decrease_l2748_274819


namespace NUMINAMATH_CALUDE_perpendicular_slope_l2748_274884

/-- The slope of a line perpendicular to 5x - 2y = 10 is -2/5 -/
theorem perpendicular_slope (x y : ℝ) : 
  (5 * x - 2 * y = 10) → 
  (slope_of_perpendicular_line : ℝ) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l2748_274884


namespace NUMINAMATH_CALUDE_min_value_expression_l2748_274857

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sqrt : y^2 = x) :
  (x^2 + y^4) / (x * y^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2748_274857


namespace NUMINAMATH_CALUDE_AR_equals_six_l2748_274812

-- Define the triangle and points
variable (A B C R P Q : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (acute_triangle : AcuteTriangle A B C)
variable (R_on_perpendicular_bisector : OnPerpendicularBisector R A C)
variable (CA_bisects_BAR : AngleBisector C A (B, R))
variable (Q_intersection : OnLine Q A C ∧ OnLine Q B R)
variable (P_on_circumcircle : OnCircumcircle P A R C)
variable (P_on_AB : SegmentND P A B)
variable (AP_length : dist A P = 1)
variable (PB_length : dist P B = 5)
variable (AQ_length : dist A Q = 2)

-- State the theorem
theorem AR_equals_six : dist A R = 6 := by sorry

end NUMINAMATH_CALUDE_AR_equals_six_l2748_274812


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l2748_274869

def geometricSequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem fifteenth_term_of_sequence :
  let a₁ : ℚ := 27
  let r : ℚ := 1/6
  geometricSequence a₁ r 15 = 1/14155776 := by
sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l2748_274869


namespace NUMINAMATH_CALUDE_ninety_sixth_digit_of_5_div_37_l2748_274883

/-- The decimal representation of 5/37 has a repeating pattern of length 3 -/
def decimal_repeat_length : ℕ := 3

/-- The repeating pattern in the decimal representation of 5/37 -/
def decimal_pattern : Fin 3 → ℕ
| 0 => 1
| 1 => 3
| 2 => 5

/-- The 96th digit after the decimal point in the decimal representation of 5/37 is 5 -/
theorem ninety_sixth_digit_of_5_div_37 : 
  decimal_pattern ((96 : ℕ) % decimal_repeat_length) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ninety_sixth_digit_of_5_div_37_l2748_274883


namespace NUMINAMATH_CALUDE_bahs_equal_to_yahs_l2748_274839

/-- The number of bahs equal to 30 rahs -/
def bahs_to_30_rahs : ℕ := 20

/-- The number of rahs equal to 20 yahs -/
def rahs_to_20_yahs : ℕ := 12

/-- The number of yahs we want to convert to bahs -/
def yahs_to_convert : ℕ := 1200

/-- The theorem stating the equivalence between bahs and yahs -/
theorem bahs_equal_to_yahs : ∃ (n : ℕ), n * bahs_to_30_rahs * rahs_to_20_yahs = yahs_to_convert * 30 * 20 :=
sorry

end NUMINAMATH_CALUDE_bahs_equal_to_yahs_l2748_274839


namespace NUMINAMATH_CALUDE_sequence_formula_correct_l2748_274848

def a (n : ℕ) : ℤ := (-1)^n * (4*n - 1)

theorem sequence_formula_correct :
  (a 1 = -3) ∧ (a 2 = 7) ∧ (a 3 = -11) ∧ (a 4 = 15) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formula_correct_l2748_274848


namespace NUMINAMATH_CALUDE_quadratic_equation_has_solution_l2748_274850

theorem quadratic_equation_has_solution (a b : ℝ) :
  ∃ x : ℝ, (a^6 - b^6) * x^2 + 2 * (a^5 - b^5) * x + (a^4 - b^4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_has_solution_l2748_274850


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_ABC_l2748_274881

/-- Triangle ABC with integer side lengths, BD angle bisector of ∠ABC, AD = 4, DC = 6, D on AC -/
structure TriangleABC where
  AB : ℕ
  BC : ℕ
  AC : ℕ
  AD : ℕ
  DC : ℕ
  hAD : AD = 4
  hDC : DC = 6
  hAC : AC = AD + DC
  hAngleBisector : AB * DC = BC * AD

/-- The minimum possible perimeter of triangle ABC is 25 -/
theorem min_perimeter_triangle_ABC (t : TriangleABC) : 
  (∀ t' : TriangleABC, t'.AB + t'.BC + t'.AC ≥ t.AB + t.BC + t.AC) → 
  t.AB + t.BC + t.AC = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_ABC_l2748_274881


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l2748_274867

theorem smallest_four_digit_multiple_of_18 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n → 1008 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l2748_274867


namespace NUMINAMATH_CALUDE_square_field_area_l2748_274854

/-- Proves that a square field with given conditions has an area of 27889 square meters -/
theorem square_field_area (wire_cost_per_meter : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) :
  wire_cost_per_meter = 1.30 →
  total_cost = 865.80 →
  gate_width = 1 →
  num_gates = 2 →
  ∃ (side_length : ℝ),
    (4 * side_length - gate_width * num_gates) * wire_cost_per_meter = total_cost ∧
    side_length^2 = 27889 :=
by sorry

end NUMINAMATH_CALUDE_square_field_area_l2748_274854


namespace NUMINAMATH_CALUDE_max_product_on_circle_l2748_274807

/-- The maximum product of xy for integer points on x^2 + y^2 = 100 is 48 -/
theorem max_product_on_circle : 
  (∃ (a b : ℤ), a^2 + b^2 = 100 ∧ a * b = 48) ∧ 
  (∀ (x y : ℤ), x^2 + y^2 = 100 → x * y ≤ 48) := by
  sorry

#check max_product_on_circle

end NUMINAMATH_CALUDE_max_product_on_circle_l2748_274807


namespace NUMINAMATH_CALUDE_angle_C_measure_l2748_274836

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (sum_of_angles : A + B + C = 180)
  (all_positive : 0 < A ∧ 0 < B ∧ 0 < C)

-- Theorem statement
theorem angle_C_measure (abc : Triangle) 
  (h1 : abc.A = 60) 
  (h2 : abc.C = 2 * abc.B) : 
  abc.C = 80 := by
sorry

end NUMINAMATH_CALUDE_angle_C_measure_l2748_274836


namespace NUMINAMATH_CALUDE_perpendicular_lines_intersection_l2748_274844

/-- Given two perpendicular lines and the foot of their perpendicular, prove m - n + p = 20 -/
theorem perpendicular_lines_intersection (m n p : ℝ) : 
  (∀ x y, m * x + 4 * y - 2 = 0 ∨ 2 * x - 5 * y + n = 0) →  -- Two lines
  (m * 2 = -4 * 5) →  -- Perpendicularity condition
  (m * 1 + 4 * p - 2 = 0) →  -- Foot satisfies first line equation
  (2 * 1 - 5 * p + n = 0) →  -- Foot satisfies second line equation
  m - n + p = 20 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_intersection_l2748_274844


namespace NUMINAMATH_CALUDE_inequality_preservation_l2748_274815

theorem inequality_preservation (x y : ℝ) (h : x > y) : 2 * x + 1 > 2 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l2748_274815


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l2748_274852

/-- The perimeter of a semicircle with radius 8 is 16 + 8π -/
theorem semicircle_perimeter : 
  ∀ (r : ℝ), r = 8 → 2 * r + (π * r) = 16 + 8 * π :=
by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l2748_274852


namespace NUMINAMATH_CALUDE_area_of_rectangle_PQRS_l2748_274813

-- Define the Point type
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define the Rectangle type
structure Rectangle :=
  (p : Point) (q : Point) (r : Point) (s : Point)

-- Define the area function for a rectangle
def rectangleArea (rect : Rectangle) : ℝ :=
  let width := abs (rect.q.x - rect.p.x)
  let height := abs (rect.p.y - rect.s.y)
  width * height

-- Theorem statement
theorem area_of_rectangle_PQRS :
  let p := Point.mk (-4) 2
  let q := Point.mk 4 2
  let r := Point.mk 4 (-2)
  let s := Point.mk (-4) (-2)
  let rect := Rectangle.mk p q r s
  rectangleArea rect = 32 := by
  sorry

end NUMINAMATH_CALUDE_area_of_rectangle_PQRS_l2748_274813


namespace NUMINAMATH_CALUDE_total_arrangements_eq_65_l2748_274878

-- Define the number of seats
def num_seats : ℕ := 10

-- Define the number of women
def num_women : ℕ := 6

-- Define the number of men
def num_men : ℕ := 1

-- Define the maximum number of seats a woman can move
def max_women_move : ℕ := 1

-- Define the maximum number of seats a man can move
def max_men_move : ℕ := 2

-- Define a function to calculate the number of reseating arrangements for women
def women_arrangements (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | n + 2 => women_arrangements (n + 1) + women_arrangements n

-- Define a function to calculate the number of reseating arrangements for the man
def man_arrangements (original_pos : ℕ) : ℕ :=
  2 * max_men_move + 1

-- Theorem: The total number of reseating arrangements is 65
theorem total_arrangements_eq_65 :
  women_arrangements num_women * man_arrangements (num_women + 1) = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_arrangements_eq_65_l2748_274878


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2748_274846

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (2 + i) / (1 + i)
  Complex.im z = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2748_274846


namespace NUMINAMATH_CALUDE_copy_pages_theorem_l2748_274845

/-- Given a cost per page in cents and a budget in dollars, 
    calculates the maximum number of full pages that can be copied. -/
def max_pages_copied (cost_per_page : ℕ) (budget_dollars : ℕ) : ℕ :=
  (budget_dollars * 100) / cost_per_page

/-- Theorem stating that with a cost of 3 cents per page and a budget of $25, 
    the maximum number of full pages that can be copied is 833. -/
theorem copy_pages_theorem : max_pages_copied 3 25 = 833 := by
  sorry

#eval max_pages_copied 3 25

end NUMINAMATH_CALUDE_copy_pages_theorem_l2748_274845


namespace NUMINAMATH_CALUDE_cricket_average_score_l2748_274802

theorem cricket_average_score 
  (total_matches : ℕ) 
  (overall_average : ℝ) 
  (first_six_average : ℝ) 
  (last_four_sum_lower : ℝ) 
  (last_four_sum_upper : ℝ) 
  (last_four_lowest : ℝ) 
  (h1 : total_matches = 10)
  (h2 : overall_average = 38.9)
  (h3 : first_six_average = 41)
  (h4 : last_four_sum_lower = 120)
  (h5 : last_four_sum_upper = 200)
  (h6 : last_four_lowest = 20)
  (h7 : last_four_sum_lower ≤ (overall_average * total_matches - first_six_average * 6))
  (h8 : (overall_average * total_matches - first_six_average * 6) ≤ last_four_sum_upper) :
  (overall_average * total_matches - first_six_average * 6) / 4 = 35.75 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_score_l2748_274802


namespace NUMINAMATH_CALUDE_quadrilaterals_from_nine_points_l2748_274853

theorem quadrilaterals_from_nine_points : ∀ n : ℕ, n = 9 →
  (Nat.choose n 4 : ℕ) = 126 := by sorry

end NUMINAMATH_CALUDE_quadrilaterals_from_nine_points_l2748_274853


namespace NUMINAMATH_CALUDE_stamp_revenue_calculation_l2748_274870

/-- The total revenue generated from stamp sales --/
theorem stamp_revenue_calculation : 
  let color_price : ℚ := 15/100
  let bw_price : ℚ := 10/100
  let color_sold : ℕ := 578833
  let bw_sold : ℕ := 523776
  let total_revenue := (color_price * color_sold) + (bw_price * bw_sold)
  total_revenue = 139202551/10000 := by
  sorry

end NUMINAMATH_CALUDE_stamp_revenue_calculation_l2748_274870


namespace NUMINAMATH_CALUDE_vector_triangle_rule_l2748_274816

-- Define a triangle ABC in a vector space
variable {V : Type*} [AddCommGroup V]
variable (A B C : V)

-- State the theorem
theorem vector_triangle_rule :
  (C - A) - (B - A) + (B - C) = (0 : V) := by
  sorry

end NUMINAMATH_CALUDE_vector_triangle_rule_l2748_274816


namespace NUMINAMATH_CALUDE_rational_terms_not_adjacent_probability_l2748_274821

theorem rational_terms_not_adjacent_probability (n : ℕ) (rational_terms : ℕ) :
  n = 9 ∧ rational_terms = 3 →
  (Nat.factorial 6 * (Nat.factorial 7 / (Nat.factorial 4 * Nat.factorial 3))) / Nat.factorial 9 = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_rational_terms_not_adjacent_probability_l2748_274821


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2748_274863

theorem arithmetic_calculation : 4 * (8 - 3) + 6 / 2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2748_274863


namespace NUMINAMATH_CALUDE_cos_difference_value_l2748_274866

theorem cos_difference_value (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by
sorry

end NUMINAMATH_CALUDE_cos_difference_value_l2748_274866


namespace NUMINAMATH_CALUDE_interior_angles_sum_l2748_274896

/-- Given a convex polygon where the sum of interior angles is 3600 degrees,
    prove that the sum of interior angles of a polygon with 3 more sides is 4140 degrees. -/
theorem interior_angles_sum (n : ℕ) : 
  (180 * (n - 2) = 3600) → (180 * ((n + 3) - 2) = 4140) := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_sum_l2748_274896


namespace NUMINAMATH_CALUDE_probability_vowel_in_mathematics_l2748_274818

def english_alphabet : Finset Char := sorry

def vowels : Finset Char := {'a', 'e', 'i', 'o', 'u'}

def mathematics : List Char := ['m', 'a', 't', 'h', 'e', 'm', 'a', 't', 'i', 'c', 's']

theorem probability_vowel_in_mathematics :
  (Finset.filter (fun c => c ∈ vowels) mathematics.toFinset).card / mathematics.length = 4 / 11 :=
by sorry

end NUMINAMATH_CALUDE_probability_vowel_in_mathematics_l2748_274818


namespace NUMINAMATH_CALUDE_no_function_exists_for_part_a_function_exists_for_part_b_l2748_274872

-- Part a
theorem no_function_exists_for_part_a :
  ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 1 :=
sorry

-- Part b
theorem function_exists_for_part_b :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = 2 * n :=
sorry

end NUMINAMATH_CALUDE_no_function_exists_for_part_a_function_exists_for_part_b_l2748_274872


namespace NUMINAMATH_CALUDE_expression_greater_than_30_l2748_274862

theorem expression_greater_than_30 :
  ∃ (expr : ℝ),
    (expr = 20 / (2 - Real.sqrt 2)) ∧
    (expr > 30) := by
  sorry

end NUMINAMATH_CALUDE_expression_greater_than_30_l2748_274862


namespace NUMINAMATH_CALUDE_subsets_with_three_adjacent_12_chairs_l2748_274832

/-- The number of chairs arranged in a circle -/
def n : ℕ := 12

/-- A function that calculates the number of subsets of n chairs arranged in a circle
    that contain at least three adjacent chairs -/
def subsets_with_three_adjacent (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of subsets of 12 chairs arranged in a circle
    that contain at least three adjacent chairs is 2056 -/
theorem subsets_with_three_adjacent_12_chairs :
  subsets_with_three_adjacent n = 2056 := by sorry

end NUMINAMATH_CALUDE_subsets_with_three_adjacent_12_chairs_l2748_274832


namespace NUMINAMATH_CALUDE_rational_inequality_solution_l2748_274843

theorem rational_inequality_solution (x : ℝ) :
  (x - 3) / (x^2 + 4*x + 13) ≤ 0 ↔ x ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_l2748_274843


namespace NUMINAMATH_CALUDE_rectangle_area_unchanged_l2748_274856

theorem rectangle_area_unchanged (x y : ℝ) :
  x > 0 ∧ y > 0 ∧
  x * y = (x + 3.5) * (y - 1.33) ∧
  x * y = (x - 3.5) * (y + 1.67) →
  x * y = 35 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_unchanged_l2748_274856


namespace NUMINAMATH_CALUDE_binary_ternary_equality_l2748_274875

/-- Represents a digit in base 2 (binary) -/
def BinaryDigit : Type := {n : ℕ // n < 2}

/-- Represents a digit in base 3 (ternary) -/
def TernaryDigit : Type := {n : ℕ // n < 3}

/-- Converts a binary number to decimal -/
def binaryToDecimal (d₂ : BinaryDigit) (d₁ : BinaryDigit) (d₀ : BinaryDigit) : ℕ :=
  d₂.val * 2^2 + d₁.val * 2^1 + d₀.val * 2^0

/-- Converts a ternary number to decimal -/
def ternaryToDecimal (d₂ : TernaryDigit) (d₁ : TernaryDigit) (d₀ : TernaryDigit) : ℕ :=
  d₂.val * 3^2 + d₁.val * 3^1 + d₀.val * 3^0

theorem binary_ternary_equality :
  ∀ (x : TernaryDigit) (y : BinaryDigit),
    binaryToDecimal ⟨1, by norm_num⟩ y ⟨1, by norm_num⟩ = ternaryToDecimal x ⟨0, by norm_num⟩ ⟨2, by norm_num⟩ →
    x.val = 1 ∧ y.val = 1 ∧ ternaryToDecimal x ⟨0, by norm_num⟩ ⟨2, by norm_num⟩ = 11 :=
by sorry

#check binary_ternary_equality

end NUMINAMATH_CALUDE_binary_ternary_equality_l2748_274875


namespace NUMINAMATH_CALUDE_distance_to_canada_is_360_l2748_274817

/-- Calculates the distance traveled given speed, total time, and stop time. -/
def distance_to_canada (speed : ℝ) (total_time : ℝ) (stop_time : ℝ) : ℝ :=
  speed * (total_time - stop_time)

/-- Proves that the distance to Canada is 360 miles under the given conditions. -/
theorem distance_to_canada_is_360 :
  distance_to_canada 60 7 1 = 360 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_canada_is_360_l2748_274817


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuses_l2748_274885

theorem right_triangle_hypotenuses : 
  let legs : List (ℕ × ℕ) := [(3, 4), (12, 5), (15, 8), (7, 24), (12, 35), (15, 36)]
  let hypotenuses : List ℕ := [5, 13, 17, 25, 37, 39]
  ∀ (i : Fin 6), 
    (legs.get i).1^2 + (legs.get i).2^2 = (hypotenuses.get i)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_right_triangle_hypotenuses_l2748_274885


namespace NUMINAMATH_CALUDE_a_b_parallel_opposite_l2748_274895

/-- Two vectors in ℝ² -/
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, -4)

/-- Predicate to check if two vectors are parallel and in opposite directions -/
def parallel_opposite (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ w = (k * v.1, k * v.2)

/-- Theorem stating that vectors a and b are parallel and in opposite directions -/
theorem a_b_parallel_opposite : parallel_opposite a b := by
  sorry

end NUMINAMATH_CALUDE_a_b_parallel_opposite_l2748_274895


namespace NUMINAMATH_CALUDE_fraction_squared_equality_l2748_274861

theorem fraction_squared_equality : ((-123456789 : ℤ) / 246913578)^2 = (15241578750190521 : ℚ) / 60995928316126584 := by
  sorry

end NUMINAMATH_CALUDE_fraction_squared_equality_l2748_274861


namespace NUMINAMATH_CALUDE_ending_number_proof_l2748_274811

theorem ending_number_proof (n : ℕ) : 
  (∃ (evens : Finset ℕ), evens.card = 35 ∧ 
    (∀ x ∈ evens, 25 < x ∧ x ≤ n ∧ Even x) ∧
    (∀ y, 25 < y ∧ y ≤ n ∧ Even y → y ∈ evens)) ↔ 
  n = 94 :=
sorry

end NUMINAMATH_CALUDE_ending_number_proof_l2748_274811


namespace NUMINAMATH_CALUDE_obtuse_triangle_side_length_range_l2748_274806

theorem obtuse_triangle_side_length_range (a : ℝ) :
  (∃ (x y z : ℝ), x = a ∧ y = a + 3 ∧ z = a + 6 ∧
   x + y > z ∧ y + z > x ∧ z + x > y ∧  -- triangle inequality
   z^2 > x^2 + y^2)  -- obtuse triangle condition
  ↔ 3 < a ∧ a < 9 := by
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_side_length_range_l2748_274806


namespace NUMINAMATH_CALUDE_smallest_value_of_expression_l2748_274886

theorem smallest_value_of_expression (x : ℝ) (h : x ≠ -7) :
  (2 * x^2 + 98) / (x + 7)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_of_expression_l2748_274886


namespace NUMINAMATH_CALUDE_box_sum_remainder_l2748_274840

theorem box_sum_remainder (n : ℕ) (a : Fin (2 * n) → ℕ) :
  ∃ (i j : Fin (2 * n)), i ≠ j ∧ (a i + i.val) % (2 * n) = (a j + j.val) % (2 * n) := by
  sorry

end NUMINAMATH_CALUDE_box_sum_remainder_l2748_274840


namespace NUMINAMATH_CALUDE_winner_determined_by_parity_l2748_274830

/-- Represents a player in the game -/
inductive Player
  | Anthelme
  | Brunehaut

/-- Represents the game state on an m × n chessboard -/
structure GameState (m n : ℕ) where
  kingPosition : ℕ × ℕ
  visitedSquares : Set (ℕ × ℕ)

/-- Determines the winner of the game based on the board dimensions -/
def determineWinner (m n : ℕ) : Player :=
  if m * n % 2 = 0 then Player.Anthelme else Player.Brunehaut

/-- Theorem stating that the winner is determined by the parity of m × n -/
theorem winner_determined_by_parity (m n : ℕ) :
  determineWinner m n = 
    if m * n % 2 = 0 then Player.Anthelme else Player.Brunehaut :=
by sorry

end NUMINAMATH_CALUDE_winner_determined_by_parity_l2748_274830


namespace NUMINAMATH_CALUDE_pattern_cost_is_15_l2748_274864

/-- The cost of a sewing pattern given the total spent, fabric cost per yard, yards of fabric bought,
    thread cost per spool, and number of thread spools bought. -/
def pattern_cost (total_spent fabric_cost_per_yard yards_fabric thread_cost_per_spool num_thread_spools : ℕ) : ℕ :=
  total_spent - (fabric_cost_per_yard * yards_fabric + thread_cost_per_spool * num_thread_spools)

/-- Theorem stating that the pattern cost is $15 given the specific conditions. -/
theorem pattern_cost_is_15 :
  pattern_cost 141 24 5 3 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_pattern_cost_is_15_l2748_274864


namespace NUMINAMATH_CALUDE_pink_yards_calculation_l2748_274835

/-- The number of yards of silk dyed green -/
def green_yards : ℕ := 61921

/-- The total number of yards of silk dyed for the order -/
def total_yards : ℕ := 111421

/-- The number of yards of silk dyed pink -/
def pink_yards : ℕ := total_yards - green_yards

theorem pink_yards_calculation : pink_yards = 49500 := by
  sorry

end NUMINAMATH_CALUDE_pink_yards_calculation_l2748_274835


namespace NUMINAMATH_CALUDE_expression_evaluation_l2748_274822

theorem expression_evaluation (x y z : ℝ) :
  3 * (x - (2 * y - 3 * z)) - 2 * ((3 * x - 2 * y) - 4 * z) = -3 * x - 2 * y + 17 * z :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2748_274822


namespace NUMINAMATH_CALUDE_correct_observation_value_l2748_274828

theorem correct_observation_value (n : ℕ) (initial_mean corrected_mean wrong_value : ℝ) 
  (h1 : n = 50)
  (h2 : initial_mean = 40)
  (h3 : corrected_mean = 40.66)
  (h4 : wrong_value = 15) :
  let total_sum := n * initial_mean
  let corrected_sum := n * corrected_mean
  let difference := corrected_sum - total_sum
  let actual_value := wrong_value + difference
  actual_value = 48 := by sorry

end NUMINAMATH_CALUDE_correct_observation_value_l2748_274828


namespace NUMINAMATH_CALUDE_paperclip_capacity_l2748_274888

/-- Given a box of volume 16 cm³ that holds 50 paperclips, 
    prove that a box of volume 48 cm³ will hold 150 paperclips, 
    assuming a direct proportion between volume and paperclip capacity. -/
theorem paperclip_capacity (v₁ v₂ : ℝ) (c₁ c₂ : ℕ) :
  v₁ = 16 → v₂ = 48 → c₁ = 50 →
  (v₁ * c₂ = v₂ * c₁) →
  c₂ = 150 := by
  sorry

#check paperclip_capacity

end NUMINAMATH_CALUDE_paperclip_capacity_l2748_274888


namespace NUMINAMATH_CALUDE_intersection_correct_l2748_274820

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℚ
  y : ℚ

/-- Represents a parametric line in 2D -/
structure ParametricLine where
  origin : Vector2D
  direction : Vector2D

/-- The first line -/
def line1 : ParametricLine :=
  { origin := { x := 2, y := 2 },
    direction := { x := 3, y := -4 } }

/-- The second line -/
def line2 : ParametricLine :=
  { origin := { x := 4, y := -6 },
    direction := { x := 5, y := 3 } }

/-- Calculates the point on a parametric line given a parameter value -/
def pointOnLine (line : ParametricLine) (t : ℚ) : Vector2D :=
  { x := line.origin.x + t * line.direction.x,
    y := line.origin.y + t * line.direction.y }

/-- The intersection point of the two lines -/
def intersectionPoint : Vector2D :=
  { x := 160 / 29, y := -160 / 29 }

/-- Theorem stating that the calculated intersection point is correct -/
theorem intersection_correct :
  ∃ (t u : ℚ), pointOnLine line1 t = intersectionPoint ∧ pointOnLine line2 u = intersectionPoint :=
by
  sorry


end NUMINAMATH_CALUDE_intersection_correct_l2748_274820


namespace NUMINAMATH_CALUDE_multiplication_division_equality_l2748_274803

theorem multiplication_division_equality : (3 * 4) / 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_division_equality_l2748_274803


namespace NUMINAMATH_CALUDE_elise_remaining_money_l2748_274842

/-- Calculates the remaining money for Elise given her initial amount, savings, and expenditures. -/
def remaining_money (initial : ℕ) (savings : ℕ) (comic_cost : ℕ) (puzzle_cost : ℕ) : ℕ :=
  initial + savings - comic_cost - puzzle_cost

/-- Proves that Elise is left with $1 given her initial amount, savings, and expenditures. -/
theorem elise_remaining_money :
  remaining_money 8 13 2 18 = 1 := by
  sorry

end NUMINAMATH_CALUDE_elise_remaining_money_l2748_274842


namespace NUMINAMATH_CALUDE_greatest_product_three_digit_l2748_274826

def Digits : Finset Nat := {3, 5, 7, 8, 9}

def is_valid_pair (a b c d e : Nat) : Prop :=
  a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧ e ∈ Digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def three_digit (a b c : Nat) : Nat := 100 * a + 10 * b + c
def two_digit (d e : Nat) : Nat := 10 * d + e

def one_odd_one_even (x y : Nat) : Prop :=
  (x % 2 = 1 ∧ y % 2 = 0) ∨ (x % 2 = 0 ∧ y % 2 = 1)

theorem greatest_product_three_digit :
  ∀ a b c d e,
    is_valid_pair a b c d e →
    one_odd_one_even (three_digit a b c) (two_digit d e) →
    three_digit a b c * two_digit d e ≤ 972 * 85 :=
  sorry

end NUMINAMATH_CALUDE_greatest_product_three_digit_l2748_274826


namespace NUMINAMATH_CALUDE_correct_answer_l2748_274894

def add_and_round_to_nearest_ten (a b : ℕ) : ℕ :=
  let sum := a + b
  let ones_digit := sum % 10
  if ones_digit < 5 then
    sum - ones_digit
  else
    sum + (10 - ones_digit)

theorem correct_answer : add_and_round_to_nearest_ten 46 37 = 80 := by
  sorry

end NUMINAMATH_CALUDE_correct_answer_l2748_274894


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2748_274887

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    x1^2 + 2*m*x1 + m^2 + m = 0 ∧ 
    x2^2 + 2*m*x2 + m^2 + m = 0 ∧ 
    x1^2 + x2^2 = 12) → 
  m = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2748_274887


namespace NUMINAMATH_CALUDE_puzzle_completion_l2748_274876

/-- Puzzle completion problem -/
theorem puzzle_completion 
  (total_pieces : ℕ) 
  (num_children : ℕ) 
  (time_limit : ℕ) 
  (reyn_rate : ℚ) :
  total_pieces = 500 →
  num_children = 4 →
  time_limit = 120 →
  reyn_rate = 25 / 30 →
  (reyn_rate * time_limit + 
   2 * reyn_rate * time_limit + 
   3 * reyn_rate * time_limit + 
   4 * reyn_rate * time_limit) ≥ total_pieces := by
  sorry


end NUMINAMATH_CALUDE_puzzle_completion_l2748_274876


namespace NUMINAMATH_CALUDE_solve_for_y_l2748_274860

theorem solve_for_y (x y : ℤ) (h1 : x = 4) (h2 : x + y = 0) : y = -4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2748_274860


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2748_274868

theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) :
  a = (2, 1) →
  b = (-1, k) →
  (∃ (t : ℝ), t ≠ 0 ∧ a.1 * t = b.1 ∧ a.2 * t = b.2) →
  k = -1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2748_274868


namespace NUMINAMATH_CALUDE_no_roots_in_interval_l2748_274833

-- Define the function f
def f (x : ℝ) : ℝ := x^4 - 4*x^3 + 10*x^2

-- State the theorem
theorem no_roots_in_interval :
  ∀ x ∈ Set.Icc 1 2, f x ≠ 0 := by sorry

end NUMINAMATH_CALUDE_no_roots_in_interval_l2748_274833


namespace NUMINAMATH_CALUDE_sixty_first_term_is_201_l2748_274898

/-- An arithmetic sequence with a_5 = 33 and common difference d = 3 -/
def arithmetic_sequence (n : ℕ) : ℝ :=
  33 + 3 * (n - 5)

/-- Theorem: The 61st term of the sequence is 201 -/
theorem sixty_first_term_is_201 : arithmetic_sequence 61 = 201 := by
  sorry

end NUMINAMATH_CALUDE_sixty_first_term_is_201_l2748_274898


namespace NUMINAMATH_CALUDE_line_length_difference_l2748_274858

-- Define the lengths of the lines in inches
def white_line_inches : ℝ := 7.666666666666667
def blue_line_inches : ℝ := 3.3333333333333335

-- Define conversion rates
def inches_to_cm : ℝ := 2.54
def cm_to_mm : ℝ := 10

-- Theorem statement
theorem line_length_difference : 
  (white_line_inches * inches_to_cm - blue_line_inches * inches_to_cm) * cm_to_mm = 110.05555555555553 := by
  sorry

end NUMINAMATH_CALUDE_line_length_difference_l2748_274858


namespace NUMINAMATH_CALUDE_rectangular_solid_depth_l2748_274874

theorem rectangular_solid_depth
  (length width surface_area : ℝ)
  (h_length : length = 9)
  (h_width : width = 8)
  (h_surface_area : surface_area = 314)
  (h_formula : surface_area = 2 * length * width + 2 * length * depth + 2 * width * depth) :
  depth = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_depth_l2748_274874


namespace NUMINAMATH_CALUDE_dress_cost_difference_l2748_274889

theorem dress_cost_difference (patty ida jean pauline : ℕ) : 
  patty = ida + 10 →
  ida = jean + 30 →
  jean < pauline →
  pauline = 30 →
  patty + ida + jean + pauline = 160 →
  pauline - jean = 10 := by
sorry

end NUMINAMATH_CALUDE_dress_cost_difference_l2748_274889


namespace NUMINAMATH_CALUDE_equation_solutions_l2748_274841

theorem equation_solutions :
  (∀ x : ℝ, (x - 1)^2 - 9 = 0 ↔ x = 4 ∨ x = -2) ∧
  (∀ x : ℝ, 2*x*(x - 3) + (x - 3) = 0 ↔ x = 3 ∨ x = -1/2) ∧
  (∀ x : ℝ, 2*x^2 - x - 1 = 0 ↔ x = 1 ∨ x = -1/2) ∧
  (∀ x : ℝ, x^2 - 6*x - 16 = 0 ↔ x = 8 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2748_274841


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2748_274827

/-- Definition of a quadratic equation in one variable x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l2748_274827


namespace NUMINAMATH_CALUDE_not_power_of_two_l2748_274829

theorem not_power_of_two (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ¬ ∃ k : ℕ, (36 * a + b) * (a + 36 * b) = 2^k :=
by sorry

end NUMINAMATH_CALUDE_not_power_of_two_l2748_274829


namespace NUMINAMATH_CALUDE_product_65_35_l2748_274865

theorem product_65_35 : 65 * 35 = 2275 := by
  sorry

end NUMINAMATH_CALUDE_product_65_35_l2748_274865


namespace NUMINAMATH_CALUDE_boxes_in_carton_l2748_274814

/-- The number of boxes in a carton -/
def boxes_per_carton : ℕ := sorry

/-- The number of packs of cheese cookies in each box -/
def packs_per_box : ℕ := 10

/-- The price of a pack of cheese cookies in dollars -/
def price_per_pack : ℕ := 1

/-- The cost of a dozen cartons in dollars -/
def cost_dozen_cartons : ℕ := 1440

/-- Theorem stating the number of boxes in a carton -/
theorem boxes_in_carton : boxes_per_carton = 12 := by sorry

end NUMINAMATH_CALUDE_boxes_in_carton_l2748_274814
