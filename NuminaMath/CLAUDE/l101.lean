import Mathlib

namespace NUMINAMATH_CALUDE_triangle_problem_l101_10112

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  A + B + C = π ∧
  Real.cos (B - C) - 2 * Real.sin B * Real.sin C = -1/2 →
  A = π/3 ∧
  (a = 5 ∧ b = 4 →
    a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
    (1/2) * b * c * Real.sin A = 2*Real.sqrt 3 + Real.sqrt 39) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l101_10112


namespace NUMINAMATH_CALUDE_neighbor_rolls_l101_10124

def total_rolls : ℕ := 12
def grandmother_rolls : ℕ := 3
def uncle_rolls : ℕ := 4
def rolls_left : ℕ := 2

theorem neighbor_rolls : 
  total_rolls - grandmother_rolls - uncle_rolls - rolls_left = 3 := by
  sorry

end NUMINAMATH_CALUDE_neighbor_rolls_l101_10124


namespace NUMINAMATH_CALUDE_statement_1_statement_2_false_statement_3_statement_4_l101_10181

-- Statement ①
theorem statement_1 (a b : ℝ) (f : ℝ → ℝ) (h : ∀ x ∈ Set.Icc (2*a - 1) (a + 4), f x = a*x^2 + (2*a + b)*x + 2) :
  (∀ x ∈ Set.Icc (2*a - 1) (a + 4), f x = f (-x)) → b = 2 := by sorry

-- Statement ②
theorem statement_2_false : ∃ f : ℝ → ℝ, 
  (∀ x, f x = min (-2*x + 2) (-2*x^2 + 4*x + 2)) ∧ 
  (∃ x, f x > 1) := by sorry

-- Statement ③
theorem statement_3 (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |2*x + a|) :
  (∀ x y, x ≥ 3 ∧ y > x → f x < f y) → a = -6 := by sorry

-- Statement ④
theorem statement_4 (f : ℝ → ℝ) 
  (h1 : ∃ x, f x ≠ 0) 
  (h2 : ∀ x y, f (x * y) = x * f y + y * f x) :
  ∀ x, f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_statement_1_statement_2_false_statement_3_statement_4_l101_10181


namespace NUMINAMATH_CALUDE_exponent_addition_l101_10125

theorem exponent_addition (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end NUMINAMATH_CALUDE_exponent_addition_l101_10125


namespace NUMINAMATH_CALUDE_distance_when_parallel_max_distance_l101_10118

/-- A parabola with vertex at the origin -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1^2}

/-- Two points on the parabola -/
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

/-- The origin -/
def O : ℝ × ℝ := (0, 0)

/-- Assumption that P and Q are on the parabola -/
axiom h_P_on_parabola : P ∈ Parabola
axiom h_Q_on_parabola : Q ∈ Parabola

/-- Assumption that OP is perpendicular to OQ -/
axiom h_perpendicular : (P.1 * Q.1 + P.2 * Q.2) = 0

/-- Distance from a point to a line -/
def distanceToLine (point : ℝ × ℝ) (line : Set (ℝ × ℝ)) : ℝ := sorry

/-- The line PQ -/
def LinePQ : Set (ℝ × ℝ) := sorry

/-- Statement: When PQ is parallel to x-axis, distance from O to PQ is 1 -/
theorem distance_when_parallel : 
  P.2 = Q.2 → distanceToLine O LinePQ = 1 := sorry

/-- Statement: The maximum distance from O to PQ is 1 -/
theorem max_distance : 
  ∀ P Q : ℝ × ℝ, P ∈ Parabola → Q ∈ Parabola → 
  (P.1 * Q.1 + P.2 * Q.2) = 0 → 
  distanceToLine O LinePQ ≤ 1 := sorry

end NUMINAMATH_CALUDE_distance_when_parallel_max_distance_l101_10118


namespace NUMINAMATH_CALUDE_supernatural_gathering_handshakes_l101_10133

/-- The number of gremlins at the Supernatural Gathering -/
def num_gremlins : ℕ := 25

/-- The number of imps at the Supernatural Gathering -/
def num_imps : ℕ := 20

/-- The number of imps shaking hands amongst themselves -/
def num_imps_shaking : ℕ := num_imps / 2

/-- Calculate the number of handshakes between two groups -/
def handshakes_between (group1 : ℕ) (group2 : ℕ) : ℕ := group1 * group2

/-- Calculate the number of handshakes within a group -/
def handshakes_within (group : ℕ) : ℕ := group * (group - 1) / 2

/-- The total number of handshakes at the Supernatural Gathering -/
def total_handshakes : ℕ :=
  handshakes_within num_gremlins +
  handshakes_within num_imps_shaking +
  handshakes_between num_gremlins num_imps

theorem supernatural_gathering_handshakes :
  total_handshakes = 845 := by sorry

end NUMINAMATH_CALUDE_supernatural_gathering_handshakes_l101_10133


namespace NUMINAMATH_CALUDE_number_ratio_l101_10180

theorem number_ratio (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : y = 3 * x) (h4 : x + y = 124) :
  x / y = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l101_10180


namespace NUMINAMATH_CALUDE_trigonometric_sum_equality_l101_10155

theorem trigonometric_sum_equality (θ φ : Real) 
  (h : (Real.cos θ)^6 / (Real.cos φ)^2 + (Real.sin θ)^6 / (Real.sin φ)^2 = 1) :
  ∃ (x : Real), x = (Real.sin φ)^6 / (Real.sin θ)^2 + (Real.cos φ)^6 / (Real.cos θ)^2 ∧ 
  (∀ (y : Real), y = (Real.sin φ)^6 / (Real.sin θ)^2 + (Real.cos φ)^6 / (Real.cos θ)^2 → y ≤ x) ∧
  x = 1 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_sum_equality_l101_10155


namespace NUMINAMATH_CALUDE_largest_lcm_with_18_l101_10175

theorem largest_lcm_with_18 : 
  (Nat.lcm 18 4).max 
    ((Nat.lcm 18 6).max 
      ((Nat.lcm 18 9).max 
        ((Nat.lcm 18 14).max 
          (Nat.lcm 18 18)))) = 126 := by
  sorry

end NUMINAMATH_CALUDE_largest_lcm_with_18_l101_10175


namespace NUMINAMATH_CALUDE_one_third_of_nine_x_minus_three_l101_10146

theorem one_third_of_nine_x_minus_three (x : ℝ) : (1 / 3) * (9 * x - 3) = 3 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_nine_x_minus_three_l101_10146


namespace NUMINAMATH_CALUDE_least_positive_linear_combination_l101_10161

theorem least_positive_linear_combination :
  ∃ (n : ℕ), n > 0 ∧ (∀ (m : ℕ), m > 0 → (∃ (x y : ℤ), 24 * x + 20 * y = m) → m ≥ n) ∧
  (∃ (x y : ℤ), 24 * x + 20 * y = n) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_linear_combination_l101_10161


namespace NUMINAMATH_CALUDE_product_of_values_l101_10123

theorem product_of_values (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 24 * Real.rpow 3 (1/3))
  (h2 : x * z = 40 * Real.rpow 3 (1/3))
  (h3 : y * z = 15 * Real.rpow 3 (1/3)) :
  x * y * z = 72 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_product_of_values_l101_10123


namespace NUMINAMATH_CALUDE_max_value_fraction_l101_10174

theorem max_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 9 * a^2 + b^2 = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → (x * y) / (3 * x + y) ≤ (a * b) / (3 * a + b)) →
  (a * b) / (3 * a + b) = Real.sqrt 2 / 12 :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l101_10174


namespace NUMINAMATH_CALUDE_parabola_focus_l101_10107

theorem parabola_focus (p : ℝ) :
  4 * p = 1/4 → (0, 1/(16 : ℝ)) = (0, p) := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l101_10107


namespace NUMINAMATH_CALUDE_projectile_max_height_l101_10130

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

/-- The maximum height reached by the projectile -/
theorem projectile_max_height : 
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 161 :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l101_10130


namespace NUMINAMATH_CALUDE_probability_equals_fraction_l101_10128

def num_forks : ℕ := 8
def num_spoons : ℕ := 5
def num_knives : ℕ := 7
def total_silverware : ℕ := num_forks + num_spoons + num_knives
def pieces_removed : ℕ := 4

def probability_two_forks_one_spoon_one_knife : ℚ :=
  (Nat.choose num_forks 2 * Nat.choose num_spoons 1 * Nat.choose num_knives 1) /
  Nat.choose total_silverware pieces_removed

theorem probability_equals_fraction :
  probability_two_forks_one_spoon_one_knife = 196 / 969 := by
  sorry

end NUMINAMATH_CALUDE_probability_equals_fraction_l101_10128


namespace NUMINAMATH_CALUDE_dana_marcus_pencil_difference_l101_10135

/-- Given that Dana has 15 more pencils than Jayden, Jayden has twice as many pencils as Marcus,
    and Jayden has 20 pencils, prove that Dana has 25 more pencils than Marcus. -/
theorem dana_marcus_pencil_difference :
  ∀ (dana jayden marcus : ℕ),
  dana = jayden + 15 →
  jayden = 2 * marcus →
  jayden = 20 →
  dana - marcus = 25 := by
sorry

end NUMINAMATH_CALUDE_dana_marcus_pencil_difference_l101_10135


namespace NUMINAMATH_CALUDE_average_age_of_students_average_age_proof_l101_10148

theorem average_age_of_students (total_students : Nat) 
  (group1_count : Nat) (group1_avg : Nat) 
  (group2_count : Nat) (group2_avg : Nat)
  (last_student_age : Nat) : Nat :=
  let total_age := group1_count * group1_avg + group2_count * group2_avg + last_student_age
  total_age / total_students

theorem average_age_proof :
  average_age_of_students 15 8 14 6 16 17 = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_students_average_age_proof_l101_10148


namespace NUMINAMATH_CALUDE_cleaning_time_calculation_l101_10149

/-- Represents the cleaning schedule for a person -/
structure CleaningSchedule where
  vacuuming : Nat × Nat  -- (minutes per day, days per week)
  dusting : Nat × Nat
  sweeping : Nat × Nat
  deep_cleaning : Nat × Nat

/-- Calculates the total cleaning time in minutes per week -/
def totalCleaningTime (schedule : CleaningSchedule) : Nat :=
  schedule.vacuuming.1 * schedule.vacuuming.2 +
  schedule.dusting.1 * schedule.dusting.2 +
  schedule.sweeping.1 * schedule.sweeping.2 +
  schedule.deep_cleaning.1 * schedule.deep_cleaning.2

/-- Converts minutes to hours and minutes -/
def minutesToHoursAndMinutes (minutes : Nat) : Nat × Nat :=
  (minutes / 60, minutes % 60)

/-- Aron's cleaning schedule -/
def aronSchedule : CleaningSchedule :=
  { vacuuming := (30, 3)
    dusting := (20, 2)
    sweeping := (15, 4)
    deep_cleaning := (45, 1) }

/-- Ben's cleaning schedule -/
def benSchedule : CleaningSchedule :=
  { vacuuming := (40, 2)
    dusting := (25, 3)
    sweeping := (20, 5)
    deep_cleaning := (60, 1) }

theorem cleaning_time_calculation :
  let aronTime := totalCleaningTime aronSchedule
  let benTime := totalCleaningTime benSchedule
  let aronHoursMinutes := minutesToHoursAndMinutes aronTime
  let benHoursMinutes := minutesToHoursAndMinutes benTime
  let timeDifference := benTime - aronTime
  let timeDifferenceHoursMinutes := minutesToHoursAndMinutes timeDifference
  aronHoursMinutes = (3, 55) ∧
  benHoursMinutes = (5, 15) ∧
  timeDifferenceHoursMinutes = (1, 20) := by
  sorry

end NUMINAMATH_CALUDE_cleaning_time_calculation_l101_10149


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l101_10179

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 2 < 0} = {x : ℝ | -2 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l101_10179


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l101_10168

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : b / a = 3 / 2) : 
  let e := Real.sqrt (a^2 + b^2) / a
  e = Real.sqrt 13 / 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l101_10168


namespace NUMINAMATH_CALUDE_chemical_solution_concentration_l101_10104

theorem chemical_solution_concentration 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (drained_volume : ℝ) 
  (added_concentration : ℝ) 
  (final_volume : ℝ)
  (h1 : initial_volume = 50)
  (h2 : initial_concentration = 0.60)
  (h3 : drained_volume = 35)
  (h4 : added_concentration = 0.40)
  (h5 : final_volume = initial_volume) :
  let remaining_volume := initial_volume - drained_volume
  let initial_chemical := initial_volume * initial_concentration
  let drained_chemical := drained_volume * initial_concentration
  let remaining_chemical := initial_chemical - drained_chemical
  let added_chemical := drained_volume * added_concentration
  let final_chemical := remaining_chemical + added_chemical
  let final_concentration := final_chemical / final_volume
  final_concentration = 0.46 := by
sorry

end NUMINAMATH_CALUDE_chemical_solution_concentration_l101_10104


namespace NUMINAMATH_CALUDE_relationship_between_variables_l101_10186

theorem relationship_between_variables (x y z w : ℝ) 
  (h : (x + y) / (y + z) = (2 * z + w) / (w + x)) : 
  x = 2 * z - w := by sorry

end NUMINAMATH_CALUDE_relationship_between_variables_l101_10186


namespace NUMINAMATH_CALUDE_fractional_unit_problem_l101_10158

def fractional_unit (n : ℕ) (d : ℕ) : ℚ := 1 / d

def smallest_prime : ℕ := 2

theorem fractional_unit_problem (n d : ℕ) (h : n = 13 ∧ d = 5) :
  let x := fractional_unit n d
  x = 1/5 ∧ n * x - 3 * x = smallest_prime := by sorry

end NUMINAMATH_CALUDE_fractional_unit_problem_l101_10158


namespace NUMINAMATH_CALUDE_homework_points_calculation_l101_10183

theorem homework_points_calculation (total_points : ℕ) 
  (h1 : total_points = 265)
  (h2 : ∀ (test_points quiz_points : ℕ), test_points = 4 * quiz_points)
  (h3 : ∀ (quiz_points homework_points : ℕ), quiz_points = homework_points + 5) :
  ∃ (homework_points : ℕ), 
    homework_points = 40 ∧ 
    homework_points + (homework_points + 5) + 4 * (homework_points + 5) = total_points :=
by sorry

end NUMINAMATH_CALUDE_homework_points_calculation_l101_10183


namespace NUMINAMATH_CALUDE_existence_of_integer_representation_l101_10154

theorem existence_of_integer_representation (n : ℤ) :
  ∃ (a b : ℤ), n = ⌊(a : ℝ) * Real.sqrt 2⌋ + ⌊(b : ℝ) * Real.sqrt 3⌋ := by
  sorry

end NUMINAMATH_CALUDE_existence_of_integer_representation_l101_10154


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l101_10106

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 1 ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} := by sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∃ x, f x a < 2*a} = {a : ℝ | a > 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l101_10106


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l101_10101

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | x < 1}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.compl M) ∩ N = {x : ℝ | x < -2} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l101_10101


namespace NUMINAMATH_CALUDE_abs_a_eq_5_and_a_plus_b_eq_0_l101_10134

theorem abs_a_eq_5_and_a_plus_b_eq_0 (a b : ℝ) (h1 : |a| = 5) (h2 : a + b = 0) :
  a - b = 10 ∨ a - b = -10 := by sorry

end NUMINAMATH_CALUDE_abs_a_eq_5_and_a_plus_b_eq_0_l101_10134


namespace NUMINAMATH_CALUDE_unique_four_digit_products_l101_10188

def digit_product (n : ℕ) : ℕ :=
  if n < 1000 ∨ n > 9999 then 0
  else (n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

def is_unique_product (n : ℕ) : Prop :=
  ∃! x : ℕ, x ≥ 1000 ∧ x ≤ 9999 ∧ digit_product x = n

theorem unique_four_digit_products :
  {n : ℕ | is_unique_product n} = {1, 625, 2401, 4096, 6561} :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_products_l101_10188


namespace NUMINAMATH_CALUDE_largest_angle_of_obtuse_isosceles_triangle_l101_10156

-- Define the triangle PQR
structure Triangle (P Q R : Point) where
  -- Add any necessary fields

-- Define the properties of the triangle
def isObtuse (t : Triangle P Q R) : Prop := sorry
def isIsosceles (t : Triangle P Q R) : Prop := sorry
def angleMeasure (p : Point) (t : Triangle P Q R) : ℝ := sorry
def largestAngle (t : Triangle P Q R) : ℝ := sorry

-- Theorem statement
theorem largest_angle_of_obtuse_isosceles_triangle 
  (P Q R : Point) (t : Triangle P Q R)
  (h_obtuse : isObtuse t)
  (h_isosceles : isIsosceles t)
  (h_angle_P : angleMeasure P t = 30) :
  largestAngle t = 120 := by sorry

end NUMINAMATH_CALUDE_largest_angle_of_obtuse_isosceles_triangle_l101_10156


namespace NUMINAMATH_CALUDE_max_xy_value_l101_10185

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 4 * x^2 + 9 * y^2 + 3 * x * y = 30) : 
  x * y ≤ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 4 * x₀^2 + 9 * y₀^2 + 3 * x₀ * y₀ = 30 ∧ x₀ * y₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l101_10185


namespace NUMINAMATH_CALUDE_stamp_problem_l101_10139

/-- Represents the number of ways to make a certain amount with given coin denominations -/
def numWays (amount : ℕ) (coins : List ℕ) : ℕ := sorry

/-- Represents the minimum number of coins needed to make a certain amount with given coin denominations -/
def minCoins (amount : ℕ) (coins : List ℕ) : ℕ := sorry

theorem stamp_problem :
  minCoins 74 [5, 7] = 12 := by sorry

end NUMINAMATH_CALUDE_stamp_problem_l101_10139


namespace NUMINAMATH_CALUDE_travis_payment_l101_10176

/-- Calculates the payment for Travis given the specified conditions --/
def calculate_payment (total_bowls : ℕ) (base_fee : ℚ) (safe_delivery_fee : ℚ) 
  (broken_glass_charge : ℚ) (broken_ceramic_charge : ℚ) (lost_glass_charge : ℚ) 
  (lost_ceramic_charge : ℚ) (glass_weight : ℚ) (ceramic_weight : ℚ) 
  (weight_fee : ℚ) (lost_glass : ℕ) (lost_ceramic : ℕ) (broken_glass : ℕ) 
  (broken_ceramic : ℕ) : ℚ :=
  let safe_bowls := total_bowls - (lost_glass + lost_ceramic + broken_glass + broken_ceramic)
  let safe_payment := safe_delivery_fee * safe_bowls
  let broken_lost_charges := broken_glass_charge * broken_glass + 
                             broken_ceramic_charge * broken_ceramic +
                             lost_glass_charge * lost_glass + 
                             lost_ceramic_charge * lost_ceramic
  let total_weight := glass_weight * (total_bowls - lost_ceramic - broken_ceramic) + 
                      ceramic_weight * (total_bowls - lost_glass - broken_glass)
  let weight_charge := weight_fee * total_weight
  base_fee + safe_payment - broken_lost_charges + weight_charge

/-- The payment for Travis should be $2894.25 given the specified conditions --/
theorem travis_payment : 
  calculate_payment 638 100 3 5 4 6 3 2 (3/2) (1/2) 9 3 10 5 = 2894.25 := by
  sorry

end NUMINAMATH_CALUDE_travis_payment_l101_10176


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l101_10192

theorem complex_number_in_second_quadrant :
  let z : ℂ := (1 + Complex.I) / (1 - Complex.I)^2
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l101_10192


namespace NUMINAMATH_CALUDE_log_expression_equality_l101_10115

theorem log_expression_equality : 
  Real.sqrt (Real.log 18 / Real.log 4 - Real.log 18 / Real.log 9 + Real.log 9 / Real.log 2) = 
  (3 * Real.log 3 - Real.log 2) / Real.sqrt (2 * Real.log 3 * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l101_10115


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l101_10102

theorem cubic_inequality_solution (x : ℝ) : 
  x^3 - 9*x^2 + 24*x > 0 ↔ (0 < x ∧ x < 3) ∨ (x > 8) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l101_10102


namespace NUMINAMATH_CALUDE_ratio_m_n_l101_10141

theorem ratio_m_n (m n : ℕ) (h1 : m > n) (h2 : ¬(n ∣ m)) 
  (h3 : m % n = (m + n) % (m - n)) : m / n = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_m_n_l101_10141


namespace NUMINAMATH_CALUDE_square_length_CD_l101_10117

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = -3 * x^2 + 2 * x + 5

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the problem statement
theorem square_length_CD (C D : PointOnParabola) : 
  (C.x = -D.x ∧ C.y = -D.y) → (C.x - D.x)^2 + (C.y - D.y)^2 = 100/3 := by
  sorry

end NUMINAMATH_CALUDE_square_length_CD_l101_10117


namespace NUMINAMATH_CALUDE_cos_squared_sixty_degrees_l101_10177

theorem cos_squared_sixty_degrees :
  let cos_sixty : ℝ := 1 / 2
  (cos_sixty ^ 2 : ℝ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_sixty_degrees_l101_10177


namespace NUMINAMATH_CALUDE_quadratic_solution_l101_10160

theorem quadratic_solution (m : ℝ) : 
  (2 : ℝ)^2 - m * 2 + 8 = 0 → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l101_10160


namespace NUMINAMATH_CALUDE_freds_spending_ratio_l101_10163

/-- The ratio of Fred's movie spending to his weekly allowance -/
def movie_allowance_ratio (weekly_allowance : ℚ) (car_wash_earnings : ℚ) (final_amount : ℚ) : ℚ × ℚ :=
  let total_before_movies := final_amount + car_wash_earnings
  let movie_spending := total_before_movies - weekly_allowance
  (movie_spending, weekly_allowance)

/-- Theorem stating the ratio of Fred's movie spending to his weekly allowance -/
theorem freds_spending_ratio :
  let weekly_allowance : ℚ := 16
  let car_wash_earnings : ℚ := 6
  let final_amount : ℚ := 14
  let (numerator, denominator) := movie_allowance_ratio weekly_allowance car_wash_earnings final_amount
  numerator / denominator = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_freds_spending_ratio_l101_10163


namespace NUMINAMATH_CALUDE_convex_pentagon_integer_point_l101_10119

-- Define a point in 2D space
structure Point where
  x : ℤ
  y : ℤ

-- Define a pentagon as a list of 5 points
def Pentagon := List Point

-- Define a predicate to check if a pentagon is convex
def isConvex (p : Pentagon) : Prop := sorry

-- Define a predicate to check if a point is inside or on the boundary of a pentagon
def isInsideOrOnBoundary (point : Point) (p : Pentagon) : Prop := sorry

-- The main theorem
theorem convex_pentagon_integer_point 
  (p : Pentagon) 
  (h1 : p.length = 5) 
  (h2 : isConvex p) : 
  ∃ (point : Point), isInsideOrOnBoundary point p := by
  sorry

end NUMINAMATH_CALUDE_convex_pentagon_integer_point_l101_10119


namespace NUMINAMATH_CALUDE_unique_triplet_solution_l101_10165

theorem unique_triplet_solution : 
  ∃! (x y z : ℕ+), 
    (x^y.val + y.val^x.val = z.val^y.val) ∧ 
    (x^y.val + 2012 = y.val^(z.val + 1)) ∧
    x = 6 ∧ y = 2 ∧ z = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_triplet_solution_l101_10165


namespace NUMINAMATH_CALUDE_expected_successful_trials_l101_10169

/-- The probability of getting neither a 5 nor a 6 on a single die -/
def p_failure_single : ℚ := 2/3

/-- The probability of a successful experiment (at least one 5 or 6 in three dice) -/
def p_success : ℚ := 1 - p_failure_single^3

/-- The number of experiments conducted -/
def num_experiments : ℕ := 54

/-- The expected number of successful trials in a binomial distribution -/
def expected_successes (p : ℚ) (n : ℕ) : ℚ := p * n

theorem expected_successful_trials :
  expected_successes p_success num_experiments = 38 := by sorry

end NUMINAMATH_CALUDE_expected_successful_trials_l101_10169


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l101_10131

/-- A coloring function that assigns a color (represented by a Boolean) to each point with integer coordinates. -/
def Coloring := ℤ × ℤ → Bool

/-- Predicate to check if a rectangle satisfies the required properties. -/
def ValidRectangle (a b c d : ℤ × ℤ) : Prop :=
  let (x₁, y₁) := a
  let (x₂, y₂) := b
  let (x₃, y₃) := c
  let (x₄, y₄) := d
  (x₁ = x₄ ∧ x₂ = x₃ ∧ y₁ = y₂ ∧ y₃ = y₄) ∧ 
  (∃ k : ℕ, (x₂ - x₁).natAbs * (y₃ - y₁).natAbs = 2^k)

/-- Theorem stating that there exists a coloring such that no valid rectangle has all vertices of the same color. -/
theorem exists_valid_coloring : 
  ∃ (f : Coloring), ∀ (a b c d : ℤ × ℤ), 
    ValidRectangle a b c d → 
    ¬(f a = f b ∧ f b = f c ∧ f c = f d) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l101_10131


namespace NUMINAMATH_CALUDE_stratified_sample_size_l101_10172

/-- Represents the composition of a school population -/
structure SchoolPopulation where
  teachers : ℕ
  male_students : ℕ
  female_students : ℕ

/-- Represents a stratified sample from the school population -/
structure StratifiedSample where
  total_size : ℕ
  female_sample : ℕ

/-- Theorem: Given a school population and a stratified sample where 80 people are drawn
    from the female students, the total sample size is 192 -/
theorem stratified_sample_size 
  (pop : SchoolPopulation) 
  (sample : StratifiedSample) :
  pop.teachers = 200 →
  pop.male_students = 1200 →
  pop.female_students = 1000 →
  sample.female_sample = 80 →
  sample.total_size = 192 := by
  sorry

#check stratified_sample_size

end NUMINAMATH_CALUDE_stratified_sample_size_l101_10172


namespace NUMINAMATH_CALUDE_scientific_notation_460_billion_l101_10184

theorem scientific_notation_460_billion :
  (460 * 10^9 : ℝ) = 4.6 * 10^11 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_460_billion_l101_10184


namespace NUMINAMATH_CALUDE_nolan_saving_months_l101_10187

def monthly_savings : ℕ := 3000
def total_saved : ℕ := 36000

theorem nolan_saving_months :
  total_saved / monthly_savings = 12 :=
by sorry

end NUMINAMATH_CALUDE_nolan_saving_months_l101_10187


namespace NUMINAMATH_CALUDE_initially_tagged_fish_count_l101_10105

/-- The number of fish initially caught and tagged in a pond -/
def initially_tagged_fish (total_fish : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) : ℕ :=
  (tagged_in_second * total_fish) / second_catch

/-- Theorem stating that the number of initially tagged fish is 50 -/
theorem initially_tagged_fish_count :
  initially_tagged_fish 250 50 10 = 50 := by
  sorry

#eval initially_tagged_fish 250 50 10

end NUMINAMATH_CALUDE_initially_tagged_fish_count_l101_10105


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l101_10173

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s * s = 144 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l101_10173


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l101_10111

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l101_10111


namespace NUMINAMATH_CALUDE_comparison_of_powers_l101_10195

theorem comparison_of_powers : 0.2^3 < 2^0.3 := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_powers_l101_10195


namespace NUMINAMATH_CALUDE_word_permutation_ratio_l101_10162

theorem word_permutation_ratio : 
  let n₁ : ℕ := 6  -- number of letters in "СКАЛКА"
  let n₂ : ℕ := 7  -- number of letters in "ТЕФТЕЛЬ"
  let r : ℕ := 2   -- number of repeated letters in each word
  
  -- number of distinct permutations for each word
  let perm₁ : ℕ := n₁! / (r! * r!)
  let perm₂ : ℕ := n₂! / (r! * r!)

  perm₂ / perm₁ = 7 := by
  sorry

end NUMINAMATH_CALUDE_word_permutation_ratio_l101_10162


namespace NUMINAMATH_CALUDE_polygon_sides_count_l101_10190

/-- Represents the number of degrees in a circle -/
def degrees_in_circle : ℝ := 360

/-- Represents the common difference in the arithmetic progression of angles -/
def common_difference : ℝ := 3

/-- Represents the measure of the largest angle in the polygon -/
def largest_angle : ℝ := 150

/-- Theorem: A convex polygon with interior angles in arithmetic progression,
    a common difference of 3°, and the largest angle of 150° has 48 sides -/
theorem polygon_sides_count :
  ∀ n : ℕ,
  (n > 2) →
  (n * (2 * largest_angle - (n - 1) * common_difference) / 2 = (n - 2) * degrees_in_circle / 2) →
  n = 48 := by
  sorry


end NUMINAMATH_CALUDE_polygon_sides_count_l101_10190


namespace NUMINAMATH_CALUDE_friend_lunch_cost_l101_10145

theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 15 → difference = 1 → friend_cost = total / 2 + difference / 2 → friend_cost = 8 := by
sorry

end NUMINAMATH_CALUDE_friend_lunch_cost_l101_10145


namespace NUMINAMATH_CALUDE_lighting_effect_improves_l101_10152

theorem lighting_effect_improves (a b m : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : 0 < m) : 
  a / b < (a + m) / (b + m) := by
  sorry

end NUMINAMATH_CALUDE_lighting_effect_improves_l101_10152


namespace NUMINAMATH_CALUDE_apples_per_pie_is_seven_l101_10166

/-- Calculates the number of apples used per pie given the initial conditions -/
def apples_per_pie (
  total_apples : ℕ)
  (num_children : ℕ)
  (apples_per_child : ℕ)
  (num_pies : ℕ)
  (remaining_apples : ℕ) : ℕ :=
  let apples_for_teachers := num_children * apples_per_child
  let apples_for_pies := total_apples - apples_for_teachers - remaining_apples
  apples_for_pies / num_pies

/-- Proves that the number of apples used per pie is 7 under the given conditions -/
theorem apples_per_pie_is_seven :
  apples_per_pie 50 2 6 2 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_pie_is_seven_l101_10166


namespace NUMINAMATH_CALUDE_broken_flagpole_l101_10182

theorem broken_flagpole (h : ℝ) (d : ℝ) (x : ℝ) 
  (height_cond : h = 10)
  (distance_cond : d = 4) :
  (x^2 + d^2 = (h - x)^2) → x = 2 * Real.sqrt 22 :=
by sorry

end NUMINAMATH_CALUDE_broken_flagpole_l101_10182


namespace NUMINAMATH_CALUDE_turn_over_five_most_effective_l101_10121

-- Define the type for card sides
inductive CardSide
| Letter (c : Char)
| Number (n : Nat)

-- Define a card as a pair of sides
def Card := (CardSide × CardSide)

-- Define a function to check if a character is a vowel
def isVowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U']

-- Define a function to check if a number is even
def isEven (n : Nat) : Bool :=
  n % 2 = 0

-- Define Jane's claim as a function
def janesClaimHolds (card : Card) : Bool :=
  match card with
  | (CardSide.Letter c, CardSide.Number n) => ¬(isVowel c) ∨ isEven n
  | (CardSide.Number n, CardSide.Letter c) => ¬(isVowel c) ∨ isEven n
  | _ => true

-- Define the set of cards on the table
def cardsOnTable : List Card := [
  (CardSide.Letter 'A', CardSide.Number 0),  -- 0 is a placeholder
  (CardSide.Letter 'T', CardSide.Number 0),
  (CardSide.Letter 'U', CardSide.Number 0),
  (CardSide.Number 5, CardSide.Letter ' '),  -- ' ' is a placeholder
  (CardSide.Number 8, CardSide.Letter ' '),
  (CardSide.Number 10, CardSide.Letter ' '),
  (CardSide.Number 14, CardSide.Letter ' ')
]

-- Theorem: Turning over the card with 5 is the most effective way to potentially disprove Jane's claim
theorem turn_over_five_most_effective :
  ∃ (card : Card), card ∈ cardsOnTable ∧ 
  (∃ (c : Char), card = (CardSide.Number 5, CardSide.Letter c)) ∧
  (∀ (otherCard : Card), otherCard ∈ cardsOnTable → otherCard ≠ card →
    (∃ (possibleChar : Char), 
      ¬(janesClaimHolds (CardSide.Number 5, CardSide.Letter possibleChar)) →
      (janesClaimHolds otherCard ∨ 
       ∀ (possibleNum : Nat), janesClaimHolds (CardSide.Letter possibleChar, CardSide.Number possibleNum))))
  := by sorry


end NUMINAMATH_CALUDE_turn_over_five_most_effective_l101_10121


namespace NUMINAMATH_CALUDE_quadratic_one_root_l101_10159

theorem quadratic_one_root (c b : ℝ) (hc : c > 0) :
  (∃! x : ℝ, x^2 + 2 * Real.sqrt c * x + b = 0) → c = b := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l101_10159


namespace NUMINAMATH_CALUDE_negative_reciprocal_of_opposite_of_neg_abs_three_l101_10197

-- Definition of opposite numbers
def opposite (a b : ℝ) : Prop := a = -b

-- Definition of reciprocal
def reciprocal (a b : ℝ) : Prop := a * b = 1

-- Theorem to prove
theorem negative_reciprocal_of_opposite_of_neg_abs_three :
  ∃ x : ℝ, opposite (-|(-3)|) x ∧ reciprocal (-1/3) (-x) := by sorry

end NUMINAMATH_CALUDE_negative_reciprocal_of_opposite_of_neg_abs_three_l101_10197


namespace NUMINAMATH_CALUDE_pokemon_card_count_l101_10132

/-- The number of people who have Pokemon cards -/
def num_people : ℕ := 4

/-- The number of Pokemon cards each person has -/
def cards_per_person : ℕ := 14

/-- The total number of Pokemon cards -/
def total_cards : ℕ := num_people * cards_per_person

theorem pokemon_card_count : total_cards = 56 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_card_count_l101_10132


namespace NUMINAMATH_CALUDE_cafeteria_red_apples_l101_10108

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := 43

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 32

/-- The number of students who wanted fruit -/
def students_wanting_fruit : ℕ := 2

/-- The number of extra apples -/
def extra_apples : ℕ := 73

/-- Theorem stating that the number of red apples ordered by the cafeteria is 43 -/
theorem cafeteria_red_apples :
  red_apples = 43 :=
by
  sorry

#check cafeteria_red_apples

end NUMINAMATH_CALUDE_cafeteria_red_apples_l101_10108


namespace NUMINAMATH_CALUDE_train_travel_times_l101_10114

theorem train_travel_times
  (usual_speed_A : ℝ)
  (usual_speed_B : ℝ)
  (distance_XM : ℝ)
  (h1 : usual_speed_A > 0)
  (h2 : usual_speed_B > 0)
  (h3 : distance_XM > 0)
  (h4 : usual_speed_B * 2 = usual_speed_A * 3) :
  let t : ℝ := 180
  let current_speed_A : ℝ := (6 / 7) * usual_speed_A
  let time_XM_reduced : ℝ := distance_XM / current_speed_A
  let time_XM_usual : ℝ := distance_XM / usual_speed_A
  let time_XY_A : ℝ := 3 * time_XM_usual
  let time_XY_B : ℝ := 810
  (time_XM_reduced = time_XM_usual + 30) ∧
  (time_XM_usual = t) ∧
  (time_XY_B = 1.5 * time_XY_A) := by
  sorry

end NUMINAMATH_CALUDE_train_travel_times_l101_10114


namespace NUMINAMATH_CALUDE_farmer_animals_l101_10189

theorem farmer_animals (cows pigs goats chickens ducks sheep : ℕ) : 
  pigs = 3 * cows →
  cows = goats + 7 →
  chickens = 2 * (cows + pigs) →
  2 * ducks = goats + chickens →
  sheep = cows + chickens + 5 →
  cows + pigs + goats + chickens + ducks + sheep = 346 →
  goats = 6 := by
sorry

end NUMINAMATH_CALUDE_farmer_animals_l101_10189


namespace NUMINAMATH_CALUDE_expression_value_l101_10122

theorem expression_value (x y z : ℤ) (hx : x = 3) (hy : y = 2) (hz : z = 1) :
  3 * x - 2 * y + 4 * z = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l101_10122


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_range_of_a_for_not_r_necessary_not_sufficient_for_not_p_l101_10157

-- Define the predicates p, q, and r
def p (x : ℝ) : Prop := |3*x - 4| > 2
def q (x : ℝ) : Prop := 1 / (x^2 - x - 2) > 0
def r (x a : ℝ) : Prop := (x - a) * (x - a - 1) < 0

-- Define the negations of p, q, and r
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)
def not_r (x a : ℝ) : Prop := ¬(r x a)

-- Theorem 1: ¬p is a sufficient but not necessary condition for ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, not_p x → not_q x) ∧ ¬(∀ x, not_q x → not_p x) :=
sorry

-- Theorem 2: Range of a for which ¬r is a necessary but not sufficient condition for ¬p
theorem range_of_a_for_not_r_necessary_not_sufficient_for_not_p :
  ∀ a, (∀ x, not_p x → not_r x a) ∧ ¬(∀ x, not_r x a → not_p x) ↔ (a ≥ 2 ∨ a ≤ -1/3) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_range_of_a_for_not_r_necessary_not_sufficient_for_not_p_l101_10157


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l101_10171

theorem decreasing_function_inequality (f : ℝ → ℝ) (a b : ℝ) 
  (h_decreasing : ∀ x y, x ≤ y → f x ≥ f y) 
  (h_sum : a + b ≤ 0) : 
  f a + f b ≥ f (-a) + f (-b) := by
sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l101_10171


namespace NUMINAMATH_CALUDE_plain_cookie_price_l101_10138

/-- The price of each box of plain cookies, given the total number of boxes sold,
    the combined value of all boxes, the number of plain cookie boxes sold,
    and the price of each box of chocolate chip cookies. -/
theorem plain_cookie_price
  (total_boxes : ℝ)
  (combined_value : ℝ)
  (plain_boxes : ℝ)
  (choc_chip_price : ℝ)
  (h1 : total_boxes = 1585)
  (h2 : combined_value = 1586.75)
  (h3 : plain_boxes = 793.375)
  (h4 : choc_chip_price = 1.25) :
  (combined_value - (total_boxes - plain_boxes) * choc_chip_price) / plain_boxes = 0.7525 := by
  sorry

end NUMINAMATH_CALUDE_plain_cookie_price_l101_10138


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l101_10151

theorem cubic_sum_theorem (a b c : ℝ) 
  (h1 : a + b + c = 2) 
  (h2 : a * b + a * c + b * c = 3) 
  (h3 : a * b * c = 1) : 
  a^3 + b^3 + c^3 = 5 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l101_10151


namespace NUMINAMATH_CALUDE_soup_cans_feeding_l101_10147

/-- Proves that given 8 total cans of soup, where each can feeds either 4 adults or 6 children,
    and 18 children have been fed, the number of adults that can be fed with the remaining soup is 20. -/
theorem soup_cans_feeding (total_cans : ℕ) (adults_per_can children_per_can : ℕ) (children_fed : ℕ) :
  total_cans = 8 →
  adults_per_can = 4 →
  children_per_can = 6 →
  children_fed = 18 →
  (total_cans - (children_fed / children_per_can)) * adults_per_can = 20 :=
by sorry

end NUMINAMATH_CALUDE_soup_cans_feeding_l101_10147


namespace NUMINAMATH_CALUDE_temperature_difference_l101_10144

def lowest_temp : ℤ := -4
def highest_temp : ℤ := 5

theorem temperature_difference : highest_temp - lowest_temp = 9 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l101_10144


namespace NUMINAMATH_CALUDE_f_at_two_l101_10103

/-- The polynomial f(x) = x^5 + 2x^4 + 3x^3 + 4x^2 + 5x + 6 -/
def f (x : ℝ) : ℝ := x^5 + 2*x^4 + 3*x^3 + 4*x^2 + 5*x + 6

/-- Theorem: The value of f(x) when x = 2 is 216 -/
theorem f_at_two : f 2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_f_at_two_l101_10103


namespace NUMINAMATH_CALUDE_initial_average_runs_l101_10109

theorem initial_average_runs (initial_matches : ℕ) (additional_runs : ℕ) (average_increase : ℕ) : 
  initial_matches = 10 →
  additional_runs = 89 →
  average_increase = 5 →
  ∃ (initial_average : ℕ),
    (initial_matches * initial_average + additional_runs) / (initial_matches + 1) = initial_average + average_increase ∧
    initial_average = 34 :=
by sorry

end NUMINAMATH_CALUDE_initial_average_runs_l101_10109


namespace NUMINAMATH_CALUDE_discounted_price_per_shirt_l101_10137

-- Define the given conditions
def number_of_shirts : ℕ := 3
def original_total_price : ℚ := 60
def discount_percentage : ℚ := 40

-- Define the theorem
theorem discounted_price_per_shirt :
  let discount_amount : ℚ := (discount_percentage / 100) * original_total_price
  let sale_price : ℚ := original_total_price - discount_amount
  let price_per_shirt : ℚ := sale_price / number_of_shirts
  price_per_shirt = 12 := by sorry

end NUMINAMATH_CALUDE_discounted_price_per_shirt_l101_10137


namespace NUMINAMATH_CALUDE_number_increase_l101_10153

theorem number_increase (n : ℕ) (m : ℕ) (increase : ℕ) : n = 18 → m = 12 → increase = m * n - n := by
  sorry

end NUMINAMATH_CALUDE_number_increase_l101_10153


namespace NUMINAMATH_CALUDE_cube_face_sum_l101_10120

theorem cube_face_sum (a b c d e f : ℕ+) : 
  (a * b * c + a * e * c + a * b * f + a * e * f + 
   d * b * c + d * e * c + d * b * f + d * e * f = 1729) → 
  (a + b + c + d + e + f = 39) := by
sorry

end NUMINAMATH_CALUDE_cube_face_sum_l101_10120


namespace NUMINAMATH_CALUDE_store_profit_optimization_l101_10110

/-- Represents the store's sales and profit model -/
structure StoreSalesModel where
  purchase_price : ℕ
  initial_selling_price : ℕ
  initial_monthly_sales : ℕ
  additional_sales_per_yuan : ℕ

/-- Calculates the monthly profit given a price reduction -/
def monthly_profit (model : StoreSalesModel) (price_reduction : ℕ) : ℕ :=
  let new_price := model.initial_selling_price - price_reduction
  let new_sales := model.initial_monthly_sales + model.additional_sales_per_yuan * price_reduction
  (new_price - model.purchase_price) * new_sales

/-- Theorem stating the initial monthly profit and the optimal price reduction -/
theorem store_profit_optimization (model : StoreSalesModel) 
  (h1 : model.purchase_price = 280)
  (h2 : model.initial_selling_price = 360)
  (h3 : model.initial_monthly_sales = 60)
  (h4 : model.additional_sales_per_yuan = 5) :
  (monthly_profit model 0 = 4800) ∧ 
  (monthly_profit model 60 = 7200) ∧ 
  (∀ x, x ≠ 60 → monthly_profit model x ≤ 7200) :=
sorry

end NUMINAMATH_CALUDE_store_profit_optimization_l101_10110


namespace NUMINAMATH_CALUDE_movie_of_the_year_threshold_l101_10164

def total_members : ℕ := 775
def threshold : ℚ := 1/4

theorem movie_of_the_year_threshold : 
  ∀ n : ℕ, (n : ℚ) ≥ threshold * total_members ∧ 
  ∀ m : ℕ, m < n → (m : ℚ) < threshold * total_members → n = 194 := by
  sorry

end NUMINAMATH_CALUDE_movie_of_the_year_threshold_l101_10164


namespace NUMINAMATH_CALUDE_animus_tower_workers_l101_10142

theorem animus_tower_workers (beavers spiders : ℕ) 
  (h1 : beavers = 318) 
  (h2 : spiders = 544) : 
  beavers + spiders = 862 := by
  sorry

end NUMINAMATH_CALUDE_animus_tower_workers_l101_10142


namespace NUMINAMATH_CALUDE_max_k_value_l101_10196

theorem max_k_value (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ x y, x > 0 → y > 0 → (x + 2*y) / (x*y) ≥ k / (2*x + y)) →
  k ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l101_10196


namespace NUMINAMATH_CALUDE_max_students_planting_trees_l101_10116

theorem max_students_planting_trees :
  ∀ (a b : ℕ),
  3 * a + 5 * b = 115 →
  ∀ (x y : ℕ),
  3 * x + 5 * y = 115 →
  a + b ≥ x + y →
  a + b = 37 :=
by sorry

end NUMINAMATH_CALUDE_max_students_planting_trees_l101_10116


namespace NUMINAMATH_CALUDE_set_problem_l101_10150

def U : Set ℕ := {x | x ≤ 20 ∧ Nat.Prime x}

theorem set_problem (A B : Set ℕ)
  (h1 : A ∩ (U \ B) = {3, 5})
  (h2 : (U \ A) ∩ B = {7, 19})
  (h3 : U \ (A ∪ B) = {2, 17}) :
  A = {3, 5, 11, 13} ∧ B = {7, 19, 11, 13} := by
  sorry

end NUMINAMATH_CALUDE_set_problem_l101_10150


namespace NUMINAMATH_CALUDE_omega_is_abc_l101_10113

theorem omega_is_abc (ω a b c x y z : ℝ) 
  (distinct : ω ≠ a ∧ ω ≠ b ∧ ω ≠ c ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (eq1 : x + y + z = 1)
  (eq2 : a^2 * x + b^2 * y + c^2 * z = ω^2)
  (eq3 : a^3 * x + b^3 * y + c^3 * z = ω^3)
  (eq4 : a^4 * x + b^4 * y + c^4 * z = ω^4) :
  ω = a ∨ ω = b ∨ ω = c :=
sorry

end NUMINAMATH_CALUDE_omega_is_abc_l101_10113


namespace NUMINAMATH_CALUDE_vector_equality_l101_10194

-- Define the triangle ABC and point D
variable (A B C D : ℝ × ℝ)

-- Define vectors
def AB : ℝ × ℝ := B - A
def AC : ℝ × ℝ := C - A
def AD : ℝ × ℝ := D - A
def BC : ℝ × ℝ := C - B
def BD : ℝ × ℝ := D - B
def DC : ℝ × ℝ := C - D

-- State the theorem
theorem vector_equality (h : BD = 3 • DC) : AD = (1/4) • AB + (3/4) • AC := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_l101_10194


namespace NUMINAMATH_CALUDE_lewis_total_earnings_l101_10100

/-- Lewis's weekly earnings in dollars -/
def weekly_earnings : ℕ := 92

/-- Number of weeks Lewis works during the harvest -/
def weeks_worked : ℕ := 5

/-- Theorem stating Lewis's total earnings during the harvest -/
theorem lewis_total_earnings : weekly_earnings * weeks_worked = 460 := by
  sorry

end NUMINAMATH_CALUDE_lewis_total_earnings_l101_10100


namespace NUMINAMATH_CALUDE_abc_theorem_l101_10191

theorem abc_theorem (a b c : ℕ+) (x y z w : ℝ) 
  (h_order : a ≤ b ∧ b ≤ c)
  (h_eq : (a : ℝ) ^ x = (b : ℝ) ^ y ∧ (b : ℝ) ^ y = (c : ℝ) ^ z ∧ (c : ℝ) ^ z = 70 ^ w)
  (h_sum : 1 / x + 1 / y + 1 / z = 1 / w) :
  c = 7 := by
  sorry

end NUMINAMATH_CALUDE_abc_theorem_l101_10191


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l101_10143

theorem quadratic_inequality_solution_set (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) →
  b^2 - 4*a*c < 0 ∧
  ¬(b^2 - 4*a*c < 0 → ∀ x : ℝ, a * x^2 + b * x + c > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l101_10143


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l101_10199

theorem rectangular_prism_volume 
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 15)
  (h_front : front_area = 10)
  (h_bottom : bottom_area = 6) :
  ∃ (a b c : ℝ), 
    a * b = side_area ∧ 
    b * c = front_area ∧ 
    c * a = bottom_area ∧ 
    a * b * c = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l101_10199


namespace NUMINAMATH_CALUDE_gross_revenue_increase_l101_10170

theorem gross_revenue_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (price_reduction_percent : ℝ) 
  (quantity_increase_percent : ℝ) 
  (h1 : price_reduction_percent = 20) 
  (h2 : quantity_increase_percent = 50) :
  let new_price := original_price * (1 - price_reduction_percent / 100)
  let new_quantity := original_quantity * (1 + quantity_increase_percent / 100)
  let original_gross := original_price * original_quantity
  let new_gross := new_price * new_quantity
  (new_gross - original_gross) / original_gross * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_gross_revenue_increase_l101_10170


namespace NUMINAMATH_CALUDE_remainder_19_pow_60_mod_7_l101_10140

theorem remainder_19_pow_60_mod_7 : 19^60 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_19_pow_60_mod_7_l101_10140


namespace NUMINAMATH_CALUDE_organization_growth_l101_10198

/-- Represents the number of people in the organization at year k -/
def people_count (k : ℕ) : ℕ :=
  if k = 0 then 30
  else 3 * people_count (k - 1) - 20

/-- The number of leaders in the organization each year -/
def num_leaders : ℕ := 10

/-- The initial number of people in the organization -/
def initial_people : ℕ := 30

theorem organization_growth :
  people_count 10 = 1180990 :=
sorry

end NUMINAMATH_CALUDE_organization_growth_l101_10198


namespace NUMINAMATH_CALUDE_sine_function_period_l101_10129

theorem sine_function_period (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, ∃ y : ℝ, y = a * Real.sin (b * x + c) + d) →
  (∃ x : ℝ, x = 4 * Real.pi ∧ (x / (2 * Real.pi / b) = 5)) →
  b = 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_sine_function_period_l101_10129


namespace NUMINAMATH_CALUDE_max_planes_of_symmetry_l101_10193

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  planes_of_symmetry : ℕ

/-- Two convex polyhedra do not intersect -/
def do_not_intersect (A B : ConvexPolyhedron) : Prop :=
  sorry

/-- The number of planes of symmetry for a figure consisting of two polyhedra -/
def combined_planes_of_symmetry (A B : ConvexPolyhedron) : ℕ :=
  sorry

theorem max_planes_of_symmetry (A B : ConvexPolyhedron) 
  (h1 : do_not_intersect A B)
  (h2 : A.planes_of_symmetry = 2012)
  (h3 : B.planes_of_symmetry = 2013) :
  combined_planes_of_symmetry A B = 2013 :=
sorry

end NUMINAMATH_CALUDE_max_planes_of_symmetry_l101_10193


namespace NUMINAMATH_CALUDE_gcf_of_18_and_10_l101_10167

theorem gcf_of_18_and_10 (h : Nat.lcm 18 10 = 36) : Nat.gcd 18 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_18_and_10_l101_10167


namespace NUMINAMATH_CALUDE_letters_ratio_l101_10136

/-- Proves that the ratio of letters Greta's mother received to the total letters Greta and her brother received is 2:1 -/
theorem letters_ratio (greta_letters brother_letters mother_letters : ℕ) : 
  greta_letters = brother_letters + 10 →
  brother_letters = 40 →
  greta_letters + brother_letters + mother_letters = 270 →
  ∃ k : ℕ, mother_letters = k * (greta_letters + brother_letters) →
  mother_letters = 2 * (greta_letters + brother_letters) := by
sorry

end NUMINAMATH_CALUDE_letters_ratio_l101_10136


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_l101_10127

theorem smallest_resolvable_debt (pig_value chicken_value : ℕ) 
  (h_pig : pig_value = 250) (h_chicken : chicken_value = 175) :
  ∃ (debt : ℕ), debt > 0 ∧ 
  (∃ (p c : ℤ), debt = pig_value * p + chicken_value * c) ∧
  (∀ (d : ℕ), d > 0 → d < debt → 
    ¬∃ (p c : ℤ), d = pig_value * p + chicken_value * c) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_l101_10127


namespace NUMINAMATH_CALUDE_sandwich_toppings_combinations_l101_10178

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

theorem sandwich_toppings_combinations :
  choose 9 3 = 84 := by sorry

end NUMINAMATH_CALUDE_sandwich_toppings_combinations_l101_10178


namespace NUMINAMATH_CALUDE_sum_of_multiples_plus_eleven_l101_10126

theorem sum_of_multiples_plus_eleven : 3 * 13 + 3 * 14 + 3 * 17 + 11 = 143 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_plus_eleven_l101_10126
