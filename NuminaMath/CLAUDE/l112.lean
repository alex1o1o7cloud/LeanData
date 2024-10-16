import Mathlib

namespace NUMINAMATH_CALUDE_cheeseBalls35ozBarrel_l112_11211

/-- Calculates the number of cheese balls in a barrel given its size in ounces -/
def cheeseBallsInBarrel (barrelSize : ℕ) : ℕ :=
  let servingsIn24oz : ℕ := 60
  let cheeseBallsPerServing : ℕ := 12
  let cheeseBallsPer24oz : ℕ := servingsIn24oz * cheeseBallsPerServing
  let cheeseBallsPerOz : ℕ := cheeseBallsPer24oz / 24
  barrelSize * cheeseBallsPerOz

theorem cheeseBalls35ozBarrel :
  cheeseBallsInBarrel 35 = 1050 :=
by sorry

end NUMINAMATH_CALUDE_cheeseBalls35ozBarrel_l112_11211


namespace NUMINAMATH_CALUDE_inequality_proof_l112_11264

theorem inequality_proof (x : ℝ) : 
  x ∈ Set.Icc (1/4 : ℝ) 3 → x ≠ 2 → x ≠ 0 → (x - 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l112_11264


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_property_l112_11240

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The sum of two consecutive terms in a sequence -/
def ConsecutiveSum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a n + a (n + 1)

theorem geometric_sequence_sum_property
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_sum1 : ConsecutiveSum a 1 = 16)
  (h_sum2 : ConsecutiveSum a 3 = 24) :
  ConsecutiveSum a 7 = 54 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_property_l112_11240


namespace NUMINAMATH_CALUDE_board_division_theorem_l112_11202

/-- Represents a cell on the board -/
structure Cell :=
  (x : Nat) (y : Nat) (shaded : Bool)

/-- Represents the board -/
def Board := List Cell

/-- Represents a rectangle on the board -/
structure Rectangle :=
  (topLeft : Cell) (width : Nat) (height : Nat)

/-- The initial board configuration -/
def initialBoard : Board := sorry

/-- Check if a cell is within a rectangle -/
def isInRectangle (cell : Cell) (rect : Rectangle) : Bool := sorry

/-- Count shaded cells in a rectangle -/
def countShadedCells (board : Board) (rect : Rectangle) : Nat := sorry

/-- Check if two rectangles are identical -/
def areIdenticalRectangles (rect1 rect2 : Rectangle) : Bool := sorry

/-- Main theorem -/
theorem board_division_theorem (board : Board) :
  ∃ (rect1 rect2 rect3 rect4 : Rectangle),
    (rect1.width = 4 ∧ rect1.height = 2) ∧
    (rect2.width = 4 ∧ rect2.height = 2) ∧
    (rect3.width = 4 ∧ rect3.height = 2) ∧
    (rect4.width = 4 ∧ rect4.height = 2) ∧
    areIdenticalRectangles rect1 rect2 ∧
    areIdenticalRectangles rect1 rect3 ∧
    areIdenticalRectangles rect1 rect4 ∧
    countShadedCells board rect1 = 3 ∧
    countShadedCells board rect2 = 3 ∧
    countShadedCells board rect3 = 3 ∧
    countShadedCells board rect4 = 3 :=
  sorry

end NUMINAMATH_CALUDE_board_division_theorem_l112_11202


namespace NUMINAMATH_CALUDE_sin_negative_120_degrees_l112_11287

theorem sin_negative_120_degrees : Real.sin (-(2 * π / 3)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_120_degrees_l112_11287


namespace NUMINAMATH_CALUDE_satisfactory_grade_fraction_l112_11286

-- Define the grade categories
inductive Grade
| A
| B
| C
| D
| F

-- Define a function to check if a grade is satisfactory
def isSatisfactory (g : Grade) : Bool :=
  match g with
  | Grade.A => true
  | Grade.B => true
  | Grade.C => true
  | _ => false

-- Define the distribution of grades
def gradeDistribution : List (Grade × Nat) :=
  [(Grade.A, 6), (Grade.B, 5), (Grade.C, 7), (Grade.D, 4), (Grade.F, 6)]

-- Theorem to prove
theorem satisfactory_grade_fraction :
  let totalStudents := (gradeDistribution.map (·.2)).sum
  let satisfactoryStudents := (gradeDistribution.filter (isSatisfactory ·.1)).map (·.2) |>.sum
  (satisfactoryStudents : Rat) / totalStudents = 9 / 14 := by
  sorry


end NUMINAMATH_CALUDE_satisfactory_grade_fraction_l112_11286


namespace NUMINAMATH_CALUDE_inequality_solution_l112_11290

theorem inequality_solution (m : ℝ) (x : ℝ) :
  (m * x - 2 ≥ 3 * x - 4 * m) ↔
  (m > 3 ∧ x ≥ (2 - 4*m) / (m - 3)) ∨
  (m < 3 ∧ x ≤ (2 - 4*m) / (m - 3)) ∨
  (m = 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l112_11290


namespace NUMINAMATH_CALUDE_only_contrapositive_correct_l112_11222

theorem only_contrapositive_correct (p q r : Prop) 
  (h : (p ∨ q) → ¬r) : 
  (¬((p ∨ q) → ¬r) ∧ 
   ¬(¬r → p) ∧ 
   ¬(r → ¬(p ∨ q)) ∧ 
   ((¬p ∧ ¬q) → r)) := by
  sorry

end NUMINAMATH_CALUDE_only_contrapositive_correct_l112_11222


namespace NUMINAMATH_CALUDE_man_crossing_bridge_l112_11220

-- Define the walking speed in km/hr
def walking_speed : ℝ := 6

-- Define the bridge length in meters
def bridge_length : ℝ := 1500

-- Define the time to cross the bridge in minutes
def crossing_time : ℝ := 15

-- Theorem statement
theorem man_crossing_bridge :
  crossing_time = bridge_length / (walking_speed * 1000 / 60) :=
by
  sorry

end NUMINAMATH_CALUDE_man_crossing_bridge_l112_11220


namespace NUMINAMATH_CALUDE_wire_cutting_l112_11292

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 100 →
  ratio = 7 / 13 →
  shorter_piece = ratio * (total_length - shorter_piece) →
  shorter_piece = 35 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l112_11292


namespace NUMINAMATH_CALUDE_complex_unit_circle_ab_range_l112_11282

theorem complex_unit_circle_ab_range (a b : ℝ) : 
  (Complex.abs (Complex.mk a b) = 1) → 
  (a * b ≥ -1/2 ∧ a * b ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_complex_unit_circle_ab_range_l112_11282


namespace NUMINAMATH_CALUDE_total_frogs_in_lakes_l112_11258

theorem total_frogs_in_lakes (lassie_frogs : ℕ) (crystal_percentage : ℚ) : 
  lassie_frogs = 45 →
  crystal_percentage = 80/100 →
  lassie_frogs + (crystal_percentage * lassie_frogs).floor = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_total_frogs_in_lakes_l112_11258


namespace NUMINAMATH_CALUDE_pen_price_payment_l112_11206

/-- Given the price of a pen and the number of pens bought, determine if the price and total payment are constants or variables -/
theorem pen_price_payment (x : ℕ) (y : ℝ) : 
  (∀ n : ℕ, 3 * n = 3 * n) ∧ (∃ m : ℕ, y ≠ 3 * m) := by
  sorry

end NUMINAMATH_CALUDE_pen_price_payment_l112_11206


namespace NUMINAMATH_CALUDE_cricket_innings_calculation_l112_11238

/-- Given a cricket player's performance data, calculate the number of innings played. -/
theorem cricket_innings_calculation (current_average : ℚ) (next_innings_runs : ℚ) (average_increase : ℚ) : 
  current_average = 35 →
  next_innings_runs = 79 →
  average_increase = 4 →
  ∃ n : ℕ, n > 0 ∧ (n : ℚ) * current_average + next_innings_runs = ((n + 1) : ℚ) * (current_average + average_increase) ∧ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_cricket_innings_calculation_l112_11238


namespace NUMINAMATH_CALUDE_negation_of_forall_quadratic_inequality_l112_11207

theorem negation_of_forall_quadratic_inequality :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_quadratic_inequality_l112_11207


namespace NUMINAMATH_CALUDE_inequality_and_floor_function_l112_11242

theorem inequality_and_floor_function (n : ℕ) : 
  (Real.sqrt (n + 1) + 2 * Real.sqrt n < Real.sqrt (9 * n + 3)) ∧ 
  ¬(∃ n : ℕ, ⌊Real.sqrt (n + 1) + 2 * Real.sqrt n⌋ < ⌊Real.sqrt (9 * n + 3)⌋) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_floor_function_l112_11242


namespace NUMINAMATH_CALUDE_complement_and_intersection_l112_11295

def U : Set ℕ := {n : ℕ | n % 2 = 0 ∧ n ≤ 10}
def A : Set ℕ := {0, 2, 4, 6}
def B : Set ℕ := {x : ℕ | x ∈ A ∧ x < 4}

theorem complement_and_intersection :
  (U \ A = {8, 10}) ∧ (A ∩ (U \ B) = {4, 6}) := by
  sorry

end NUMINAMATH_CALUDE_complement_and_intersection_l112_11295


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l112_11204

def geometric_sequence (a₁ : ℝ) (r : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => geometric_sequence a₁ r n * r

def expression (a₁ r : ℝ) : ℝ :=
  5 * (geometric_sequence a₁ r 1) + 6 * (geometric_sequence a₁ r 2)

theorem min_value_geometric_sequence :
  ∃ (min_val : ℝ), min_val = -25/12 ∧
  ∀ (r : ℝ), expression 2 r ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l112_11204


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l112_11209

theorem arithmetic_simplification :
  (4 + 6 + 4) / 3 - 4 / 3 = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l112_11209


namespace NUMINAMATH_CALUDE_k_range_l112_11280

-- Define the propositions p and q
def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := x^2 - x > 2

-- Define what it means for p to be sufficient but not necessary for q
def sufficient_not_necessary (k : ℝ) : Prop :=
  (∀ x, p x k → q x) ∧ (∃ x, q x ∧ ¬p x k)

-- Theorem statement
theorem k_range :
  ∀ k : ℝ, sufficient_not_necessary k ↔ k > 2 :=
sorry

end NUMINAMATH_CALUDE_k_range_l112_11280


namespace NUMINAMATH_CALUDE_perfect_square_condition_l112_11236

theorem perfect_square_condition (n : ℕ+) (p : ℕ) :
  (Nat.Prime p) → (∃ k : ℕ, p^2 + 7^n.val = k^2) ↔ (n = 1 ∧ p = 3) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l112_11236


namespace NUMINAMATH_CALUDE_ratio_value_l112_11271

theorem ratio_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 2 * c) 
  (h3 : c = 5 * d) 
  (h4 : b ≠ 0) 
  (h5 : d ≠ 0) : 
  (a * c) / (b * d) = 20 := by
sorry

end NUMINAMATH_CALUDE_ratio_value_l112_11271


namespace NUMINAMATH_CALUDE_quadratic_point_m_value_l112_11270

theorem quadratic_point_m_value (a m : ℝ) : 
  a > 0 → 
  m ≠ 0 → 
  3 = -a * m^2 + 2 * a * m + 3 → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_point_m_value_l112_11270


namespace NUMINAMATH_CALUDE_sequence_sum_l112_11279

/-- Given a geometric sequence a and an arithmetic sequence b,
    if 2a₃ - a₂a₄ = 0 and b₃ = a₃, then the sum of the first 5 terms of b is 10 -/
theorem sequence_sum (a b : ℕ → ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1)  -- a is geometric
  (h_arith : ∀ n : ℕ, b (n + 1) - b n = b 2 - b 1)  -- b is arithmetic
  (h_eq : 2 * a 3 - a 2 * a 4 = 0)
  (h_b3 : b 3 = a 3) :
  (b 1 + b 2 + b 3 + b 4 + b 5) = 10 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l112_11279


namespace NUMINAMATH_CALUDE_no_valid_A_l112_11231

theorem no_valid_A : ¬∃ A : ℕ, 
  0 ≤ A ∧ A ≤ 9 ∧ 
  45 % A = 0 ∧ 
  (3571 * 10 + A) * 10 + 6 % 4 = 0 ∧ 
  (3571 * 10 + A) * 10 + 6 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_A_l112_11231


namespace NUMINAMATH_CALUDE_no_perfect_square_203_base_n_l112_11275

theorem no_perfect_square_203_base_n : 
  ¬ ∃ n : ℤ, 4 ≤ n ∧ n ≤ 18 ∧ ∃ k : ℤ, 2 * n^2 + 3 = k^2 :=
sorry

end NUMINAMATH_CALUDE_no_perfect_square_203_base_n_l112_11275


namespace NUMINAMATH_CALUDE_complex_modulus_l112_11224

theorem complex_modulus (a b : ℝ) : 
  (1 + 2*Complex.I) / (Complex.mk a b) = 1 + Complex.I → 
  Complex.abs (Complex.mk a b) = Real.sqrt 10 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_l112_11224


namespace NUMINAMATH_CALUDE_tangent_slope_and_function_lower_bound_l112_11223

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * Real.exp x - a * Real.log x

theorem tangent_slope_and_function_lower_bound 
  (a : ℝ) 
  (h1 : ∀ x > 0, HasDerivAt (f a) ((2 * x + x^2) * Real.exp x - a / x) x) 
  (h2 : HasDerivAt (f a) (3 * Real.exp 1 - 1) 1) :
  a = 1 ∧ ∀ x > 0, f a x > 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_and_function_lower_bound_l112_11223


namespace NUMINAMATH_CALUDE_discount_relationship_l112_11230

/-- Represents the banker's discount in Rupees -/
def bankers_discount : ℝ := 78

/-- Represents the true discount in Rupees -/
def true_discount : ℝ := 66

/-- Represents the sum due (present value) in Rupees -/
def sum_due : ℝ := 363

/-- Theorem stating the relationship between banker's discount, true discount, and sum due -/
theorem discount_relationship : 
  bankers_discount = true_discount + (true_discount^2 / sum_due) :=
by sorry

end NUMINAMATH_CALUDE_discount_relationship_l112_11230


namespace NUMINAMATH_CALUDE_problem_statement_l112_11269

theorem problem_statement (a : ℝ) (h : (a + 1/a)^3 = 4) :
  a^4 + 1/a^4 = -158/81 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l112_11269


namespace NUMINAMATH_CALUDE_wax_already_possessed_l112_11200

/-- Given the total amount of wax needed and the additional amount required,
    calculate the amount of wax already possessed. -/
theorem wax_already_possessed
  (total_wax : ℕ)
  (additional_wax : ℕ)
  (h1 : total_wax = 288)
  (h2 : additional_wax = 260)
  : total_wax - additional_wax = 28 :=
by sorry

end NUMINAMATH_CALUDE_wax_already_possessed_l112_11200


namespace NUMINAMATH_CALUDE_weekend_to_weekday_ratio_is_two_to_one_l112_11201

/-- Represents the earnings of an Italian restaurant over a month. -/
structure RestaurantEarnings where
  weekday_earnings : ℕ  -- Daily earnings on weekdays
  total_earnings : ℕ    -- Total earnings for the month
  weeks_per_month : ℕ   -- Number of weeks in the month

/-- Calculates the ratio of weekend to weekday earnings. -/
def weekend_to_weekday_ratio (r : RestaurantEarnings) : ℚ :=
  let weekday_total := r.weekday_earnings * 5 * r.weeks_per_month
  let weekend_total := r.total_earnings - weekday_total
  let weekend_daily := weekend_total / (2 * r.weeks_per_month)
  weekend_daily / r.weekday_earnings

/-- Theorem stating that the ratio of weekend to weekday earnings is 2:1. -/
theorem weekend_to_weekday_ratio_is_two_to_one 
  (r : RestaurantEarnings) 
  (h1 : r.weekday_earnings = 600)
  (h2 : r.total_earnings = 21600)
  (h3 : r.weeks_per_month = 4) : 
  weekend_to_weekday_ratio r = 2 := by
  sorry

end NUMINAMATH_CALUDE_weekend_to_weekday_ratio_is_two_to_one_l112_11201


namespace NUMINAMATH_CALUDE_max_value_ratio_l112_11277

theorem max_value_ratio (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_ratio_l112_11277


namespace NUMINAMATH_CALUDE_cobbler_working_hours_l112_11253

/-- Represents the number of pairs of shoes a cobbler can mend in an hour -/
def shoes_per_hour : ℕ := 3

/-- Represents the number of hours the cobbler works on Friday -/
def friday_hours : ℕ := 3

/-- Represents the total number of pairs of shoes the cobbler can mend in a week -/
def total_shoes_per_week : ℕ := 105

/-- Represents the number of working days from Monday to Thursday -/
def working_days : ℕ := 4

theorem cobbler_working_hours :
  ∃ (h : ℕ), h * working_days * shoes_per_hour + friday_hours * shoes_per_hour = total_shoes_per_week ∧ h = 8 := by
  sorry

end NUMINAMATH_CALUDE_cobbler_working_hours_l112_11253


namespace NUMINAMATH_CALUDE_apple_count_l112_11213

theorem apple_count (red : ℕ) (green : ℕ) : 
  red = 16 → green = red + 12 → red + green = 44 := by
  sorry

end NUMINAMATH_CALUDE_apple_count_l112_11213


namespace NUMINAMATH_CALUDE_kite_altitude_l112_11266

theorem kite_altitude (C D K : ℝ × ℝ) (h1 : D.1 - C.1 = 15) (h2 : C.2 = D.2)
  (h3 : K.1 = C.1) (h4 : Real.tan (45 * π / 180) = (K.2 - C.2) / (K.1 - C.1))
  (h5 : Real.tan (30 * π / 180) = (K.2 - D.2) / (D.1 - K.1)) :
  K.2 - C.2 = 15 * (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_kite_altitude_l112_11266


namespace NUMINAMATH_CALUDE_square_sum_greater_than_one_l112_11221

theorem square_sum_greater_than_one
  (x y z t : ℝ)
  (h : (x^2 + y^2 - 1) * (z^2 + t^2 - 1) > (x*z + y*t - 1)^2) :
  x^2 + y^2 > 1 := by
sorry

end NUMINAMATH_CALUDE_square_sum_greater_than_one_l112_11221


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l112_11235

theorem geometric_series_first_term 
  (a r : ℝ) 
  (h1 : a / (1 - r) = 30) 
  (h2 : a^2 / (1 - r^2) = 150) : 
  a = 60 / 7 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l112_11235


namespace NUMINAMATH_CALUDE_pythagoras_field_planted_fraction_l112_11225

theorem pythagoras_field_planted_fraction :
  ∀ (a b c x : ℝ),
    a = 5 ∧ b = 12 ∧ c^2 = a^2 + b^2 →
    x > 0 →
    (a - x) * (b - x) / 2 = 1 →
    (a * b / 2 - x^2) / (a * b / 2) = 2951 / 3000 := by
  sorry

end NUMINAMATH_CALUDE_pythagoras_field_planted_fraction_l112_11225


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l112_11215

/-- A curve is an ellipse if both coefficients are positive and not equal -/
def is_ellipse (a b : ℝ) : Prop := a > 0 ∧ b > 0 ∧ a ≠ b

/-- The condition that m is between 3 and 7 -/
def m_between_3_and_7 (m : ℝ) : Prop := 3 < m ∧ m < 7

/-- The curve equation in terms of m -/
def curve_equation (m : ℝ) : Prop := is_ellipse (7 - m) (m - 3)

theorem necessary_not_sufficient :
  (∀ m : ℝ, curve_equation m → m_between_3_and_7 m) ∧
  (∃ m : ℝ, m_between_3_and_7 m ∧ ¬curve_equation m) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l112_11215


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l112_11239

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 2^p) (hy : y = 1 + 2^(-p)) : 
  y = x / (x - 1) := by
sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l112_11239


namespace NUMINAMATH_CALUDE_expected_girls_left_10_7_l112_11273

/-- The expected number of girls standing to the left of all boys in a random arrangement -/
def expected_girls_left (num_boys num_girls : ℕ) : ℚ :=
  num_girls / (num_boys + 1)

/-- Theorem: In a random arrangement of 10 boys and 7 girls, 
    the expected number of girls standing to the left of all boys is 7/11 -/
theorem expected_girls_left_10_7 : 
  expected_girls_left 10 7 = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_expected_girls_left_10_7_l112_11273


namespace NUMINAMATH_CALUDE_tyler_saltwater_animals_l112_11210

/-- The number of aquariums Tyler has -/
def num_aquariums : ℕ := 8

/-- The number of animals in each aquarium -/
def animals_per_aquarium : ℕ := 64

/-- The total number of saltwater animals Tyler has -/
def total_animals : ℕ := num_aquariums * animals_per_aquarium

/-- Theorem stating that the total number of saltwater animals Tyler has is 512 -/
theorem tyler_saltwater_animals : total_animals = 512 := by
  sorry

end NUMINAMATH_CALUDE_tyler_saltwater_animals_l112_11210


namespace NUMINAMATH_CALUDE_expression_evaluation_l112_11260

theorem expression_evaluation : 4 * (8 - 3 + 2) - 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l112_11260


namespace NUMINAMATH_CALUDE_volcano_ash_height_l112_11245

theorem volcano_ash_height (radius : ℝ) (height : ℝ) : 
  radius = 2700 → 2 * radius = 18 * height → height = 300 := by
  sorry

end NUMINAMATH_CALUDE_volcano_ash_height_l112_11245


namespace NUMINAMATH_CALUDE_total_cookies_l112_11255

theorem total_cookies (chris kenny glenn : ℕ) : 
  chris = kenny / 2 →
  glenn = 4 * kenny →
  glenn = 24 →
  chris + kenny + glenn = 33 := by
sorry

end NUMINAMATH_CALUDE_total_cookies_l112_11255


namespace NUMINAMATH_CALUDE_soccer_ball_max_height_l112_11217

/-- The height function of a soccer ball's path -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

/-- The maximum height reached by the soccer ball -/
def max_height : ℝ := 40

theorem soccer_ball_max_height :
  ∀ t : ℝ, h t ≤ max_height :=
sorry

end NUMINAMATH_CALUDE_soccer_ball_max_height_l112_11217


namespace NUMINAMATH_CALUDE_power_difference_l112_11218

theorem power_difference (a m n : ℝ) (hm : a^m = 6) (hn : a^n = 2) : a^(m-n) = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_l112_11218


namespace NUMINAMATH_CALUDE_mean_median_difference_l112_11247

/-- Represents the score distribution of a math test -/
structure ScoreDistribution where
  score60 : Float
  score75 : Float
  score85 : Float
  score90 : Float
  score100 : Float
  sum_to_one : score60 + score75 + score85 + score90 + score100 = 1

/-- Calculates the mean score given a ScoreDistribution -/
def meanScore (d : ScoreDistribution) : Float :=
  60 * d.score60 + 75 * d.score75 + 85 * d.score85 + 90 * d.score90 + 100 * d.score100

/-- Calculates the median score given a ScoreDistribution -/
def medianScore (d : ScoreDistribution) : Float :=
  if d.score60 + d.score75 > 0.5 then 75
  else if d.score60 + d.score75 + d.score85 > 0.5 then 85
  else 90

/-- The main theorem stating the difference between mean and median -/
theorem mean_median_difference (d : ScoreDistribution) 
  (h1 : d.score60 = 0.15)
  (h2 : d.score75 = 0.20)
  (h3 : d.score85 = 0.25)
  (h4 : d.score90 = 0.25) :
  medianScore d - meanScore d = 2.25 := by
  sorry


end NUMINAMATH_CALUDE_mean_median_difference_l112_11247


namespace NUMINAMATH_CALUDE_no_solution_iff_m_geq_three_l112_11227

theorem no_solution_iff_m_geq_three (m : ℝ) :
  (∀ x : ℝ, ¬(x - m ≥ 0 ∧ (1/2) * x + (1/2) < 2)) ↔ m ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_geq_three_l112_11227


namespace NUMINAMATH_CALUDE_pauls_score_l112_11254

theorem pauls_score (total_points cousin_points : ℕ) 
  (h1 : total_points = 5816)
  (h2 : cousin_points = 2713) :
  total_points - cousin_points = 3103 := by
  sorry

end NUMINAMATH_CALUDE_pauls_score_l112_11254


namespace NUMINAMATH_CALUDE_solution_to_system_l112_11289

theorem solution_to_system : 
  ∀ x y : ℝ, 
  3 * x^2 - 9 * y^2 = 0 → 
  x + y = 5 → 
  ((x = (15 - 5 * Real.sqrt 3) / 2 ∧ y = (5 * Real.sqrt 3 - 5) / 2) ∨
   (x = (-15 - 5 * Real.sqrt 3) / 2 ∧ y = (5 + 5 * Real.sqrt 3) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_to_system_l112_11289


namespace NUMINAMATH_CALUDE_orange_juice_production_l112_11276

/-- The amount of oranges (in million tons) used for juice production -/
def juice_production (total : ℝ) (export_percent : ℝ) (juice_percent : ℝ) : ℝ :=
  total * (1 - export_percent) * juice_percent

/-- Theorem stating the amount of oranges used for juice production -/
theorem orange_juice_production :
  let total := 8
  let export_percent := 0.25
  let juice_percent := 0.60
  juice_production total export_percent juice_percent = 3.6 := by
sorry

#eval juice_production 8 0.25 0.60

end NUMINAMATH_CALUDE_orange_juice_production_l112_11276


namespace NUMINAMATH_CALUDE_sequence_limit_inequality_l112_11261

theorem sequence_limit_inequality (a b : ℕ → ℝ) (A B : ℝ) :
  (∀ n : ℕ, a n > b n) →
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - A| < ε) →
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |b n - B| < ε) →
  A ≥ B := by
  sorry

end NUMINAMATH_CALUDE_sequence_limit_inequality_l112_11261


namespace NUMINAMATH_CALUDE_pyramid_volume_l112_11212

/-- Represents a pyramid with a square base ABCD and vertex E -/
structure Pyramid where
  base_area : ℝ
  triangle_ABE_area : ℝ
  triangle_CDE_area : ℝ

/-- The volume of the pyramid is 64√15 -/
theorem pyramid_volume (p : Pyramid) 
  (h1 : p.base_area = 64)
  (h2 : p.triangle_ABE_area = 48)
  (h3 : p.triangle_CDE_area = 40) :
  ∃ (v : ℝ), v = 64 * Real.sqrt 15 ∧ v = (1/3) * p.base_area * Real.sqrt ((2 * p.triangle_ABE_area / Real.sqrt p.base_area)^2 - (Real.sqrt p.base_area - 2 * p.triangle_CDE_area / Real.sqrt p.base_area)^2) := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l112_11212


namespace NUMINAMATH_CALUDE_smallest_b_not_prime_nine_satisfies_condition_nine_is_smallest_l112_11229

theorem smallest_b_not_prime (b : ℕ) (h : b > 8) :
  (∀ x : ℤ, ¬ Prime (x^4 + b^2 : ℤ)) →
  b ≥ 9 :=
by sorry

theorem nine_satisfies_condition :
  ∀ x : ℤ, ¬ Prime (x^4 + 9^2 : ℤ) :=
by sorry

theorem nine_is_smallest :
  ∀ b : ℕ, b > 8 →
  (∀ x : ℤ, ¬ Prime (x^4 + b^2 : ℤ)) →
  b ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_not_prime_nine_satisfies_condition_nine_is_smallest_l112_11229


namespace NUMINAMATH_CALUDE_rectangle_45_odd_intersections_impossible_l112_11296

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a point is on a grid line -/
def isOnGridLine (p : Point) : Prop :=
  ∃ n : ℤ, p.x = n ∨ p.y = n

/-- Checks if two line segments are at 45° angle to each other -/
def isAt45Degree (p1 p2 q1 q2 : Point) : Prop :=
  |p2.x - p1.x| = |p2.y - p1.y| ∧ |q2.x - q1.x| = |q2.y - q1.y|

/-- Counts the number of grid line intersections for a line segment -/
def gridIntersections (p1 p2 : Point) : ℕ :=
  sorry

/-- Main theorem: It's impossible for all sides of a 45° rectangle to intersect an odd number of grid lines -/
theorem rectangle_45_odd_intersections_impossible (rect : Rectangle) :
  (¬ isOnGridLine rect.A) →
  (¬ isOnGridLine rect.B) →
  (¬ isOnGridLine rect.C) →
  (¬ isOnGridLine rect.D) →
  isAt45Degree rect.A rect.B rect.B rect.C →
  ¬ (Odd (gridIntersections rect.A rect.B) ∧
     Odd (gridIntersections rect.B rect.C) ∧
     Odd (gridIntersections rect.C rect.D) ∧
     Odd (gridIntersections rect.D rect.A)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_45_odd_intersections_impossible_l112_11296


namespace NUMINAMATH_CALUDE_prob_one_six_max_l112_11251

/-- The probability of rolling exactly one six when rolling n dice -/
def prob_one_six (n : ℕ) : ℚ :=
  (n : ℚ) * (5 ^ (n - 1) : ℚ) / (6 ^ n : ℚ)

/-- The statement that the probability of rolling exactly one six is maximized for 5 or 6 dice -/
theorem prob_one_six_max :
  (∀ k : ℕ, prob_one_six k ≤ prob_one_six 5) ∧
  (prob_one_six 5 = prob_one_six 6) ∧
  (∀ k : ℕ, k > 6 → prob_one_six k < prob_one_six 6) :=
sorry

end NUMINAMATH_CALUDE_prob_one_six_max_l112_11251


namespace NUMINAMATH_CALUDE_prob_ride_each_car_once_two_cars_two_rides_l112_11293

/-- Represents a roller coaster with a given number of cars. -/
structure RollerCoaster where
  num_cars : ℕ

/-- Represents a passenger's ride on the roller coaster. -/
structure Ride where
  coaster : RollerCoaster
  num_rides : ℕ

/-- The probability of a passenger riding in each car exactly once. -/
def prob_ride_each_car_once (r : Ride) : ℚ :=
  sorry

/-- Theorem stating the probability of riding in each car once for a 2-car coaster with 2 rides. -/
theorem prob_ride_each_car_once_two_cars_two_rides :
  ∀ (r : Ride), r.coaster.num_cars = 2 → r.num_rides = 2 →
  prob_ride_each_car_once r = 1/2 :=
sorry

end NUMINAMATH_CALUDE_prob_ride_each_car_once_two_cars_two_rides_l112_11293


namespace NUMINAMATH_CALUDE_min_value_theorem_l112_11259

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 2/b + 3/c = 2) :
  a + 2*b + 3*c ≥ 18 ∧ (a + 2*b + 3*c = 18 ↔ a = 3 ∧ b = 3 ∧ c = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l112_11259


namespace NUMINAMATH_CALUDE_hydrogen_mass_percentage_l112_11219

/-- Molecular weight of C3H6O in g/mol -/
def mw_C3H6O : ℝ := 58.09

/-- Molecular weight of NH3 in g/mol -/
def mw_NH3 : ℝ := 17.04

/-- Molecular weight of H2SO4 in g/mol -/
def mw_H2SO4 : ℝ := 98.09

/-- Mass of hydrogen in one mole of C3H6O in g -/
def mass_H_in_C3H6O : ℝ := 6.06

/-- Mass of hydrogen in one mole of NH3 in g -/
def mass_H_in_NH3 : ℝ := 3.03

/-- Mass of hydrogen in one mole of H2SO4 in g -/
def mass_H_in_H2SO4 : ℝ := 2.02

/-- Number of moles of C3H6O in the mixture -/
def moles_C3H6O : ℝ := 3

/-- Number of moles of NH3 in the mixture -/
def moles_NH3 : ℝ := 2

/-- Number of moles of H2SO4 in the mixture -/
def moles_H2SO4 : ℝ := 1

/-- Theorem stating that the mass percentage of hydrogen in the given mixture is approximately 8.57% -/
theorem hydrogen_mass_percentage :
  let total_mass_H := moles_C3H6O * mass_H_in_C3H6O + moles_NH3 * mass_H_in_NH3 + moles_H2SO4 * mass_H_in_H2SO4
  let total_mass_mixture := moles_C3H6O * mw_C3H6O + moles_NH3 * mw_NH3 + moles_H2SO4 * mw_H2SO4
  let mass_percentage_H := (total_mass_H / total_mass_mixture) * 100
  ∃ ε > 0, |mass_percentage_H - 8.57| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_hydrogen_mass_percentage_l112_11219


namespace NUMINAMATH_CALUDE_exists_coprime_in_ten_consecutive_integers_l112_11205

theorem exists_coprime_in_ten_consecutive_integers (n : ℤ) :
  ∃ k ∈ Finset.range 10, ∀ m ∈ Finset.range 10, m ≠ k → Int.gcd (n + k) (n + m) = 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_coprime_in_ten_consecutive_integers_l112_11205


namespace NUMINAMATH_CALUDE_max_m_value_l112_11249

theorem max_m_value (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → (4 / (1 - x)) ≥ m - (1 / x)) → 
  m ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_max_m_value_l112_11249


namespace NUMINAMATH_CALUDE_pet_store_birds_pet_store_birds_solution_l112_11233

/-- Proves that the initial number of birds in the pet store was 12 -/
theorem pet_store_birds : ℕ → Prop :=
  fun initial_birds =>
    let initial_puppies : ℕ := 9
    let initial_cats : ℕ := 5
    let initial_spiders : ℕ := 15
    let remaining_birds : ℕ := initial_birds / 2
    let remaining_puppies : ℕ := initial_puppies - 3
    let remaining_cats : ℕ := initial_cats
    let remaining_spiders : ℕ := initial_spiders - 7
    let total_remaining : ℕ := remaining_birds + remaining_puppies + remaining_cats + remaining_spiders
    total_remaining = 25 → initial_birds = 12

/-- The theorem holds for 12 initial birds -/
theorem pet_store_birds_solution : pet_store_birds 12 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_pet_store_birds_solution_l112_11233


namespace NUMINAMATH_CALUDE_last_positive_term_is_six_l112_11294

/-- Represents an arithmetic sequence with a given start and common difference. -/
structure ArithmeticSequence where
  start : ℤ
  diff : ℤ

/-- Calculates the nth term of an arithmetic sequence. -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.start + (n - 1 : ℤ) * seq.diff

/-- Theorem: The last term greater than 0 in the sequence (72, 61, 50, ...) is 6. -/
theorem last_positive_term_is_six :
  let seq := ArithmeticSequence.mk 72 (-11)
  ∃ n : ℕ, 
    (nthTerm seq n = 6) ∧ 
    (nthTerm seq n > 0) ∧ 
    (nthTerm seq (n + 1) ≤ 0) :=
by sorry

#check last_positive_term_is_six

end NUMINAMATH_CALUDE_last_positive_term_is_six_l112_11294


namespace NUMINAMATH_CALUDE_equation_solution_l112_11284

theorem equation_solution (a b x : ℤ) : 
  (a * x^2 + b * x + 14)^2 + (b * x^2 + a * x + 8)^2 = 0 →
  a = -6 ∧ b = -5 ∧ x = -2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l112_11284


namespace NUMINAMATH_CALUDE_problem_solution_l112_11299

theorem problem_solution (x y : ℝ) : 
  (65 / 100 : ℝ) * 900 = (40 / 100 : ℝ) * x → 
  (35 / 100 : ℝ) * 1200 = (25 / 100 : ℝ) * y → 
  x + y = 3142.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l112_11299


namespace NUMINAMATH_CALUDE_factorization_equalities_l112_11214

theorem factorization_equalities (a b x y : ℝ) : 
  (3 * a * x^2 + 6 * a * x * y + 3 * a * y^2 = 3 * a * (x + y)^2) ∧
  (a^2 * (x - y) - b^2 * (x - y) = (x - y) * (a + b) * (a - b)) ∧
  (a^4 + 3 * a^2 - 4 = (a + 1) * (a - 1) * (a^2 + 4)) ∧
  (4 * x^2 - y^2 - 2 * y - 1 = (2 * x + y + 1) * (2 * x - y - 1)) :=
by sorry

end NUMINAMATH_CALUDE_factorization_equalities_l112_11214


namespace NUMINAMATH_CALUDE_total_time_is_14_25_years_l112_11274

def time_to_get_in_shape : ℕ := 2 * 12  -- 2 years in months
def time_to_learn_climbing : ℕ := 2 * time_to_get_in_shape
def time_for_survival_skills : ℕ := 9
def time_for_photography : ℕ := 3
def downtime : ℕ := 1
def time_for_summits : List ℕ := [4, 5, 6, 8, 7, 9, 10]
def time_to_learn_diving : ℕ := 13
def time_for_cave_diving : ℕ := 2 * 12  -- 2 years in months

theorem total_time_is_14_25_years :
  let total_months : ℕ := time_to_get_in_shape + time_to_learn_climbing +
                          time_for_survival_skills + time_for_photography +
                          downtime + (time_for_summits.sum) +
                          time_to_learn_diving + time_for_cave_diving
  (total_months : ℚ) / 12 = 14.25 := by
  sorry

end NUMINAMATH_CALUDE_total_time_is_14_25_years_l112_11274


namespace NUMINAMATH_CALUDE_total_count_is_six_l112_11226

def problem (total_count : ℕ) (group1_count group2_count group3_count : ℕ) 
  (total_avg group1_avg group2_avg group3_avg : ℚ) : Prop :=
  total_count = group1_count + group2_count + group3_count ∧
  group1_count = 2 ∧
  group2_count = 2 ∧
  group3_count = 2 ∧
  total_avg = 3.95 ∧
  group1_avg = 3.6 ∧
  group2_avg = 3.85 ∧
  group3_avg = 4.400000000000001

theorem total_count_is_six :
  ∃ (total_count : ℕ) (group1_count group2_count group3_count : ℕ)
    (total_avg group1_avg group2_avg group3_avg : ℚ),
  problem total_count group1_count group2_count group3_count
    total_avg group1_avg group2_avg group3_avg ∧
  total_count = 6 :=
by sorry

end NUMINAMATH_CALUDE_total_count_is_six_l112_11226


namespace NUMINAMATH_CALUDE_parallel_line_slope_l112_11228

/-- Given a line parallel to 3x - 6y = 21, its slope is 1/2 -/
theorem parallel_line_slope (a b c : ℝ) (h : a * x - b * y = c) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ (a, b, c) = (3 * k, 6 * k, 21 * k)) :
  (a / b : ℝ) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l112_11228


namespace NUMINAMATH_CALUDE_athlete_weights_problem_l112_11256

theorem athlete_weights_problem (a b c : ℝ) (k₁ k₂ k₃ : ℤ) : 
  (a + b) / 2 = 40 →
  (b + c) / 2 = 43 →
  (a + c) / 2 = 44 →
  a + b = 5 * k₁ →
  b + c = 5 * k₂ →
  a + c = 5 * k₃ →
  b = 40 := by
  sorry

end NUMINAMATH_CALUDE_athlete_weights_problem_l112_11256


namespace NUMINAMATH_CALUDE_fundraising_goal_exceeded_l112_11246

theorem fundraising_goal_exceeded (goal ken_amount : ℕ) 
  (h1 : ken_amount = 600)
  (h2 : goal = 4000) : 
  let mary_amount := 5 * ken_amount
  let scott_amount := mary_amount / 3
  ken_amount + mary_amount + scott_amount - goal = 600 := by
sorry

end NUMINAMATH_CALUDE_fundraising_goal_exceeded_l112_11246


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l112_11268

theorem value_of_a_minus_b (a b : ℤ) 
  (eq1 : 2020 * a + 2024 * b = 2040)
  (eq2 : 2022 * a + 2026 * b = 2044) : 
  a - b = 1002 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l112_11268


namespace NUMINAMATH_CALUDE_area_of_CALI_l112_11278

/-- Square BERK with side length 10 -/
def BERK : Set (ℝ × ℝ) := sorry

/-- Points T, O, W, N as midpoints of BE, ER, RK, KB respectively -/
def T : ℝ × ℝ := sorry
def O : ℝ × ℝ := sorry
def W : ℝ × ℝ := sorry
def N : ℝ × ℝ := sorry

/-- Square CALI whose edges contain vertices of BERK -/
def CALI : Set (ℝ × ℝ) := sorry

/-- CA is parallel to BO -/
def CA_parallel_BO : Prop := sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

theorem area_of_CALI : area CALI = 180 :=
sorry

end NUMINAMATH_CALUDE_area_of_CALI_l112_11278


namespace NUMINAMATH_CALUDE_chinese_chess_draw_probability_l112_11250

/-- The probability of A and B drawing in Chinese chess -/
theorem chinese_chess_draw_probability 
  (p_a_not_lose : ℝ) 
  (p_b_not_lose : ℝ) 
  (h1 : p_a_not_lose = 0.8) 
  (h2 : p_b_not_lose = 0.7) : 
  ∃ (p_draw : ℝ), p_draw = 0.5 ∧ p_a_not_lose = (1 - p_b_not_lose) + p_draw :=
sorry

end NUMINAMATH_CALUDE_chinese_chess_draw_probability_l112_11250


namespace NUMINAMATH_CALUDE_rectangle_perimeter_area_equality_l112_11283

theorem rectangle_perimeter_area_equality (k : ℝ) (h : k > 0) :
  (∃ w : ℝ, w > 0 ∧ 
    8 * w = k ∧  -- Perimeter equals k
    3 * w^2 = k) -- Area equals k
  → k = 64 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_area_equality_l112_11283


namespace NUMINAMATH_CALUDE_profit_share_difference_l112_11208

/-- Represents the initial capital and interest rate for each partner --/
structure Partner where
  capital : ℕ
  rate : ℚ

/-- Calculates the interest earned by a partner --/
def interest (p : Partner) : ℚ := p.capital * p.rate

/-- Calculates the profit share of a partner --/
def profitShare (p : Partner) (totalProfit : ℕ) : ℚ :=
  p.capital + interest p

theorem profit_share_difference
  (a b c : Partner)
  (ha : a.capital = 8000 ∧ a.rate = 5/100)
  (hb : b.capital = 10000 ∧ b.rate = 6/100)
  (hc : c.capital = 12000 ∧ c.rate = 7/100)
  (totalProfit : ℕ)
  (hProfit : profitShare b totalProfit = 13600) :
  profitShare c totalProfit - profitShare a totalProfit = 4440 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_difference_l112_11208


namespace NUMINAMATH_CALUDE_average_weight_proof_l112_11265

theorem average_weight_proof (total_boys : Nat) (group1_boys : Nat) (group2_boys : Nat)
  (group2_avg_weight : ℝ) (total_avg_weight : ℝ) (group1_avg_weight : ℝ) :
  total_boys = 30 →
  group1_boys = 22 →
  group2_boys = 8 →
  group2_avg_weight = 45.15 →
  total_avg_weight = 48.89 →
  group1_avg_weight = 50.25 →
  (group1_boys : ℝ) * group1_avg_weight + (group2_boys : ℝ) * group2_avg_weight =
    (total_boys : ℝ) * total_avg_weight :=
by sorry

end NUMINAMATH_CALUDE_average_weight_proof_l112_11265


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_composite_l112_11243

theorem quadratic_roots_imply_composite (a b : ℤ) :
  (∃ x₁ x₂ : ℕ+, x₁^2 + a * x₁ + b + 1 = 0 ∧ x₂^2 + a * x₂ + b + 1 = 0) →
  ∃ m n : ℕ, m > 1 ∧ n > 1 ∧ a^2 + b^2 = m * n :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_composite_l112_11243


namespace NUMINAMATH_CALUDE_geometric_figure_pieces_l112_11262

/-- Calculates the sum of the first n natural numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the number of rows in the geometric figure -/
def num_rows : ℕ := 10

/-- Calculates the number of rods in the geometric figure -/
def num_rods : ℕ := 3 * triangular_number num_rows

/-- Calculates the number of connectors in the geometric figure -/
def num_connectors : ℕ := triangular_number (num_rows + 1)

/-- Calculates the number of unit squares in the geometric figure -/
def num_squares : ℕ := triangular_number num_rows

/-- The total number of pieces in the geometric figure -/
def total_pieces : ℕ := num_rods + num_connectors + num_squares

theorem geometric_figure_pieces :
  total_pieces = 286 :=
sorry

end NUMINAMATH_CALUDE_geometric_figure_pieces_l112_11262


namespace NUMINAMATH_CALUDE_tan_double_angle_special_case_l112_11234

theorem tan_double_angle_special_case (θ : ℝ) :
  3 * Real.cos (π / 2 - θ) + Real.cos (π + θ) = 0 →
  Real.tan (2 * θ) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_case_l112_11234


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l112_11298

/-- Given a function f(x) = (x+a)e^x that satisfies f(x) ≥ (1/6)x^3 - x - 2 for all x ∈ ℝ,
    prove that a ≥ -2. -/
theorem function_inequality_implies_parameter_bound (a : ℝ) :
  (∀ x : ℝ, (x + a) * Real.exp x ≥ (1/6) * x^3 - x - 2) →
  a ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l112_11298


namespace NUMINAMATH_CALUDE_sandwich_combinations_l112_11288

theorem sandwich_combinations (num_meats num_cheeses : ℕ) : 
  num_meats = 12 → num_cheeses = 11 → 
  (num_meats * num_cheeses) + (num_meats * (num_cheeses.choose 2)) = 792 := by
  sorry

#check sandwich_combinations

end NUMINAMATH_CALUDE_sandwich_combinations_l112_11288


namespace NUMINAMATH_CALUDE_triangle_formation_and_acuteness_l112_11257

theorem triangle_formation_and_acuteness 
  (a b c : ℝ) (n k : ℕ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_n_ge_2 : n ≥ 2) (h_k_lt_n : k < n) 
  (h_sum_powers : a^n + b^n = c^n) : 
  (a^k + b^k > c^k) ∧ 
  (a^k^2 + b^k^2 > c^k^2 ↔ k < n / 2) := by
sorry

end NUMINAMATH_CALUDE_triangle_formation_and_acuteness_l112_11257


namespace NUMINAMATH_CALUDE_regular_polygon_120_degrees_l112_11267

/-- A regular polygon with interior angles of 120° has 6 sides -/
theorem regular_polygon_120_degrees (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 / n = 120) → 
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_120_degrees_l112_11267


namespace NUMINAMATH_CALUDE_expression_simplification_l112_11237

theorem expression_simplification (α : ℝ) :
  4.59 * (Real.cos (2*α) - Real.cos (6*α) + Real.cos (10*α) - Real.cos (14*α)) /
  (Real.sin (2*α) + Real.sin (6*α) + Real.sin (10*α) + Real.sin (14*α)) =
  Real.tan (2*α) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l112_11237


namespace NUMINAMATH_CALUDE_equal_derivative_points_l112_11203

theorem equal_derivative_points (x₀ : ℝ) : 
  (2 * x₀ = -3 * x₀^2) → (x₀ = 0 ∨ x₀ = -2/3) :=
by sorry

end NUMINAMATH_CALUDE_equal_derivative_points_l112_11203


namespace NUMINAMATH_CALUDE_divisibility_by_37_l112_11263

theorem divisibility_by_37 : ∃ k : ℤ, 333^555 + 555^333 = 37 * k := by sorry

end NUMINAMATH_CALUDE_divisibility_by_37_l112_11263


namespace NUMINAMATH_CALUDE_clock_chimes_theorem_l112_11216

/-- Represents the number of chimes at a given hour -/
def chimes_at_hour (hour : ℕ) : ℕ := hour

/-- Represents the time taken for a given number of chimes -/
def time_for_chimes (chimes : ℕ) : ℕ :=
  if chimes ≤ 1 then chimes else chimes - 1 + 1

/-- The theorem statement -/
theorem clock_chimes_theorem (hour : ℕ) (chimes : ℕ) (time : ℕ) 
  (h1 : hour = 2 → chimes = 2)
  (h2 : hour = 2 → time = 2)
  (h3 : hour = 12 → chimes = 12) :
  hour = 12 → time_for_chimes chimes = 12 := by
  sorry

#check clock_chimes_theorem

end NUMINAMATH_CALUDE_clock_chimes_theorem_l112_11216


namespace NUMINAMATH_CALUDE_arc_length_45_degrees_l112_11285

/-- Given a circle with circumference 90 meters, prove that an arc subtending a 45° angle at the center has a length of 11.25 meters. -/
theorem arc_length_45_degrees (D : Real) (EF : Real) :
  D = 90 →  -- circumference of the circle
  EF = (45 / 360) * D →  -- arc length is proportional to the angle it subtends
  EF = 11.25 := by
sorry

end NUMINAMATH_CALUDE_arc_length_45_degrees_l112_11285


namespace NUMINAMATH_CALUDE_completing_square_result_l112_11252

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 2*x - 2 = 0

-- Define the completed square form
def completed_square (x n : ℝ) : Prop := (x - 1)^2 = n

-- Theorem statement
theorem completing_square_result : 
  ∃ n : ℝ, (∀ x : ℝ, quadratic_equation x ↔ completed_square x n) ∧ n = 3 :=
sorry

end NUMINAMATH_CALUDE_completing_square_result_l112_11252


namespace NUMINAMATH_CALUDE_k_equals_p_l112_11241

theorem k_equals_p (k p : ℕ) : 
  (∃ (nums_k : Finset ℕ) (nums_p : Finset ℕ), 
    (Finset.card nums_k = k) ∧ 
    (Finset.card nums_p = p) ∧
    (∀ x ∈ nums_k, x = 2*p + 3) ∧
    (∀ y ∈ nums_p, y = 5 - 2*k) ∧
    ((Finset.sum nums_k id + Finset.sum nums_p id) / (k + p : ℝ) = 4)) →
  k = p :=
sorry

end NUMINAMATH_CALUDE_k_equals_p_l112_11241


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_l112_11291

theorem quadrilateral_inequality (a b c : ℝ) : 
  (a > 0) →  -- EF has positive length
  (b > 0) →  -- EG has positive length
  (c > 0) →  -- EH has positive length
  (a < b) →  -- F is between E and G
  (b < c) →  -- G is between E and H
  (2 * b > c) →  -- Condition for positive area after rotation
  (a < c / 3) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_l112_11291


namespace NUMINAMATH_CALUDE_sum_of_factors_180_l112_11272

/-- The sum of positive factors of a natural number n -/
def sum_of_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of positive factors of 180 is 546 -/
theorem sum_of_factors_180 : sum_of_factors 180 = 546 := by sorry

end NUMINAMATH_CALUDE_sum_of_factors_180_l112_11272


namespace NUMINAMATH_CALUDE_max_sum_of_goods_l112_11281

theorem max_sum_of_goods (a b : ℕ) : 
  a > 0 → b > 0 → 5 * a + 19 * b = 213 → (∀ x y : ℕ, x > 0 → y > 0 → 5 * x + 19 * y = 213 → a + b ≥ x + y) → a + b = 37 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_goods_l112_11281


namespace NUMINAMATH_CALUDE_inequality_proof_l112_11244

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a > b) :
  1 / (a * b^2) > 1 / (a^2 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l112_11244


namespace NUMINAMATH_CALUDE_number_of_women_l112_11248

-- Define the total number of family members
def total_members : ℕ := 15

-- Define the time it takes for a woman to complete the work
def woman_work_days : ℕ := 180

-- Define the time it takes for a man to complete the work
def man_work_days : ℕ := 120

-- Define the time it takes to complete the work with alternating schedule
def alternating_work_days : ℕ := 17

-- Define the function to calculate the number of women
def calculate_women (total : ℕ) (woman_days : ℕ) (man_days : ℕ) (alt_days : ℕ) : ℕ :=
  sorry

-- Theorem statement
theorem number_of_women : 
  calculate_women total_members woman_work_days man_work_days alternating_work_days = 3 :=
sorry

end NUMINAMATH_CALUDE_number_of_women_l112_11248


namespace NUMINAMATH_CALUDE_elena_earnings_l112_11297

def charging_sequence : List Nat := [3, 4, 5, 6, 7]

def calculate_earnings (hours : Nat) : Nat :=
  let complete_cycles := hours / 5
  let remaining_hours := hours % 5
  let cycle_earnings := charging_sequence.sum * complete_cycles
  let remaining_earnings := (charging_sequence.take remaining_hours).sum
  cycle_earnings + remaining_earnings

theorem elena_earnings :
  calculate_earnings 47 = 232 := by
  sorry

end NUMINAMATH_CALUDE_elena_earnings_l112_11297


namespace NUMINAMATH_CALUDE_commute_speed_l112_11232

theorem commute_speed (v : ℝ) (h1 : v > 0) : 
  (18 / v + 18 / 30 = 1) → v = 45 := by
  sorry

end NUMINAMATH_CALUDE_commute_speed_l112_11232
