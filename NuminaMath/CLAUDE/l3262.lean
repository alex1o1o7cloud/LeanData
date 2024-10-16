import Mathlib

namespace NUMINAMATH_CALUDE_race_distance_l3262_326224

theorem race_distance (a_time b_time lead_distance : ℕ) 
  (ha : a_time = 28)
  (hb : b_time = 32)
  (hl : lead_distance = 28) : 
  (b_time * lead_distance) / (b_time - a_time) = 224 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l3262_326224


namespace NUMINAMATH_CALUDE_circle_line_intersection_l3262_326268

/-- Circle C: x^2 + y^2 - 2x + 2y - 4 = 0 -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 2*y - 4 = 0

/-- Line l: y = x + b with slope 1 -/
def Line (x y b : ℝ) : Prop := y = x + b

/-- Intersection points of Circle and Line -/
def Intersection (b : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂ : ℝ, 
  x₁ ≠ x₂ ∧ Circle x₁ y₁ ∧ Circle x₂ y₂ ∧ Line x₁ y₁ b ∧ Line x₂ y₂ b

/-- The circle with diameter AB passes through the origin -/
def CircleThroughOrigin (b : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂ : ℝ,
  Circle x₁ y₁ ∧ Circle x₂ y₂ ∧ Line x₁ y₁ b ∧ Line x₂ y₂ b ∧
  x₁*x₂ + y₁*y₂ = 0

theorem circle_line_intersection :
  (∀ b : ℝ, Intersection b ↔ -3-3*Real.sqrt 2 < b ∧ b < -3+3*Real.sqrt 2) ∧
  (∃! b₁ b₂ : ℝ, b₁ ≠ b₂ ∧ CircleThroughOrigin b₁ ∧ CircleThroughOrigin b₂ ∧
    b₁ = 1 ∧ b₂ = -4) :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3262_326268


namespace NUMINAMATH_CALUDE_bike_to_tractor_speed_ratio_l3262_326255

/-- Prove that the ratio of bike speed to tractor speed is 2:1 -/
theorem bike_to_tractor_speed_ratio :
  let tractor_speed := 575 / 23
  let car_speed := 360 / 4
  let bike_speed := car_speed / (9/5)
  bike_speed / tractor_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_bike_to_tractor_speed_ratio_l3262_326255


namespace NUMINAMATH_CALUDE_banana_pile_count_l3262_326264

/-- The total number of bananas in a pile after adding more bananas -/
def total_bananas (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: Given 2 initial bananas and 7 added bananas, the total is 9 -/
theorem banana_pile_count : total_bananas 2 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_banana_pile_count_l3262_326264


namespace NUMINAMATH_CALUDE_your_bill_before_tax_friend_order_equation_l3262_326213

/-- The cost of a taco in dollars -/
def taco_cost : ℝ := sorry

/-- The cost of an enchilada in dollars -/
def enchilada_cost : ℝ := 2

/-- The cost of 3 tacos and 5 enchiladas in dollars -/
def friend_order_cost : ℝ := 12.70

theorem your_bill_before_tax :
  2 * taco_cost + 3 * enchilada_cost = 7.80 :=
by
  sorry

/-- The friend's order cost equation -/
theorem friend_order_equation :
  3 * taco_cost + 5 * enchilada_cost = friend_order_cost :=
by
  sorry

end NUMINAMATH_CALUDE_your_bill_before_tax_friend_order_equation_l3262_326213


namespace NUMINAMATH_CALUDE_girls_average_height_l3262_326201

/-- Calculates the average height of female students in a class -/
def average_height_girls (total_students : ℕ) (boys : ℕ) (avg_height_all : ℚ) (avg_height_boys : ℚ) : ℚ :=
  let girls := total_students - boys
  let total_height := (total_students : ℚ) * avg_height_all
  let boys_height := (boys : ℚ) * avg_height_boys
  let girls_height := total_height - boys_height
  girls_height / (girls : ℚ)

/-- Theorem stating the average height of girls in the class -/
theorem girls_average_height :
  average_height_girls 30 18 140 144 = 134 := by
  sorry

end NUMINAMATH_CALUDE_girls_average_height_l3262_326201


namespace NUMINAMATH_CALUDE_mashed_potatoes_vs_bacon_difference_l3262_326218

/-- The number of students who suggested adding bacon to the menu. -/
def bacon_students : ℕ := 269

/-- The number of students who suggested adding mashed potatoes to the menu. -/
def mashed_potatoes_students : ℕ := 330

/-- The number of students who suggested adding tomatoes to the menu. -/
def tomato_students : ℕ := 76

/-- The theorem states that the difference between the number of students who suggested
    mashed potatoes and the number of students who suggested bacon is 61. -/
theorem mashed_potatoes_vs_bacon_difference :
  mashed_potatoes_students - bacon_students = 61 := by
  sorry

end NUMINAMATH_CALUDE_mashed_potatoes_vs_bacon_difference_l3262_326218


namespace NUMINAMATH_CALUDE_triangle_inradius_l3262_326290

/-- Given a triangle with perimeter 32 cm and area 40 cm², prove that its inradius is 2.5 cm. -/
theorem triangle_inradius (P : ℝ) (A : ℝ) (r : ℝ) 
  (h_perimeter : P = 32) 
  (h_area : A = 40) 
  (h_inradius : A = r * (P / 2)) : 
  r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l3262_326290


namespace NUMINAMATH_CALUDE_at_most_one_perfect_square_l3262_326295

def sequence_a : ℕ → ℕ
  | 0 => 1  -- arbitrary starting value
  | n + 1 => (sequence_a n)^3 + 103

theorem at_most_one_perfect_square :
  ∃ k : ℕ, ∀ n m : ℕ, 
    (∃ i : ℕ, sequence_a n = i^2) → 
    (∃ j : ℕ, sequence_a m = j^2) → 
    n = m ∨ (n < k ∧ m < k) := by
  sorry

end NUMINAMATH_CALUDE_at_most_one_perfect_square_l3262_326295


namespace NUMINAMATH_CALUDE_ab_in_terms_of_m_and_n_l3262_326230

theorem ab_in_terms_of_m_and_n (a b m n : ℝ) 
  (h1 : (a + b)^2 = m) 
  (h2 : (a - b)^2 = n) : 
  a * b = (m - n) / 4 := by
sorry

end NUMINAMATH_CALUDE_ab_in_terms_of_m_and_n_l3262_326230


namespace NUMINAMATH_CALUDE_range_of_a_l3262_326297

/-- Given a function f(x) = x^2 + 2(a-1)x + 2 that is monotonically decreasing on (-∞, 4],
    prove that the range of a is (-∞, -3]. -/
theorem range_of_a (a : ℝ) : 
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → (x^2 + 2*(a-1)*x + 2) > (y^2 + 2*(a-1)*y + 2)) →
  a ∈ Set.Iic (-3 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3262_326297


namespace NUMINAMATH_CALUDE_mrs_flannery_muffins_count_l3262_326249

/-- The number of muffins baked by Mrs. Brier's class -/
def mrs_brier_muffins : ℕ := 18

/-- The number of muffins baked by Mrs. MacAdams's class -/
def mrs_macadams_muffins : ℕ := 20

/-- The total number of muffins baked by all first grade classes -/
def total_muffins : ℕ := 55

/-- The number of muffins baked by Mrs. Flannery's class -/
def mrs_flannery_muffins : ℕ := total_muffins - (mrs_brier_muffins + mrs_macadams_muffins)

theorem mrs_flannery_muffins_count : mrs_flannery_muffins = 17 := by
  sorry

end NUMINAMATH_CALUDE_mrs_flannery_muffins_count_l3262_326249


namespace NUMINAMATH_CALUDE_travelers_checks_average_l3262_326257

theorem travelers_checks_average (x y : ℕ) : 
  x + y = 30 →
  50 * x + 100 * y = 1800 →
  let remaining_50 := x - 6
  let remaining_100 := y
  let total_remaining := remaining_50 + remaining_100
  let total_value := 50 * remaining_50 + 100 * remaining_100
  (total_value : ℚ) / total_remaining = 125/2 := by sorry

end NUMINAMATH_CALUDE_travelers_checks_average_l3262_326257


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3262_326284

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3262_326284


namespace NUMINAMATH_CALUDE_drill_bits_purchase_l3262_326277

theorem drill_bits_purchase (cost_per_set : ℝ) (tax_rate : ℝ) (total_paid : ℝ) 
  (h1 : cost_per_set = 6)
  (h2 : tax_rate = 0.1)
  (h3 : total_paid = 33) :
  ∃ (num_sets : ℕ), (cost_per_set * (num_sets : ℝ)) * (1 + tax_rate) = total_paid ∧ num_sets = 5 := by
  sorry

end NUMINAMATH_CALUDE_drill_bits_purchase_l3262_326277


namespace NUMINAMATH_CALUDE_boat_speed_proof_l3262_326261

/-- The speed of the boat in standing water -/
def boat_speed : ℝ := 9

/-- The speed of the stream -/
def stream_speed : ℝ := 6

/-- The distance traveled in one direction -/
def distance : ℝ := 170

/-- The total time taken for the round trip -/
def total_time : ℝ := 68

theorem boat_speed_proof :
  (distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed)) = total_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_proof_l3262_326261


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3262_326200

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  ((4 - 6*i) / (4 + 6*i) + (4 + 6*i) / (4 - 6*i)) = (-10 : ℚ) / 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3262_326200


namespace NUMINAMATH_CALUDE_legendre_symbol_values_legendre_symbol_square_equivalence_minus_one_square_mod_p_eleven_power_sum_of_squares_l3262_326263

-- Define the necessary variables and functions
variable (p : ℕ) (hp : Nat.Prime p) (hodd : p % 2 = 1)
variable (a : ℕ) (hcoprime : Nat.Coprime a p)

-- Theorem 1
theorem legendre_symbol_values :
  (a ^ ((p - 1) / 2)) % p = 1 ∨ (a ^ ((p - 1) / 2)) % p = p - 1 :=
sorry

-- Theorem 2
theorem legendre_symbol_square_equivalence :
  (a ^ ((p - 1) / 2)) % p = 1 ↔ ∃ x, (x * x) % p = a % p :=
sorry

-- Theorem 3
theorem minus_one_square_mod_p :
  (∃ x, (x * x) % p = p - 1) ↔ p % 4 = 1 :=
sorry

-- Theorem 4
theorem eleven_power_sum_of_squares (n : ℕ) :
  ∀ a b : ℕ, 11^n = a^2 + b^2 →
    ∃ k : ℕ, n = 2*k ∧ ((a = 11^k ∧ b = 0) ∨ (a = 0 ∧ b = 11^k)) :=
sorry

end NUMINAMATH_CALUDE_legendre_symbol_values_legendre_symbol_square_equivalence_minus_one_square_mod_p_eleven_power_sum_of_squares_l3262_326263


namespace NUMINAMATH_CALUDE_product_simplification_l3262_326293

theorem product_simplification :
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l3262_326293


namespace NUMINAMATH_CALUDE_square_perimeter_l3262_326244

/-- The perimeter of a square with side length 13 centimeters is 52 centimeters. -/
theorem square_perimeter : ∀ (s : ℝ), s = 13 → 4 * s = 52 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3262_326244


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3262_326278

/-- The solution set of (ax-1)(x-1) > 0 is (1/a, 1) -/
def SolutionSet (a : ℝ) : Prop :=
  ∀ x, (a * x - 1) * (x - 1) > 0 ↔ 1/a < x ∧ x < 1

theorem sufficient_not_necessary :
  (∀ a : ℝ, SolutionSet a → a < 1/2) ∧
  (∃ a : ℝ, a < 1/2 ∧ ¬(SolutionSet a)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3262_326278


namespace NUMINAMATH_CALUDE_parallel_to_same_line_implies_parallel_l3262_326256

-- Define a type for lines in a plane
variable {Line : Type}

-- Define a relation for parallel lines
variable (parallel : Line → Line → Prop)

-- Axiom: Parallel relation is symmetric
axiom parallel_symmetric {l1 l2 : Line} : parallel l1 l2 → parallel l2 l1

-- Axiom: Parallel relation is transitive
axiom parallel_transitive {l1 l2 l3 : Line} : parallel l1 l2 → parallel l2 l3 → parallel l1 l3

-- Theorem: If two lines are parallel to a third line, they are parallel to each other
theorem parallel_to_same_line_implies_parallel (l1 l2 l3 : Line) :
  parallel l1 l3 → parallel l2 l3 → parallel l1 l2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_to_same_line_implies_parallel_l3262_326256


namespace NUMINAMATH_CALUDE_line_slope_l3262_326265

theorem line_slope (t : ℝ) : 
  let x := 3 - (Real.sqrt 3 / 2) * t
  let y := 1 + (1 / 2) * t
  (y - 1) / (x - 3) = -Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l3262_326265


namespace NUMINAMATH_CALUDE_points_scored_in_quarter_l3262_326233

/-- Calculates the total points scored in a basketball quarter -/
def total_points_scored (two_point_shots : ℕ) (three_point_shots : ℕ) : ℕ :=
  2 * two_point_shots + 3 * three_point_shots

/-- Proves that given four 2-point shots and two 3-point shots, the total points scored is 14 -/
theorem points_scored_in_quarter : total_points_scored 4 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_points_scored_in_quarter_l3262_326233


namespace NUMINAMATH_CALUDE_B_60_is_identity_l3262_326220

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, -1, 0],
    ![1,  0, 0],
    ![0,  0, 1]]

theorem B_60_is_identity :
  B^60 = 1 := by sorry

end NUMINAMATH_CALUDE_B_60_is_identity_l3262_326220


namespace NUMINAMATH_CALUDE_negation_of_existence_logarithm_inequality_negation_l3262_326276

open Real

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x ∈ Set.Ioo 0 3, p x) ↔ (∀ x ∈ Set.Ioo 0 3, ¬ p x) := by sorry

theorem logarithm_inequality_negation :
  (¬ ∃ x₀ ∈ Set.Ioo 0 3, x₀ - 2 < log x₀) ↔
  (∀ x ∈ Set.Ioo 0 3, x - 2 ≥ log x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_logarithm_inequality_negation_l3262_326276


namespace NUMINAMATH_CALUDE_quadratic_root_difference_condition_l3262_326226

/-- For a quadratic equation x^2 + px + q = 0, 
    the condition for the difference of its roots to be 'a' is a^2 - p^2 = -4q -/
theorem quadratic_root_difference_condition 
  (p q a : ℝ) 
  (hq : ∃ x₁ x₂ : ℝ, x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0 ∧ x₁ ≠ x₂) :
  (∃ x₁ x₂ : ℝ, x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0 ∧ x₁ - x₂ = a) ↔ 
  a^2 - p^2 = -4*q :=
by sorry


end NUMINAMATH_CALUDE_quadratic_root_difference_condition_l3262_326226


namespace NUMINAMATH_CALUDE_xyz_value_l3262_326279

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 26 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l3262_326279


namespace NUMINAMATH_CALUDE_calculate_expression_l3262_326252

theorem calculate_expression : 2 * (-3)^2 - 4 * (-3) - 15 = 15 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3262_326252


namespace NUMINAMATH_CALUDE_sarah_investment_l3262_326241

/-- Proves that given a total investment of $250,000 and the investment in real estate
    being 6 times the investment in mutual funds, the amount invested in real estate
    is $214,285.71. -/
theorem sarah_investment (total : ℝ) (real_estate : ℝ) (mutual_funds : ℝ) 
    (h1 : total = 250000)
    (h2 : real_estate = 6 * mutual_funds)
    (h3 : total = real_estate + mutual_funds) :
  real_estate = 214285.71 := by
  sorry

end NUMINAMATH_CALUDE_sarah_investment_l3262_326241


namespace NUMINAMATH_CALUDE_highway_project_employees_l3262_326205

/-- Represents the highway construction project -/
structure HighwayProject where
  initial_workforce : ℕ
  total_length : ℕ
  initial_days : ℕ
  initial_hours_per_day : ℕ
  days_worked : ℕ
  work_completed : ℚ
  remaining_days : ℕ
  new_hours_per_day : ℕ

/-- Calculates the number of additional employees needed to complete the project on time -/
def additional_employees_needed (project : HighwayProject) : ℕ :=
  sorry

/-- Theorem stating that 60 additional employees are needed for the given project -/
theorem highway_project_employees (project : HighwayProject) 
  (h1 : project.initial_workforce = 100)
  (h2 : project.total_length = 2)
  (h3 : project.initial_days = 50)
  (h4 : project.initial_hours_per_day = 8)
  (h5 : project.days_worked = 25)
  (h6 : project.work_completed = 1/3)
  (h7 : project.remaining_days = 25)
  (h8 : project.new_hours_per_day = 10) :
  additional_employees_needed project = 60 :=
sorry

end NUMINAMATH_CALUDE_highway_project_employees_l3262_326205


namespace NUMINAMATH_CALUDE_actual_sleep_time_l3262_326267

/-- The required sleep time for middle school students -/
def requiredSleepTime : ℝ := 9

/-- The recorded excess sleep time for Xiao Ming -/
def recordedExcessTime : ℝ := 0.4

/-- Theorem: Actual sleep time is the sum of required sleep time and recorded excess time -/
theorem actual_sleep_time : 
  requiredSleepTime + recordedExcessTime = 9.4 := by
  sorry

end NUMINAMATH_CALUDE_actual_sleep_time_l3262_326267


namespace NUMINAMATH_CALUDE_min_product_xy_l3262_326206

theorem min_product_xy (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 6) :
  (∀ a b : ℕ+, (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 6 → x * y ≤ a * b) ∧ x * y = 96 :=
sorry

end NUMINAMATH_CALUDE_min_product_xy_l3262_326206


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l3262_326250

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_t_value :
  let m : ℝ × ℝ := (2, 8)
  let n : ℝ × ℝ := (-4, t)
  parallel m n → t = -16 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l3262_326250


namespace NUMINAMATH_CALUDE_output_for_15_l3262_326231

def function_machine (input : ℤ) : ℤ :=
  let step1 := input * 3
  if step1 > 25 then step1 - 7 else step1 + 10

theorem output_for_15 : function_machine 15 = 38 := by
  sorry

end NUMINAMATH_CALUDE_output_for_15_l3262_326231


namespace NUMINAMATH_CALUDE_square_greater_than_abs_l3262_326210

theorem square_greater_than_abs (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_abs_l3262_326210


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l3262_326275

def m : ℕ := 2017^2 + 2^2017

theorem units_digit_of_expression (n : ℕ) : n % 10 = (3 * (m^2 + 2^m)) % 10 → n % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l3262_326275


namespace NUMINAMATH_CALUDE_paint_time_problem_l3262_326270

theorem paint_time_problem (anthony_time : ℝ) (combined_time : ℝ) (first_person_time : ℝ) : 
  anthony_time = 5 →
  combined_time = 20 / 7 →
  (1 / first_person_time + 1 / anthony_time) * combined_time = 2 →
  first_person_time = 2 := by
sorry

end NUMINAMATH_CALUDE_paint_time_problem_l3262_326270


namespace NUMINAMATH_CALUDE_find_y_value_l3262_326294

theorem find_y_value (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 20) : y = 40 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l3262_326294


namespace NUMINAMATH_CALUDE_max_square_cookies_l3262_326207

theorem max_square_cookies (length width : ℕ) (h1 : length = 24) (h2 : width = 18) :
  let cookie_size := Nat.gcd length width
  (length / cookie_size) * (width / cookie_size) = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_square_cookies_l3262_326207


namespace NUMINAMATH_CALUDE_car_journey_time_l3262_326296

/-- Calculates the total time for a car journey with two segments and a stop -/
theorem car_journey_time (distance1 : ℝ) (speed1 : ℝ) (stop_time : ℝ) (distance2 : ℝ) (speed2 : ℝ) :
  distance1 = 150 ∧ speed1 = 50 ∧ stop_time = 0.5 ∧ distance2 = 200 ∧ speed2 = 75 →
  distance1 / speed1 + stop_time + distance2 / speed2 = 6.17 := by
  sorry

#eval (150 / 50 + 0.5 + 200 / 75 : Float)

end NUMINAMATH_CALUDE_car_journey_time_l3262_326296


namespace NUMINAMATH_CALUDE_circles_intersection_range_l3262_326202

-- Define the circles
def C₁ (t x y : ℝ) : Prop := x^2 + y^2 - 2*t*x + t^2 - 4 = 0
def C₂ (t x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*t*y + 4*t^2 - 8 = 0

-- Define the intersection condition
def intersect (t : ℝ) : Prop := ∃ x y : ℝ, C₁ t x y ∧ C₂ t x y

-- State the theorem
theorem circles_intersection_range :
  ∀ t : ℝ, intersect t ↔ ((-12/5 < t ∧ t < -2/5) ∨ (0 < t ∧ t < 2)) :=
by sorry

end NUMINAMATH_CALUDE_circles_intersection_range_l3262_326202


namespace NUMINAMATH_CALUDE_class_vision_median_l3262_326227

/-- Represents the vision data for a class of students -/
structure VisionData where
  values : List Float
  frequencies : List Nat
  total_students : Nat

/-- Calculates the median of the vision data -/
def median (data : VisionData) : Float :=
  sorry

/-- The specific vision data for the class -/
def class_vision_data : VisionData :=
  { values := [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0],
    frequencies := [1, 2, 6, 3, 3, 4, 1, 2, 5, 7, 5],
    total_students := 39 }

/-- Theorem stating that the median of the class vision data is 4.6 -/
theorem class_vision_median :
  median class_vision_data = 4.6 := by
  sorry

end NUMINAMATH_CALUDE_class_vision_median_l3262_326227


namespace NUMINAMATH_CALUDE_car_price_calculation_l3262_326214

theorem car_price_calculation (reduced_price : ℝ) (discount_percentage : ℝ) (original_price : ℝ) : 
  reduced_price = 7500 ∧ 
  discount_percentage = 25 ∧ 
  reduced_price = original_price * (1 - discount_percentage / 100) → 
  original_price = 10000 := by
sorry

end NUMINAMATH_CALUDE_car_price_calculation_l3262_326214


namespace NUMINAMATH_CALUDE_polynomial_expansion_problem_l3262_326208

theorem polynomial_expansion_problem (p q : ℝ) : 
  p > 0 → q > 0 → p + q = 1 → 
  55 * p^9 * q^2 = 165 * p^8 * q^3 → 
  p = 3/4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_problem_l3262_326208


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3262_326281

theorem polynomial_division_theorem (x : ℝ) : 
  x^5 - 25*x^3 + 13*x^2 - 16*x + 12 = (x - 3) * (x^4 + 3*x^3 - 16*x^2 - 35*x - 121) + (-297) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3262_326281


namespace NUMINAMATH_CALUDE_vertical_asymptote_at_neg_two_l3262_326245

/-- The function f(x) = (x^2 + 6x + 9) / (x + 2) has a vertical asymptote at x = -2 -/
theorem vertical_asymptote_at_neg_two :
  ∃ (f : ℝ → ℝ), 
    (∀ x ≠ -2, f x = (x^2 + 6*x + 9) / (x + 2)) ∧
    (∃ (L : ℝ → ℝ), (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x + 2| ∧ |x + 2| < δ → |f x| > L ε)) :=
by sorry

end NUMINAMATH_CALUDE_vertical_asymptote_at_neg_two_l3262_326245


namespace NUMINAMATH_CALUDE_initial_milk_cost_initial_milk_cost_is_four_l3262_326223

/-- Calculates the initial cost of milk given the grocery shopping scenario --/
theorem initial_milk_cost (total_money : ℝ) (bread_cost : ℝ) (detergent_cost : ℝ) 
  (banana_cost_per_pound : ℝ) (banana_pounds : ℝ) (detergent_coupon : ℝ) 
  (milk_discount_factor : ℝ) (money_left : ℝ) : ℝ :=
  let total_spent := total_money - money_left
  let banana_cost := banana_cost_per_pound * banana_pounds
  let discounted_detergent_cost := detergent_cost - detergent_coupon
  let non_milk_cost := bread_cost + banana_cost + discounted_detergent_cost
  let milk_cost := total_spent - non_milk_cost
  milk_cost / milk_discount_factor

/-- The initial cost of milk is $4 --/
theorem initial_milk_cost_is_four :
  initial_milk_cost 20 3.5 10.25 0.75 2 1.25 0.5 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_milk_cost_initial_milk_cost_is_four_l3262_326223


namespace NUMINAMATH_CALUDE_total_triangles_is_nine_l3262_326246

/-- Represents a triangular grid with a specific number of rows and triangles per row. -/
structure TriangularGrid where
  rows : Nat
  triangles_per_row : Nat → Nat
  row_count_correct : rows = 3
  top_row_correct : triangles_per_row 0 = 3
  second_row_correct : triangles_per_row 1 = 2
  bottom_row_correct : triangles_per_row 2 = 1

/-- Calculates the total number of triangles in the grid, including larger triangles formed by combining smaller ones. -/
def totalTriangles (grid : TriangularGrid) : Nat :=
  sorry

/-- Theorem stating that the total number of triangles in the specified triangular grid is 9. -/
theorem total_triangles_is_nine (grid : TriangularGrid) : totalTriangles grid = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_triangles_is_nine_l3262_326246


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3262_326291

theorem polynomial_expansion (x : ℝ) :
  (3 * x^2 - 4 * x + 3) * (-2 * x^2 + 3 * x - 4) =
  -6 * x^4 + 17 * x^3 - 30 * x^2 + 25 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3262_326291


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l3262_326283

/-- 
Given two bases a and b, both greater than 2, 
this function returns the base-10 representation of 21 in base a
-/
def base_a_to_10 (a : ℕ) : ℕ := 2 * a + 1

/-- 
Given two bases a and b, both greater than 2, 
this function returns the base-10 representation of 12 in base b
-/
def base_b_to_10 (b : ℕ) : ℕ := b + 2

/--
This theorem states that 7 is the smallest base-10 integer that can be represented
as 21_a in one base and 12_b in a different base, where a and b are any bases larger than 2.
-/
theorem smallest_dual_base_representation :
  ∀ a b : ℕ, a > 2 → b > 2 →
  (∃ n : ℕ, n < 7 ∧ base_a_to_10 a = n ∧ base_b_to_10 b = n) → False :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l3262_326283


namespace NUMINAMATH_CALUDE_tournament_result_l3262_326262

/-- The number of athletes with k points after m rounds in a tournament with 2^n participants -/
def f (n m k : ℕ) : ℕ := 2^(n - m) * (m.choose k)

/-- The number of athletes with 4 points after 7 rounds in a tournament with 2^n + 6 participants -/
def athletes_with_four_points (n : ℕ) : ℕ := 35 * 2^(n - 7) + 2

theorem tournament_result (n : ℕ) (h : n > 7) :
  athletes_with_four_points n = f n 7 4 + 2 := by sorry

end NUMINAMATH_CALUDE_tournament_result_l3262_326262


namespace NUMINAMATH_CALUDE_tank_water_fraction_l3262_326274

theorem tank_water_fraction (tank_capacity : ℚ) (initial_fraction : ℚ) (added_water : ℚ) : 
  tank_capacity = 56 →
  initial_fraction = 3/4 →
  added_water = 7 →
  (initial_fraction * tank_capacity + added_water) / tank_capacity = 7/8 := by
sorry

end NUMINAMATH_CALUDE_tank_water_fraction_l3262_326274


namespace NUMINAMATH_CALUDE_probability_second_new_given_first_new_l3262_326288

/-- Represents the total number of balls in the box -/
def total_balls : ℕ := 10

/-- Represents the number of new balls initially in the box -/
def new_balls : ℕ := 6

/-- Represents the number of old balls initially in the box -/
def old_balls : ℕ := 4

/-- Theorem stating the probability of drawing a new ball on the second draw,
    given that the first ball drawn was new -/
theorem probability_second_new_given_first_new :
  (new_balls - 1) / (total_balls - 1) = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_second_new_given_first_new_l3262_326288


namespace NUMINAMATH_CALUDE_sequence_term_l3262_326266

theorem sequence_term (a : ℕ → ℝ) (h : ∀ n, a n = Real.sqrt (3 * n - 1)) :
  a 7 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sequence_term_l3262_326266


namespace NUMINAMATH_CALUDE_gary_gold_amount_l3262_326236

/-- Proves that Gary has 30 grams of gold given the conditions of the problem -/
theorem gary_gold_amount (gary_cost_per_gram : ℝ) (anna_amount : ℝ) (anna_cost_per_gram : ℝ) (total_cost : ℝ)
  (h1 : gary_cost_per_gram = 15)
  (h2 : anna_amount = 50)
  (h3 : anna_cost_per_gram = 20)
  (h4 : total_cost = 1450)
  (h5 : gary_cost_per_gram * gary_amount + anna_amount * anna_cost_per_gram = total_cost) :
  gary_amount = 30 := by
  sorry

end NUMINAMATH_CALUDE_gary_gold_amount_l3262_326236


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_f_b_geq_f_a_l3262_326259

-- Define the function f
def f (a x : ℝ) : ℝ := |x - 2*a| + |x - a|

-- Theorem for part I
theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x > 3} = {x : ℝ | x < 0 ∨ x > 3} := by sorry

-- Theorem for part II
theorem f_b_geq_f_a (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  f a b ≥ f a a ∧
  (f a b = f a a ↔ (2*a - b) * (b - a) ≥ 0) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_f_b_geq_f_a_l3262_326259


namespace NUMINAMATH_CALUDE_ratio_equality_l3262_326280

theorem ratio_equality (a b : ℝ) (h1 : 7 * a = 8 * b) (h2 : a * b ≠ 0) :
  (a / 8) / (b / 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3262_326280


namespace NUMINAMATH_CALUDE_jack_ernie_income_ratio_l3262_326271

theorem jack_ernie_income_ratio :
  ∀ (ernie_prev ernie_curr jack_curr : ℝ),
    ernie_curr = (4/5) * ernie_prev →
    ernie_curr + jack_curr = 16800 →
    ernie_prev = 6000 →
    jack_curr / ernie_prev = 2 := by
  sorry

end NUMINAMATH_CALUDE_jack_ernie_income_ratio_l3262_326271


namespace NUMINAMATH_CALUDE_edward_board_game_cost_l3262_326217

def board_game_cost (total_cost : ℕ) (num_figures : ℕ) (figure_cost : ℕ) : ℕ :=
  total_cost - (num_figures * figure_cost)

theorem edward_board_game_cost :
  board_game_cost 30 4 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_edward_board_game_cost_l3262_326217


namespace NUMINAMATH_CALUDE_claire_age_in_two_years_l3262_326209

/-- Given that Jessica is 24 years old and 6 years older than Claire, 
    prove that Claire will be 20 years old in two years. -/
theorem claire_age_in_two_years 
  (jessica_age : ℕ) 
  (claire_age : ℕ) 
  (h1 : jessica_age = 24)
  (h2 : jessica_age = claire_age + 6) : 
  claire_age + 2 = 20 := by
sorry

end NUMINAMATH_CALUDE_claire_age_in_two_years_l3262_326209


namespace NUMINAMATH_CALUDE_revenue_equals_scientific_notation_l3262_326247

/-- Represents the total revenue in yuan -/
def total_revenue : ℝ := 998.64e9

/-- Represents the scientific notation of the total revenue -/
def scientific_notation : ℝ := 9.9864e11

/-- Theorem stating that the total revenue is equal to its scientific notation representation -/
theorem revenue_equals_scientific_notation : total_revenue = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_revenue_equals_scientific_notation_l3262_326247


namespace NUMINAMATH_CALUDE_lcm_18_24_l3262_326240

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l3262_326240


namespace NUMINAMATH_CALUDE_nested_fraction_equals_seven_halves_l3262_326237

theorem nested_fraction_equals_seven_halves :
  2 + 2 / (1 + 1 / (2 + 1)) = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equals_seven_halves_l3262_326237


namespace NUMINAMATH_CALUDE_original_people_in_room_l3262_326298

theorem original_people_in_room (x : ℝ) : 
  (x / 2 = 15) → 
  (x / 3 + x / 4 * (2 / 3) + 15 = x) → 
  x = 30 := by
sorry

end NUMINAMATH_CALUDE_original_people_in_room_l3262_326298


namespace NUMINAMATH_CALUDE_false_propositions_count_l3262_326215

-- Define the original proposition
def original_prop (m n : ℝ) : Prop := m > -n → m^2 > n^2

-- Define the contrapositive
def contrapositive (m n : ℝ) : Prop := ¬(m^2 > n^2) → ¬(m > -n)

-- Define the inverse
def inverse (m n : ℝ) : Prop := m^2 > n^2 → m > -n

-- Define the negation
def negation (m n : ℝ) : Prop := ¬(m > -n → m^2 > n^2)

-- Theorem statement
theorem false_propositions_count :
  ∃ (m n : ℝ), (¬(original_prop m n) ∧ ¬(contrapositive m n) ∧ ¬(inverse m n) ∧ ¬(negation m n)) :=
by sorry

end NUMINAMATH_CALUDE_false_propositions_count_l3262_326215


namespace NUMINAMATH_CALUDE_valid_parameterization_l3262_326232

/-- Defines a line in 2D space --/
structure Line2D where
  slope : ℚ
  intercept : ℚ

/-- Defines a vector parameterization of a line --/
structure VectorParam where
  x₀ : ℚ
  y₀ : ℚ
  a : ℚ
  b : ℚ

/-- Checks if a vector parameterization is valid for a given line --/
def isValidParam (l : Line2D) (p : VectorParam) : Prop :=
  ∃ (k : ℚ), p.a = k * 5 ∧ p.b = k * 7 ∧ 
  p.y₀ = l.slope * p.x₀ + l.intercept

/-- The main theorem to prove --/
theorem valid_parameterization (l : Line2D) (p : VectorParam) :
  l.slope = 7/5 ∧ l.intercept = -23/5 →
  isValidParam l p ↔ 
    (p.x₀ = 5 ∧ p.y₀ = 2 ∧ p.a = -5 ∧ p.b = -7) ∨
    (p.x₀ = 23 ∧ p.y₀ = 7 ∧ p.a = 10 ∧ p.b = 14) ∨
    (p.x₀ = 3 ∧ p.y₀ = -8/5 ∧ p.a = 7/5 ∧ p.b = 1) ∨
    (p.x₀ = 0 ∧ p.y₀ = -23/5 ∧ p.a = 25 ∧ p.b = -35) :=
by sorry

end NUMINAMATH_CALUDE_valid_parameterization_l3262_326232


namespace NUMINAMATH_CALUDE_car_trip_distance_l3262_326228

theorem car_trip_distance (D : ℝ) :
  let remaining_after_first_stop := D / 2
  let remaining_after_second_stop := remaining_after_first_stop * 2 / 3
  let remaining_after_third_stop := remaining_after_second_stop * 3 / 5
  remaining_after_third_stop = 180
  → D = 900 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_distance_l3262_326228


namespace NUMINAMATH_CALUDE_other_coin_denomination_l3262_326251

/-- Proves that the denomination of the other type of coin is 25 paise -/
theorem other_coin_denomination :
  let total_coins : ℕ := 342
  let total_value : ℕ := 7100  -- in paise
  let twenty_paise_coins : ℕ := 290
  let twenty_paise_value : ℕ := 20

  let other_coins : ℕ := total_coins - twenty_paise_coins
  let other_coins_value : ℕ := total_value - (twenty_paise_coins * twenty_paise_value)
  
  other_coins_value / other_coins = 25 := by
    sorry

end NUMINAMATH_CALUDE_other_coin_denomination_l3262_326251


namespace NUMINAMATH_CALUDE_ab_neg_necessary_not_sufficient_for_hyperbola_l3262_326204

/-- Represents a conic section in the form ax^2 + by^2 = c -/
structure ConicSection where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to determine if a conic section is a hyperbola -/
def IsHyperbola (conic : ConicSection) : Prop :=
  sorry  -- The actual definition would depend on the formal definition of a hyperbola

/-- The main theorem stating that ab < 0 is necessary but not sufficient for a hyperbola -/
theorem ab_neg_necessary_not_sufficient_for_hyperbola :
  (∀ conic : ConicSection, IsHyperbola conic → conic.a * conic.b < 0) ∧
  (∃ conic : ConicSection, conic.a * conic.b < 0 ∧ ¬IsHyperbola conic) :=
sorry

end NUMINAMATH_CALUDE_ab_neg_necessary_not_sufficient_for_hyperbola_l3262_326204


namespace NUMINAMATH_CALUDE_problem_statement_l3262_326282

theorem problem_statement (a b c : ℝ) (h1 : a - b = 3) (h2 : b - c = 2) :
  (a - c)^2 + 3*a + 1 - 3*c = 41 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3262_326282


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3262_326221

theorem absolute_value_inequality (x : ℝ) : 
  abs (x - 3) + abs (x - 5) ≥ 4 ↔ x ≥ 6 ∨ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3262_326221


namespace NUMINAMATH_CALUDE_circle_point_range_l3262_326289

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - Real.sqrt 3)^2 = 1

-- Define points A and B
def point_A (m : ℝ) : ℝ × ℝ := (0, m)
def point_B (m : ℝ) : ℝ × ℝ := (0, -m)

-- Define the condition for point P
def point_P_condition (P : ℝ × ℝ) (m : ℝ) : Prop :=
  circle_C P.1 P.2 ∧ 
  ∃ (A B : ℝ × ℝ), A = point_A m ∧ B = point_B m ∧ 
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

-- Main theorem
theorem circle_point_range (m : ℝ) :
  m > 0 → (∃ P : ℝ × ℝ, point_P_condition P m) → 1 ≤ m ∧ m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_point_range_l3262_326289


namespace NUMINAMATH_CALUDE_calculate_principal_l3262_326269

/-- Given simple interest, time, and rate, calculate the principal amount -/
theorem calculate_principal (simple_interest : ℝ) (time : ℝ) (rate : ℝ) :
  simple_interest = 140 ∧ time = 2 ∧ rate = 17.5 →
  (simple_interest / (rate * time / 100) : ℝ) = 400 := by
  sorry

end NUMINAMATH_CALUDE_calculate_principal_l3262_326269


namespace NUMINAMATH_CALUDE_pencils_with_eraser_count_l3262_326287

/-- The number of pencils with an eraser sold in a stationery store -/
def pencils_with_eraser : ℕ := sorry

/-- The price of a pencil with an eraser -/
def price_eraser : ℚ := 8/10

/-- The price of a regular pencil -/
def price_regular : ℚ := 1/2

/-- The price of a short pencil -/
def price_short : ℚ := 4/10

/-- The number of regular pencils sold -/
def regular_sold : ℕ := 40

/-- The number of short pencils sold -/
def short_sold : ℕ := 35

/-- The total revenue from all pencil sales -/
def total_revenue : ℚ := 194

/-- Theorem stating that the number of pencils with an eraser sold is 200 -/
theorem pencils_with_eraser_count : pencils_with_eraser = 200 :=
  by sorry

end NUMINAMATH_CALUDE_pencils_with_eraser_count_l3262_326287


namespace NUMINAMATH_CALUDE_system_solution_l3262_326286

def solution_set : Set (ℝ × ℝ) :=
  {(-2/Real.sqrt 5, 1/Real.sqrt 5), (-2/Real.sqrt 5, -1/Real.sqrt 5),
   (2/Real.sqrt 5, -1/Real.sqrt 5), (2/Real.sqrt 5, 1/Real.sqrt 5)}

def satisfies_system (x y : ℝ) : Prop :=
  x^2 + y^2 ≤ 1 ∧
  16*x^4 - 8*x^2*y^2 + y^4 - 40*x^2 - 10*y^2 + 25 = 0

theorem system_solution :
  ∀ x y : ℝ, satisfies_system x y ↔ (x, y) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3262_326286


namespace NUMINAMATH_CALUDE_difference_second_third_bus_l3262_326225

/-- The number of buses hired for the school trip -/
def num_buses : ℕ := 4

/-- The number of people on the first bus -/
def first_bus : ℕ := 12

/-- The number of people on the second bus -/
def second_bus : ℕ := 2 * first_bus

/-- The number of people on the fourth bus -/
def fourth_bus : ℕ := first_bus + 9

/-- The total number of people going to the museum -/
def total_people : ℕ := 75

/-- The number of people on the third bus -/
def third_bus : ℕ := total_people - (first_bus + second_bus + fourth_bus)

theorem difference_second_third_bus : second_bus - third_bus = 6 := by
  sorry

end NUMINAMATH_CALUDE_difference_second_third_bus_l3262_326225


namespace NUMINAMATH_CALUDE_sequence_proof_l3262_326229

theorem sequence_proof (a : Fin 8 → ℕ) 
  (h1 : a 0 = 11)
  (h2 : a 7 = 12)
  (h3 : ∀ i : Fin 6, a i + a (i + 1) + a (i + 2) = 50) :
  a = ![11, 12, 27, 11, 12, 27, 11, 12] := by
sorry

end NUMINAMATH_CALUDE_sequence_proof_l3262_326229


namespace NUMINAMATH_CALUDE_flour_recipe_reduction_reduced_recipe_as_mixed_number_l3262_326235

theorem flour_recipe_reduction :
  let original_recipe : ℚ := 19/4  -- 4 3/4 as an improper fraction
  let reduced_recipe : ℚ := original_recipe / 3
  reduced_recipe = 19/12 := by sorry

theorem reduced_recipe_as_mixed_number :
  (19 : ℚ) / 12 = 1 + 7/12 := by sorry

end NUMINAMATH_CALUDE_flour_recipe_reduction_reduced_recipe_as_mixed_number_l3262_326235


namespace NUMINAMATH_CALUDE_solution_range_solution_range_converse_l3262_326299

/-- The system of equations has two distinct solutions -/
def has_two_distinct_solutions (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
  y₁ = Real.sqrt (-x₁^2 - 2*x₁) ∧ x₁ + y₁ - m = 0 ∧
  y₂ = Real.sqrt (-x₂^2 - 2*x₂) ∧ x₂ + y₂ - m = 0

/-- The main theorem -/
theorem solution_range (m : ℝ) : 
  has_two_distinct_solutions m → m ∈ Set.Icc 0 (-1 + Real.sqrt 2) :=
by
  sorry

/-- The converse of the main theorem -/
theorem solution_range_converse (m : ℝ) : 
  m ∈ Set.Ioo 0 (-1 + Real.sqrt 2) → has_two_distinct_solutions m :=
by
  sorry

end NUMINAMATH_CALUDE_solution_range_solution_range_converse_l3262_326299


namespace NUMINAMATH_CALUDE_siblings_weekly_water_consumption_l3262_326203

/-- The amount of water consumed by three siblings in a week -/
def water_consumption (theo_daily : ℕ) (mason_daily : ℕ) (roxy_daily : ℕ) (days_in_week : ℕ) : ℕ :=
  (theo_daily + mason_daily + roxy_daily) * days_in_week

/-- Theorem stating that the siblings drink 168 cups of water in a week -/
theorem siblings_weekly_water_consumption :
  water_consumption 8 7 9 7 = 168 := by
  sorry

end NUMINAMATH_CALUDE_siblings_weekly_water_consumption_l3262_326203


namespace NUMINAMATH_CALUDE_number_of_factors_of_30_l3262_326216

theorem number_of_factors_of_30 : Finset.card (Nat.divisors 30) = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_30_l3262_326216


namespace NUMINAMATH_CALUDE_cube_difference_l3262_326260

theorem cube_difference (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 66) :
  a^3 - b^3 = 486 := by sorry

end NUMINAMATH_CALUDE_cube_difference_l3262_326260


namespace NUMINAMATH_CALUDE_age_ratio_in_two_years_l3262_326272

def lennon_current_age : ℕ := 8
def ophelia_current_age : ℕ := 38
def years_passed : ℕ := 2

def lennon_future_age : ℕ := lennon_current_age + years_passed
def ophelia_future_age : ℕ := ophelia_current_age + years_passed

theorem age_ratio_in_two_years :
  ophelia_future_age / lennon_future_age = 4 ∧ ophelia_future_age % lennon_future_age = 0 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_in_two_years_l3262_326272


namespace NUMINAMATH_CALUDE_sandys_age_l3262_326211

theorem sandys_age (sandy_age molly_age : ℕ) : 
  molly_age = sandy_age + 18 →
  sandy_age * 9 = molly_age * 7 →
  sandy_age = 63 := by
sorry

end NUMINAMATH_CALUDE_sandys_age_l3262_326211


namespace NUMINAMATH_CALUDE_harriett_quarters_l3262_326239

/-- Represents the number of coins of each type found by Harriett --/
structure CoinCount where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total value in cents for a given coin count --/
def totalValue (coins : CoinCount) : ℕ :=
  coins.quarters * 25 + coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- The coin count found by Harriett --/
def harriettCoins : CoinCount := {
  quarters := 10,  -- This is what we want to prove
  dimes := 3,
  nickels := 3,
  pennies := 5
}

theorem harriett_quarters : 
  harriettCoins.quarters = 10 ∧ totalValue harriettCoins = 300 := by
  sorry

end NUMINAMATH_CALUDE_harriett_quarters_l3262_326239


namespace NUMINAMATH_CALUDE_combined_age_in_five_years_l3262_326248

/-- Given the current ages and relationships of Amy, Mark, and Emily, 
    prove their combined age in 5 years. -/
theorem combined_age_in_five_years 
  (amy_age : ℕ) 
  (mark_age : ℕ) 
  (emily_age : ℕ) 
  (h1 : amy_age = 15)
  (h2 : mark_age = amy_age + 7)
  (h3 : emily_age = 2 * amy_age) :
  amy_age + 5 + (mark_age + 5) + (emily_age + 5) = 82 :=
by sorry

end NUMINAMATH_CALUDE_combined_age_in_five_years_l3262_326248


namespace NUMINAMATH_CALUDE_train_crossing_time_l3262_326222

/-- Proves that a train with given length and speed takes a specific time to cross a pole -/
theorem train_crossing_time (train_length_m : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) :
  train_length_m = 1250 →
  train_speed_kmh = 300 →
  crossing_time_s = 15 →
  crossing_time_s = (train_length_m / 1000) / (train_speed_kmh / 3600) := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3262_326222


namespace NUMINAMATH_CALUDE_quadratic_equation_consequence_l3262_326238

theorem quadratic_equation_consequence (m : ℝ) (h : m^2 + 2*m - 1 = 0) :
  2*m^2 + 4*m - 3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_consequence_l3262_326238


namespace NUMINAMATH_CALUDE_find_y_l3262_326212

theorem find_y (x y : ℤ) (h1 : x + y = 300) (h2 : x - y = 200) : y = 50 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l3262_326212


namespace NUMINAMATH_CALUDE_inequality_of_cubes_l3262_326243

theorem inequality_of_cubes (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  4 * (a^3 + b^3) > (a + b)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_cubes_l3262_326243


namespace NUMINAMATH_CALUDE_car_transfer_equation_l3262_326253

theorem car_transfer_equation (x : ℕ) : 
  (100 - x = 68 + x) ↔ 
  (∃ (team_a team_b : ℕ), 
    team_a = 100 ∧ 
    team_b = 68 ∧ 
    team_a - x = team_b + x) :=
sorry

end NUMINAMATH_CALUDE_car_transfer_equation_l3262_326253


namespace NUMINAMATH_CALUDE_square_area_to_cube_volume_ratio_l3262_326234

theorem square_area_to_cube_volume_ratio 
  (cube : Real → Real) 
  (square : Real → Real) 
  (h : ∀ s : Real, s > 0 → s * Real.sqrt 3 = 4 * square s) :
  ∀ s : Real, s > 0 → (square s)^2 / (cube s) = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_square_area_to_cube_volume_ratio_l3262_326234


namespace NUMINAMATH_CALUDE_unique_integral_solution_l3262_326285

theorem unique_integral_solution (x y z n : ℤ) 
  (h1 : x * y + y * z + z * x = 3 * n^2 - 1)
  (h2 : x + y + z = 3 * n)
  (h3 : x ≥ y ∧ y ≥ z) :
  x = n + 1 ∧ y = n ∧ z = n - 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_integral_solution_l3262_326285


namespace NUMINAMATH_CALUDE_diorama_time_proof_l3262_326292

/-- Proves that the total time spent on a diorama is 67 minutes, given the specified conditions. -/
theorem diorama_time_proof (planning_time building_time : ℕ) : 
  building_time = 3 * planning_time - 5 →
  building_time = 49 →
  planning_time + building_time = 67 := by
  sorry

#check diorama_time_proof

end NUMINAMATH_CALUDE_diorama_time_proof_l3262_326292


namespace NUMINAMATH_CALUDE_expansion_coefficient_l3262_326273

/-- The coefficient of x^(3/2) in the expansion of (√x - a/x)^6 -/
def coefficient (a : ℝ) : ℝ := 6 * (-a)

theorem expansion_coefficient (a : ℝ) : coefficient a = 30 → a = -5 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l3262_326273


namespace NUMINAMATH_CALUDE_least_positive_difference_l3262_326219

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

def sequence_C (n : ℕ) : ℝ := geometric_sequence 3 3 n

def sequence_D (n : ℕ) : ℝ := arithmetic_sequence 10 20 n

def valid_C (n : ℕ) : Prop := sequence_C n ≤ 200

def valid_D (n : ℕ) : Prop := sequence_D n ≤ 200

theorem least_positive_difference :
  ∃ (m n : ℕ) (h₁ : valid_C m) (h₂ : valid_D n),
    ∀ (p q : ℕ) (h₃ : valid_C p) (h₄ : valid_D q),
      |sequence_C m - sequence_D n| ≤ |sequence_C p - sequence_D q| ∧
      |sequence_C m - sequence_D n| > 0 ∧
      |sequence_C m - sequence_D n| = 9 :=
sorry

end NUMINAMATH_CALUDE_least_positive_difference_l3262_326219


namespace NUMINAMATH_CALUDE_zoo_guides_theorem_l3262_326242

/-- The total number of children addressed by zoo guides --/
def total_children (total_guides : ℕ) 
                   (english_guides : ℕ) 
                   (french_guides : ℕ) 
                   (english_children : ℕ) 
                   (french_children : ℕ) 
                   (spanish_children : ℕ) : ℕ :=
  let spanish_guides := total_guides - english_guides - french_guides
  english_guides * english_children + 
  french_guides * french_children + 
  spanish_guides * spanish_children

/-- Theorem stating the total number of children addressed by zoo guides --/
theorem zoo_guides_theorem : 
  total_children 22 10 6 19 25 30 = 520 := by
  sorry

end NUMINAMATH_CALUDE_zoo_guides_theorem_l3262_326242


namespace NUMINAMATH_CALUDE_simplify_expression_l3262_326258

theorem simplify_expression : 0.4 * 0.5 + 0.3 * 0.2 = 0.26 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3262_326258


namespace NUMINAMATH_CALUDE_equation_solutions_l3262_326254

theorem equation_solutions :
  let eq1 : ℝ → Prop := λ x ↦ x^2 - 8*x + 6 = 0
  let eq2 : ℝ → Prop := λ x ↦ (x - 1)^2 = 3*x - 3
  let sol1 : Set ℝ := {4 + Real.sqrt 10, 4 - Real.sqrt 10}
  let sol2 : Set ℝ := {1, 4}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ y ∉ sol1, ¬eq1 y) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ y ∉ sol2, ¬eq2 y) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3262_326254
