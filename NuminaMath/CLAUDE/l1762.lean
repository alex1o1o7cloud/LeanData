import Mathlib

namespace NUMINAMATH_CALUDE_min_value_expression_l1762_176203

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 3) (h_rel : x = 2 * y) :
  ∃ (min : ℝ), min = 4/3 ∧ ∀ x y z, x > 0 → y > 0 → z > 0 → x + y + z = 3 → x = 2 * y →
    (x + y) / (x * y * z) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1762_176203


namespace NUMINAMATH_CALUDE_guan_yu_travel_liu_bei_travel_speed_ratio_l1762_176266

/-- The distance between the long and short pavilions -/
def D : ℝ := sorry

/-- Guan Yu's speed -/
def speed_guan_yu : ℝ := sorry

/-- Liu Bei's speed -/
def speed_liu_bei : ℝ := sorry

/-- Guan Yu travels 1.5D in 5 hours -/
theorem guan_yu_travel : 1.5 * D = 5 * speed_guan_yu := sorry

/-- Liu Bei travels 0.5D in 2 hours -/
theorem liu_bei_travel : 0.5 * D = 2 * speed_liu_bei := sorry

/-- The ratio of Liu Bei's speed to Guan Yu's speed is 5:6 -/
theorem speed_ratio : speed_liu_bei / speed_guan_yu = 5 / 6 := sorry

end NUMINAMATH_CALUDE_guan_yu_travel_liu_bei_travel_speed_ratio_l1762_176266


namespace NUMINAMATH_CALUDE_smallest_with_12_divisors_l1762_176205

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is the smallest positive integer satisfying property P -/
def is_smallest_satisfying (n : ℕ) (P : ℕ → Prop) : Prop :=
  P n ∧ ∀ m : ℕ, 0 < m ∧ m < n → ¬P m

theorem smallest_with_12_divisors :
  is_smallest_satisfying 288 (λ n => num_divisors n = 12) := by sorry

end NUMINAMATH_CALUDE_smallest_with_12_divisors_l1762_176205


namespace NUMINAMATH_CALUDE_tan_derivative_l1762_176232

-- Define the tangent function
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- State the theorem
theorem tan_derivative (x : ℝ) :
  deriv tan x = 1 / (Real.cos x)^2 :=
by sorry

end NUMINAMATH_CALUDE_tan_derivative_l1762_176232


namespace NUMINAMATH_CALUDE_wall_ratio_l1762_176217

theorem wall_ratio (width height length volume : ℝ) :
  width = 4 →
  height = 6 * width →
  volume = width * height * length →
  volume = 16128 →
  length / height = 7 := by
sorry

end NUMINAMATH_CALUDE_wall_ratio_l1762_176217


namespace NUMINAMATH_CALUDE_stock_investment_change_l1762_176210

theorem stock_investment_change (initial_investment : ℝ) : 
  initial_investment > 0 → 
  let first_year := initial_investment * (1 + 0.80)
  let second_year := first_year * (1 - 0.30)
  second_year = initial_investment * 1.26 := by
sorry

end NUMINAMATH_CALUDE_stock_investment_change_l1762_176210


namespace NUMINAMATH_CALUDE_stating_medication_duration_l1762_176263

/-- Represents the number of pills in one supply of medication -/
def pills_per_supply : ℕ := 60

/-- Represents the fraction of a pill taken each time -/
def pill_fraction : ℚ := 1/3

/-- Represents the number of days between each dose -/
def days_between_doses : ℕ := 3

/-- Represents the number of types of medication -/
def medication_types : ℕ := 2

/-- Represents the approximate number of days in a month -/
def days_per_month : ℕ := 30

/-- 
Theorem stating that the combined supply of medication will last 540 days,
which is approximately 18 months.
-/
theorem medication_duration :
  (pills_per_supply : ℚ) * days_between_doses / pill_fraction * medication_types = 540 ∧
  540 / days_per_month = 18 := by
  sorry


end NUMINAMATH_CALUDE_stating_medication_duration_l1762_176263


namespace NUMINAMATH_CALUDE_test_probabilities_l1762_176286

theorem test_probabilities (p_A p_B p_C : ℝ) 
  (h_A : p_A = 0.8) (h_B : p_B = 0.6) (h_C : p_C = 0.5) : 
  p_A * p_B * p_C = 0.24 ∧ 
  1 - (1 - p_A) * (1 - p_B) * (1 - p_C) = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_test_probabilities_l1762_176286


namespace NUMINAMATH_CALUDE_cheaper_count_l1762_176243

def C (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 20 then 15 * n
  else if 21 ≤ n ∧ n ≤ 40 then 13 * n - 10
  else if 41 ≤ n ∧ n ≤ 60 then 12 * n
  else 11 * n

def cheaper_to_buy_more (n : ℕ) : Prop :=
  ∃ k : ℕ+, C (n + k) < C n

theorem cheaper_count : 
  (∃ s : Finset ℕ, s.card = 7 ∧ ∀ n, n ∈ s ↔ cheaper_to_buy_more n) :=
sorry

end NUMINAMATH_CALUDE_cheaper_count_l1762_176243


namespace NUMINAMATH_CALUDE_triangle_area_l1762_176235

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  A = π / 6 →  -- 30°
  C = π / 4 →  -- 45°
  a = 2 →
  B + C + A = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 + 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l1762_176235


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1762_176268

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^3 + x ≥ 0) ↔ (∃ x : ℝ, x < 0 ∧ x^3 + x < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1762_176268


namespace NUMINAMATH_CALUDE_brick_length_calculation_l1762_176209

theorem brick_length_calculation (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (brick_width : ℝ) (total_bricks : ℕ) :
  courtyard_length = 25 →
  courtyard_width = 15 →
  brick_width = 0.1 →
  total_bricks = 18750 →
  (courtyard_length * courtyard_width * 10000) / (total_bricks * brick_width) = 20 := by
  sorry

end NUMINAMATH_CALUDE_brick_length_calculation_l1762_176209


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l1762_176249

-- Define the circles O₁ and O₂ in polar coordinates
def circle_O₁ (ρ θ : ℝ) : Prop := ρ = Real.sin θ
def circle_O₂ (ρ θ : ℝ) : Prop := ρ = Real.cos θ

-- Define the line in Cartesian coordinates
def intersection_line (x y : ℝ) : Prop := x - y = 0

-- Theorem statement
theorem intersection_line_of_circles :
  ∀ (x y : ℝ), (∃ (ρ θ : ℝ), circle_O₁ ρ θ ∧ circle_O₂ ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  intersection_line x y :=
sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l1762_176249


namespace NUMINAMATH_CALUDE_complex_cube_sum_ratio_l1762_176233

theorem complex_cube_sum_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 10)
  (h_squared_diff : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 13 :=
by sorry

end NUMINAMATH_CALUDE_complex_cube_sum_ratio_l1762_176233


namespace NUMINAMATH_CALUDE_missing_number_in_mean_l1762_176287

theorem missing_number_in_mean (known_numbers : List ℝ) (mean : ℝ) : 
  known_numbers = [1, 22, 23, 24, 26, 27, 2] ∧ 
  mean = 20 ∧ 
  (List.sum known_numbers + 35) / 8 = mean →
  35 = 8 * mean - List.sum known_numbers :=
by
  sorry

#check missing_number_in_mean

end NUMINAMATH_CALUDE_missing_number_in_mean_l1762_176287


namespace NUMINAMATH_CALUDE_pizza_combinations_l1762_176204

def number_of_toppings : ℕ := 8
def toppings_per_pizza : ℕ := 3

theorem pizza_combinations :
  Nat.choose number_of_toppings toppings_per_pizza = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l1762_176204


namespace NUMINAMATH_CALUDE_triangle_properties_l1762_176273

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side opposite to A
  b : ℝ  -- side opposite to B
  c : ℝ  -- side opposite to C

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.a ≠ abc.b)
  (h2 : abc.c = Real.sqrt 3)
  (h3 : (Real.cos abc.A)^2 - (Real.cos abc.B)^2 = Real.sqrt 3 * Real.sin abc.A * Real.cos abc.A - Real.sqrt 3 * Real.sin abc.B * Real.cos abc.B)
  (h4 : Real.sin abc.A = 4/5) :
  abc.C = π/3 ∧ 
  (1/2 * abc.a * abc.b * Real.sin abc.C) = (8 * Real.sqrt 3 + 18) / 25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1762_176273


namespace NUMINAMATH_CALUDE_jenny_games_against_mark_l1762_176276

theorem jenny_games_against_mark (mark_wins : ℕ) (jenny_wins : ℕ) 
  (h1 : mark_wins = 1)
  (h2 : jenny_wins = 14) :
  ∃ m : ℕ,
    (m - mark_wins) + (2 * m - (3/4 * 2 * m)) = jenny_wins ∧ 
    m = 30 := by
  sorry

end NUMINAMATH_CALUDE_jenny_games_against_mark_l1762_176276


namespace NUMINAMATH_CALUDE_complex_product_equals_two_l1762_176267

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_product_equals_two : (1 - i)^2 * i = 2 := by sorry

end NUMINAMATH_CALUDE_complex_product_equals_two_l1762_176267


namespace NUMINAMATH_CALUDE_parabola_equation_proof_l1762_176260

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 13 - y^2 / 12 = 1

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (5, 0)

/-- The vertex of the parabola -/
def parabola_vertex : ℝ × ℝ := (0, 0)

/-- The focus of the parabola -/
def parabola_focus : ℝ × ℝ := right_focus

/-- The equation of the parabola -/
def parabola_equation (x y : ℝ) : Prop := y^2 = 20 * x

theorem parabola_equation_proof :
  ∀ x y : ℝ, parabola_equation x y ↔ 
  (parabola_vertex = (0, 0) ∧ parabola_focus = right_focus) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_proof_l1762_176260


namespace NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l1762_176279

/-- Represents a tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  /-- The angle between faces ABC and BCD in radians -/
  angle : ℝ
  /-- The area of face ABC -/
  area_ABC : ℝ
  /-- The area of face BCD -/
  area_BCD : ℝ
  /-- The length of edge BC -/
  length_BC : ℝ

/-- Calculates the volume of the tetrahedron -/
def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 320 -/
theorem volume_of_specific_tetrahedron :
  let t : Tetrahedron := {
    angle := 30 * π / 180,  -- 30 degrees in radians
    area_ABC := 120,
    area_BCD := 80,
    length_BC := 10
  }
  volume t = 320 := by sorry

end NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l1762_176279


namespace NUMINAMATH_CALUDE_cube_of_ten_expansion_l1762_176248

theorem cube_of_ten_expansion : 9^3 + 3*(9^2) + 3*9 + 1 = 1000 := by sorry

end NUMINAMATH_CALUDE_cube_of_ten_expansion_l1762_176248


namespace NUMINAMATH_CALUDE_quadratic_function_with_log_range_l1762_176277

/-- A quadratic function -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_with_log_range
  (f : ℝ → ℝ)
  (h1 : QuadraticFunction f)
  (h2 : Set.range (fun x ↦ Real.log (f x)) = Set.Ici 0) :
  ∃ a b : ℝ, f = fun x ↦ x^2 + 2*x + 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_with_log_range_l1762_176277


namespace NUMINAMATH_CALUDE_sqrt_x_minus_two_real_l1762_176250

theorem sqrt_x_minus_two_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_two_real_l1762_176250


namespace NUMINAMATH_CALUDE_abs_value_equality_l1762_176281

theorem abs_value_equality (m : ℝ) : |m| = |-3| → m = 3 ∨ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_equality_l1762_176281


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1762_176238

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ ∀ n, a (n + 1) = a n * r

/-- The fourth term of a geometric sequence with first term 3 and third term 75 is 375. -/
theorem fourth_term_of_geometric_sequence (a : ℕ → ℕ) (h : IsGeometricSequence a) 
    (h1 : a 1 = 3) (h3 : a 3 = 75) : a 4 = 375 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1762_176238


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l1762_176253

theorem smallest_number_with_conditions : ∃ n : ℕ, 
  (n = 1801) ∧ 
  (∀ m : ℕ, m < n → 
    (11 ∣ n) ∧ 
    (n % 2 = 1) ∧ 
    (n % 3 = 1) ∧ 
    (n % 4 = 1) ∧ 
    (n % 5 = 1) ∧ 
    (n % 6 = 1) ∧ 
    (n % 8 = 1) → 
    ¬((11 ∣ m) ∧ 
      (m % 2 = 1) ∧ 
      (m % 3 = 1) ∧ 
      (m % 4 = 1) ∧ 
      (m % 5 = 1) ∧ 
      (m % 6 = 1) ∧ 
      (m % 8 = 1))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l1762_176253


namespace NUMINAMATH_CALUDE_isabella_babysitting_afternoons_l1762_176255

/-- Calculates the number of afternoons Isabella babysits per week -/
def babysitting_afternoons (hourly_rate : ℚ) (hours_per_day : ℚ) (total_weeks : ℕ) (total_earnings : ℚ) : ℚ :=
  (total_earnings / total_weeks) / (hourly_rate * hours_per_day)

/-- Proves that Isabella babysits 6 afternoons per week -/
theorem isabella_babysitting_afternoons :
  babysitting_afternoons 5 5 7 1050 = 6 := by
  sorry

end NUMINAMATH_CALUDE_isabella_babysitting_afternoons_l1762_176255


namespace NUMINAMATH_CALUDE_geometric_sequence_cosine_l1762_176215

open Real

theorem geometric_sequence_cosine (a : ℝ) : 
  0 < a → a < 2 * π → 
  (∃ r : ℝ, cos a * r = cos (2 * a) ∧ cos (2 * a) * r = cos (3 * a)) → 
  a = π := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_cosine_l1762_176215


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l1762_176228

/-- Given two planar vectors a and b satisfying certain conditions, 
    prove that the cosine of the angle between them is -√10/10 -/
theorem cosine_of_angle_between_vectors (a b : ℝ × ℝ) : 
  (2 • a + b = (3, 3)) → 
  (a - 2 • b = (-1, 4)) → 
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.cos θ = -Real.sqrt 10 / 10 := by
sorry

end NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l1762_176228


namespace NUMINAMATH_CALUDE_lines_parallel_l1762_176226

/-- The value of k that makes the given lines parallel -/
def k : ℚ := 16/5

/-- The first line's direction vector -/
def v1 : Fin 2 → ℚ := ![5, -8]

/-- The second line's direction vector -/
def v2 : Fin 2 → ℚ := ![-2, k]

/-- Theorem stating that k makes the lines parallel -/
theorem lines_parallel : ∃ (c : ℚ), v1 = c • v2 := by sorry

end NUMINAMATH_CALUDE_lines_parallel_l1762_176226


namespace NUMINAMATH_CALUDE_valid_arrangements_l1762_176246

/-- The number of letters to be arranged -/
def n : ℕ := 8

/-- The number of pairs of repeated letters -/
def k : ℕ := 3

/-- The total number of unrestricted arrangements -/
def total_arrangements : ℕ := n.factorial / (2^k)

/-- The number of arrangements with one pair of identical letters together -/
def arrangements_one_pair : ℕ := k * ((n-1).factorial / (2^(k-1)))

/-- The number of arrangements with two pairs of identical letters together -/
def arrangements_two_pairs : ℕ := (k.choose 2) * ((n-2).factorial / (2^(k-2)))

/-- The number of arrangements with three pairs of identical letters together -/
def arrangements_three_pairs : ℕ := (n-3).factorial

/-- The theorem stating the number of valid arrangements -/
theorem valid_arrangements :
  total_arrangements - arrangements_one_pair + arrangements_two_pairs - arrangements_three_pairs = 2220 :=
sorry

end NUMINAMATH_CALUDE_valid_arrangements_l1762_176246


namespace NUMINAMATH_CALUDE_cat_weight_difference_l1762_176264

/-- Given the weights of two cats belonging to Meg and Anne, prove the weight difference --/
theorem cat_weight_difference 
  (weight_meg : ℝ) 
  (weight_anne : ℝ) 
  (h1 : weight_meg / weight_anne = 13 / 21)
  (h2 : weight_meg = 20 + 0.5 * weight_anne) :
  weight_anne - weight_meg = 64 := by
  sorry

end NUMINAMATH_CALUDE_cat_weight_difference_l1762_176264


namespace NUMINAMATH_CALUDE_completing_square_transformation_l1762_176212

theorem completing_square_transformation (x : ℝ) :
  x^2 - 8*x - 11 = 0 ↔ (x - 4)^2 = 27 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l1762_176212


namespace NUMINAMATH_CALUDE_prize_problem_solution_l1762_176219

/-- Represents the prices and quantities of notebooks and pens -/
structure PrizeInfo where
  notebook_price : ℕ
  pen_price : ℕ
  notebook_quantity : ℕ
  pen_quantity : ℕ

/-- Theorem stating the solution to the prize problem -/
theorem prize_problem_solution :
  ∃ (info : PrizeInfo),
    -- Each notebook costs 3 yuan more than each pen
    info.notebook_price = info.pen_price + 3 ∧
    -- The number of notebooks purchased for 390 yuan is the same as the number of pens purchased for 300 yuan
    390 / info.notebook_price = 300 / info.pen_price ∧
    -- The total cost of purchasing prizes for 50 students should not exceed 560 yuan
    info.notebook_quantity + info.pen_quantity = 50 ∧
    info.notebook_price * info.notebook_quantity + info.pen_price * info.pen_quantity ≤ 560 ∧
    -- The notebook price is 13 yuan
    info.notebook_price = 13 ∧
    -- The pen price is 10 yuan
    info.pen_price = 10 ∧
    -- The maximum number of notebooks that can be purchased is 20
    info.notebook_quantity = 20 ∧
    -- This is the maximum possible number of notebooks
    ∀ (other_info : PrizeInfo),
      other_info.notebook_price = other_info.pen_price + 3 →
      other_info.notebook_quantity + other_info.pen_quantity = 50 →
      other_info.notebook_price * other_info.notebook_quantity + other_info.pen_price * other_info.pen_quantity ≤ 560 →
      other_info.notebook_quantity ≤ info.notebook_quantity :=
by
  sorry


end NUMINAMATH_CALUDE_prize_problem_solution_l1762_176219


namespace NUMINAMATH_CALUDE_bankers_gain_example_l1762_176292

/-- Calculates the banker's gain given the present worth, interest rate, and time period. -/
def bankers_gain (present_worth : ℝ) (interest_rate : ℝ) (time_period : ℕ) : ℝ :=
  let amount_due := present_worth * (1 + interest_rate) ^ time_period
  amount_due - present_worth

/-- The banker's gain for a present worth of 400, interest rate of 10%, and time period of 3 years is 132.4. -/
theorem bankers_gain_example : 
  ∃ ε > 0, |bankers_gain 400 0.1 3 - 132.4| < ε :=
sorry

end NUMINAMATH_CALUDE_bankers_gain_example_l1762_176292


namespace NUMINAMATH_CALUDE_restaurant_problem_l1762_176230

theorem restaurant_problem (people : ℕ) 
  (h1 : 7 * 10 + (88 / people + 7) = 88) : people = 8 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_problem_l1762_176230


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l1762_176207

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l1762_176207


namespace NUMINAMATH_CALUDE_fraction_product_equals_one_fourteenth_l1762_176252

def product_fraction (n : ℕ) : ℚ := (n^2 - 1) / (n^2 + 1)

theorem fraction_product_equals_one_fourteenth :
  (product_fraction 2) * (product_fraction 3) * (product_fraction 4) * 
  (product_fraction 5) * (product_fraction 6) = 1 / 14 := by
sorry

end NUMINAMATH_CALUDE_fraction_product_equals_one_fourteenth_l1762_176252


namespace NUMINAMATH_CALUDE_square_inequality_equivalence_l1762_176244

theorem square_inequality_equivalence (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a > b ↔ a^2 > b^2 := by sorry

end NUMINAMATH_CALUDE_square_inequality_equivalence_l1762_176244


namespace NUMINAMATH_CALUDE_inverse_of_matrix_A_l1762_176213

theorem inverse_of_matrix_A :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![3, 4; 1, 2]
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![1, -2; -1/2, 3/2]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_matrix_A_l1762_176213


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1762_176284

-- Define the original inequality
def original_inequality (x : ℝ) : Prop := 2 * x^2 - 5*x - 3 ≥ 0

-- Define the solution to the original inequality
def solution_inequality (x : ℝ) : Prop := x ≤ -1/2 ∨ x ≥ 3

-- Define the proposed necessary but not sufficient condition
def proposed_condition (x : ℝ) : Prop := x < -1 ∨ x > 4

-- State the theorem
theorem necessary_but_not_sufficient :
  (∀ x : ℝ, original_inequality x ↔ solution_inequality x) →
  (∀ x : ℝ, solution_inequality x → proposed_condition x) ∧
  ¬(∀ x : ℝ, proposed_condition x → solution_inequality x) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1762_176284


namespace NUMINAMATH_CALUDE_domain_of_g_l1762_176224

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-12) 3

-- Define the function g in terms of f
def g (x : ℝ) : ℝ := f (3 * x)

-- State the theorem
theorem domain_of_g : 
  {x : ℝ | g x ∈ Set.range f} = Set.Icc (-4) 1 := by sorry

end NUMINAMATH_CALUDE_domain_of_g_l1762_176224


namespace NUMINAMATH_CALUDE_parallel_line_plane_false_l1762_176216

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- Defines when a line is parallel to a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when a line is in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when two lines are parallel -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

theorem parallel_line_plane_false : 
  ¬ (∀ (α : Plane3D) (b : Line3D), 
    parallel_line_plane b α → 
    (∀ (a : Line3D), line_in_plane a α → parallel_lines b a)) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_plane_false_l1762_176216


namespace NUMINAMATH_CALUDE_average_difference_theorem_l1762_176247

/-- The average number of students per teacher -/
def t (total_students : ℕ) (num_teachers : ℕ) : ℚ :=
  total_students / num_teachers

/-- The average number of students per student -/
def s (class_sizes : List ℕ) (total_students : ℕ) : ℚ :=
  (class_sizes.map (λ size => size * (size : ℚ) / total_students)).sum

theorem average_difference_theorem (total_students : ℕ) (num_teachers : ℕ) (class_sizes : List ℕ) :
  total_students = 120 →
  num_teachers = 5 →
  class_sizes = [60, 30, 20, 5, 5] →
  t total_students num_teachers - s class_sizes total_students = -17.25 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_theorem_l1762_176247


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1762_176297

-- Define the quadratic function f(x)
def f (x : ℝ) := 2 * x^2 - 10 * x

-- State the theorem
theorem quadratic_function_properties :
  (∀ x ∈ Set.Icc (-1 : ℝ) 4, f x ≤ 12) ∧  -- maximum value is 12 on [-1,4]
  (∀ x : ℝ, f x < 0 ↔ x ∈ Set.Ioo 0 5) ∧  -- solution set of f(x) < 0 is (0,5)
  (∀ x m : ℝ, m < -5 ∨ m > 1 → f (2 - 2 * Real.cos x) < f (1 - Real.cos x - m)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1762_176297


namespace NUMINAMATH_CALUDE_trail_mix_weight_l1762_176261

theorem trail_mix_weight : 
  let peanuts : ℚ := 0.16666666666666666
  let chocolate_chips : ℚ := 0.16666666666666666
  let raisins : ℚ := 0.08333333333333333
  let almonds : ℚ := 0.14583333333333331
  let cashews : ℚ := 1/8
  let dried_cranberries : ℚ := 3/32
  peanuts + chocolate_chips + raisins + almonds + cashews + dried_cranberries = 0.78125 := by
  sorry

end NUMINAMATH_CALUDE_trail_mix_weight_l1762_176261


namespace NUMINAMATH_CALUDE_interest_rate_equality_l1762_176270

theorem interest_rate_equality (I : ℝ) (r : ℝ) : 
  I = 1000 * 0.12 * 2 → 
  I = 200 * r * 12 → 
  r = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_equality_l1762_176270


namespace NUMINAMATH_CALUDE_man_rowing_downstream_speed_l1762_176271

/-- The speed of a man rowing downstream, given his speed in still water and the speed of the stream. -/
def speed_downstream (speed_still_water : ℝ) (speed_stream : ℝ) : ℝ :=
  speed_still_water + speed_stream

/-- Theorem: The speed of the man rowing downstream is 18 kmph. -/
theorem man_rowing_downstream_speed :
  let speed_still_water : ℝ := 12
  let speed_stream : ℝ := 6
  speed_downstream speed_still_water speed_stream = 18 := by
  sorry

end NUMINAMATH_CALUDE_man_rowing_downstream_speed_l1762_176271


namespace NUMINAMATH_CALUDE_complex_equality_implies_real_value_l1762_176251

theorem complex_equality_implies_real_value (a : ℝ) : 
  (Complex.re ((1 + 2*Complex.I) * (a + Complex.I)) = Complex.im ((1 + 2*Complex.I) * (a + Complex.I))) → 
  a = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_implies_real_value_l1762_176251


namespace NUMINAMATH_CALUDE_sum_of_distinct_integers_l1762_176242

theorem sum_of_distinct_integers (p q r s t : ℤ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ 
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ 
  r ≠ s ∧ r ≠ t ∧ 
  s ≠ t → 
  (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 120 →
  p + q + r + s + t = 25 := by
sorry

end NUMINAMATH_CALUDE_sum_of_distinct_integers_l1762_176242


namespace NUMINAMATH_CALUDE_quadrilateral_interior_angles_mean_l1762_176214

/-- The mean value of the measures of the four interior angles of any quadrilateral is 90°. -/
theorem quadrilateral_interior_angles_mean :
  let sum_of_angles : ℝ := 360
  let number_of_angles : ℕ := 4
  (sum_of_angles / number_of_angles : ℝ) = 90 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_interior_angles_mean_l1762_176214


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1762_176206

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotone_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

theorem solution_set_of_inequality (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_monotone : is_monotone_increasing_on_nonneg f) :
  {a : ℝ | f 1 < f a} = {a : ℝ | a < -1 ∨ 1 < a} :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1762_176206


namespace NUMINAMATH_CALUDE_alloy_mixture_percentage_l1762_176282

/-- Proves that mixing 66 ounces of 10% alloy with 55 ounces of 21% alloy
    results in 121 ounces of an alloy with 15% copper content. -/
theorem alloy_mixture_percentage :
  let alloy_10_amount : ℝ := 66
  let alloy_10_percentage : ℝ := 10
  let alloy_21_amount : ℝ := 55
  let alloy_21_percentage : ℝ := 21
  let total_amount : ℝ := alloy_10_amount + alloy_21_amount
  let total_copper : ℝ := (alloy_10_amount * alloy_10_percentage / 100) +
                          (alloy_21_amount * alloy_21_percentage / 100)
  let final_percentage : ℝ := total_copper / total_amount * 100
  total_amount = 121 ∧ final_percentage = 15 := by sorry

end NUMINAMATH_CALUDE_alloy_mixture_percentage_l1762_176282


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l1762_176269

/-- The area of a square with a perimeter of 40 meters is 100 square meters. -/
theorem square_area_from_perimeter :
  ∀ (side : ℝ), 
  (4 * side = 40) →  -- perimeter is 40 meters
  (side * side = 100) -- area is 100 square meters
:= by sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l1762_176269


namespace NUMINAMATH_CALUDE_other_number_is_25_l1762_176265

theorem other_number_is_25 (x y : ℤ) : 
  (3 * x + 4 * y + 2 * x = 160) → 
  ((x = 12 ∧ y ≠ 12) ∨ (y = 12 ∧ x ≠ 12)) → 
  (x = 25 ∨ y = 25) := by
sorry

end NUMINAMATH_CALUDE_other_number_is_25_l1762_176265


namespace NUMINAMATH_CALUDE_angle_ABC_equals_cos_inverse_l1762_176222

/-- The angle ABC given three points A, B, and C in 3D space -/
def angle_ABC (A B C : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Converts radians to degrees -/
def to_degrees (x : ℝ) : ℝ := sorry

theorem angle_ABC_equals_cos_inverse :
  let A : ℝ × ℝ × ℝ := (-3, 1, 5)
  let B : ℝ × ℝ × ℝ := (-4, -2, 1)
  let C : ℝ × ℝ × ℝ := (-5, -2, 2)
  to_degrees (angle_ABC A B C) = Real.arccos ((3 * Real.sqrt 13) / 26) := by sorry

end NUMINAMATH_CALUDE_angle_ABC_equals_cos_inverse_l1762_176222


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_on_negative_l1762_176208

def f (x : ℝ) := -x^2

theorem f_is_even_and_increasing_on_negative : 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_on_negative_l1762_176208


namespace NUMINAMATH_CALUDE_no_solution_exists_l1762_176275

theorem no_solution_exists : 
  ¬∃ (a b : ℕ+), a * b + 82 = 25 * Nat.lcm a b + 15 * Nat.gcd a b :=
sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1762_176275


namespace NUMINAMATH_CALUDE_counterexample_exists_l1762_176283

theorem counterexample_exists :
  ∃ (n : ℕ), n ≥ 2 ∧ ¬(∃ (k : ℕ), (2^(2^n) % (2^n - 1)) = 4^k) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1762_176283


namespace NUMINAMATH_CALUDE_pencils_and_pens_count_pencils_and_pens_count_proof_l1762_176234

theorem pencils_and_pens_count : ℕ → ℕ → Prop :=
  fun initial_pencils initial_pens =>
    (initial_pencils : ℚ) / initial_pens = 4 / 5 ∧
    ((initial_pencils + 1 : ℚ) / (initial_pens - 1) = 7 / 8) →
    initial_pencils + initial_pens = 45

-- The proof goes here
theorem pencils_and_pens_count_proof : ∃ (p q : ℕ), pencils_and_pens_count p q :=
  sorry

end NUMINAMATH_CALUDE_pencils_and_pens_count_pencils_and_pens_count_proof_l1762_176234


namespace NUMINAMATH_CALUDE_valid_configuration_iff_n_eq_4_l1762_176237

/-- A configuration of n points in the plane with associated real numbers -/
structure PointConfiguration (n : ℕ) where
  points : Fin n → ℝ × ℝ
  values : Fin n → ℝ

/-- The area of a triangle formed by three points -/
def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

/-- The condition that no three points are collinear -/
def noThreeCollinear (config : PointConfiguration n) : Prop :=
  ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k →
    triangleArea (config.points i) (config.points j) (config.points k) ≠ 0

/-- The condition that the area of any triangle equals the sum of corresponding values -/
def areaEqualsSumOfValues (config : PointConfiguration n) : Prop :=
  ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k →
    triangleArea (config.points i) (config.points j) (config.points k) =
      config.values i + config.values j + config.values k

/-- The main theorem stating that a valid configuration exists if and only if n = 4 -/
theorem valid_configuration_iff_n_eq_4 :
  (∃ (config : PointConfiguration n), n > 3 ∧ noThreeCollinear config ∧ areaEqualsSumOfValues config) ↔
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_valid_configuration_iff_n_eq_4_l1762_176237


namespace NUMINAMATH_CALUDE_second_frog_hops_l1762_176239

/-- Represents the number of hops taken by each frog -/
structure FrogHops :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)
  (fourth : ℕ)

/-- The conditions of the frog hopping problem -/
def frog_problem (hops : FrogHops) : Prop :=
  hops.first = 4 * hops.second ∧
  hops.second = 2 * hops.third ∧
  hops.fourth = 3 * hops.second ∧
  hops.first + hops.second + hops.third + hops.fourth = 156 ∧
  60 ≤ 120  -- represents the time constraint (60 meters in 2 minutes or less)

/-- The theorem stating that the second frog takes 18 hops -/
theorem second_frog_hops :
  ∃ (hops : FrogHops), frog_problem hops ∧ hops.second = 18 :=
by sorry

end NUMINAMATH_CALUDE_second_frog_hops_l1762_176239


namespace NUMINAMATH_CALUDE_no_sum_of_three_different_squares_128_l1762_176220

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def sum_of_three_different_squares (n : ℕ) : Prop :=
  ∃ a b c : ℕ, 
    is_perfect_square a ∧ 
    is_perfect_square b ∧ 
    is_perfect_square c ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = n

theorem no_sum_of_three_different_squares_128 : 
  ¬(sum_of_three_different_squares 128) := by
  sorry

end NUMINAMATH_CALUDE_no_sum_of_three_different_squares_128_l1762_176220


namespace NUMINAMATH_CALUDE_john_spent_625_l1762_176223

/-- The amount John spent on purchases with a coupon -/
def johnsSpending (vacuumPrice dishwasherPrice couponValue : ℕ) : ℕ :=
  vacuumPrice + dishwasherPrice - couponValue

/-- Theorem stating that John spent $625 -/
theorem john_spent_625 :
  johnsSpending 250 450 75 = 625 := by
  sorry

end NUMINAMATH_CALUDE_john_spent_625_l1762_176223


namespace NUMINAMATH_CALUDE_det_A_eq_140_l1762_176200

def A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 1, -2; 8, 5, -4; 1, 3, 6]

theorem det_A_eq_140 : Matrix.det A = 140 := by sorry

end NUMINAMATH_CALUDE_det_A_eq_140_l1762_176200


namespace NUMINAMATH_CALUDE_sum_of_squares_l1762_176254

theorem sum_of_squares (a b c : ℝ) 
  (h1 : a * b + b * c + a * c = 131) 
  (h2 : a + b + c = 22) : 
  a^2 + b^2 + c^2 = 222 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1762_176254


namespace NUMINAMATH_CALUDE_store_discount_is_ten_percent_l1762_176289

/-- Calculates the discount percentage given the number of items, cost per item, 
    discount threshold, and final cost after discount. -/
def discount_percentage (num_items : ℕ) (cost_per_item : ℚ) 
  (discount_threshold : ℚ) (final_cost : ℚ) : ℚ :=
  let total_cost := num_items * cost_per_item
  let discount_amount := total_cost - final_cost
  let eligible_amount := total_cost - discount_threshold
  (discount_amount / eligible_amount) * 100

/-- Proves that the discount percentage is 10% for the given scenario. -/
theorem store_discount_is_ten_percent :
  discount_percentage 7 200 1000 1360 = 10 := by
  sorry

end NUMINAMATH_CALUDE_store_discount_is_ten_percent_l1762_176289


namespace NUMINAMATH_CALUDE_factorization_equality_l1762_176293

theorem factorization_equality (a b : ℝ) : a^2 - 4*a*b^2 = a*(a - 4*b^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1762_176293


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l1762_176285

theorem greatest_integer_satisfying_inequality :
  ∀ x : ℕ+, x ≤ 4 ↔ (x : ℝ)^4 / (x : ℝ)^2 < 18 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l1762_176285


namespace NUMINAMATH_CALUDE_circle_ratio_l1762_176229

theorem circle_ratio (r R a b : ℝ) (hr : r > 0) (hR : R > r) (ha : a > 0) (hb : b > 0) 
  (h : π * R^2 = (a / b) * (π * R^2 - π * r^2)) :
  R / r = Real.sqrt a / Real.sqrt (a - b) := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_l1762_176229


namespace NUMINAMATH_CALUDE_star_polygon_angle_sum_l1762_176291

/-- A star polygon created from a regular n-gon --/
structure StarPolygon where
  n : ℕ
  n_ge_6 : n ≥ 6

/-- The sum of interior angles of a star polygon --/
def sum_interior_angles (s : StarPolygon) : ℝ :=
  180 * (s.n - 2)

/-- Theorem: The sum of interior angles of a star polygon is 180°(n-2) --/
theorem star_polygon_angle_sum (s : StarPolygon) :
  sum_interior_angles s = 180 * (s.n - 2) :=
by sorry

end NUMINAMATH_CALUDE_star_polygon_angle_sum_l1762_176291


namespace NUMINAMATH_CALUDE_min_dot_product_l1762_176258

def OA : ℝ × ℝ := (2, 2)
def OB : ℝ × ℝ := (4, 1)

def AP (x : ℝ) : ℝ × ℝ := (x - OA.1, -OA.2)
def BP (x : ℝ) : ℝ × ℝ := (x - OB.1, -OB.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem min_dot_product :
  ∃ (x : ℝ), ∀ (y : ℝ),
    dot_product (AP x) (BP x) ≤ dot_product (AP y) (BP y) ∧
    x = 3 :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_l1762_176258


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l1762_176211

theorem smallest_number_with_conditions : ∃ x : ℕ, 
  (∃ k : ℤ, (x : ℤ) + 3 = 7 * k) ∧ 
  (∃ m : ℤ, (x : ℤ) - 5 = 8 * m) ∧ 
  (∀ y : ℕ, y < x → ¬((∃ k : ℤ, (y : ℤ) + 3 = 7 * k) ∧ (∃ m : ℤ, (y : ℤ) - 5 = 8 * m))) ∧
  x = 53 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l1762_176211


namespace NUMINAMATH_CALUDE_jenny_mike_earnings_l1762_176227

theorem jenny_mike_earnings (t : ℝ) : 
  (t + 3) * (4 * t - 6) = (4 * t - 7) * (t + 3) + 3 → t = 3 := by
  sorry

end NUMINAMATH_CALUDE_jenny_mike_earnings_l1762_176227


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1762_176240

theorem min_reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (1 / a + 1 / b + 1 / c) ≥ 3 := by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1762_176240


namespace NUMINAMATH_CALUDE_min_value_expression_l1762_176257

theorem min_value_expression (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  (x^2 / (y - 1) + y^2 / (z - 1) + z^2 / (x - 1)) ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1762_176257


namespace NUMINAMATH_CALUDE_total_pupils_across_schools_l1762_176225

/-- The total number of pupils across three schools -/
def total_pupils (school_a_girls school_a_boys school_b_girls school_b_boys school_c_girls school_c_boys : ℕ) : ℕ :=
  school_a_girls + school_a_boys + school_b_girls + school_b_boys + school_c_girls + school_c_boys

/-- Theorem stating that the total number of pupils across the three schools is 3120 -/
theorem total_pupils_across_schools :
  total_pupils 542 387 713 489 628 361 = 3120 := by
  sorry

end NUMINAMATH_CALUDE_total_pupils_across_schools_l1762_176225


namespace NUMINAMATH_CALUDE_quadratic_one_zero_properties_l1762_176280

/-- A quadratic function with exactly one zero -/
structure QuadraticWithOneZero where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : ∃! x, x^2 + a*x + b = 0

theorem quadratic_one_zero_properties (f : QuadraticWithOneZero) :
  (f.a^2 - f.b^2 ≤ 4) ∧
  (f.a^2 + 1/f.b ≥ 4) ∧
  (∀ c x₁ x₂, (∀ x, x^2 + f.a*x + f.b < c ↔ x₁ < x ∧ x < x₂) → |x₁ - x₂| = 4 → c = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_zero_properties_l1762_176280


namespace NUMINAMATH_CALUDE_apple_cost_price_l1762_176202

theorem apple_cost_price (selling_price : ℝ) (loss_fraction : ℝ) (cost_price : ℝ) : 
  selling_price = 19 →
  loss_fraction = 1/6 →
  selling_price = cost_price - loss_fraction * cost_price →
  cost_price = 22.8 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_price_l1762_176202


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l1762_176201

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- Theorem stating that 3(2-i) + i(3+2i) = 4 -/
theorem simplify_complex_expression : 3 * (2 - i) + i * (3 + 2 * i) = (4 : ℂ) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l1762_176201


namespace NUMINAMATH_CALUDE_cubic_polynomial_theorem_l1762_176272

/-- Represents a cubic polynomial a₃x³ - x² + a₁x - 7 = 0 -/
structure CubicPolynomial where
  a₃ : ℝ
  a₁ : ℝ

/-- Represents the roots of the cubic polynomial -/
structure Roots where
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Checks if the given roots satisfy the condition -/
def satisfiesCondition (r : Roots) : Prop :=
  (225 * r.α^2) / (r.α^2 + 7) = (144 * r.β^2) / (r.β^2 + 7) ∧
  (144 * r.β^2) / (r.β^2 + 7) = (100 * r.γ^2) / (r.γ^2 + 7)

/-- Checks if the given roots are positive -/
def arePositive (r : Roots) : Prop :=
  r.α > 0 ∧ r.β > 0 ∧ r.γ > 0

/-- Checks if the given roots are valid for the cubic polynomial -/
def areValidRoots (p : CubicPolynomial) (r : Roots) : Prop :=
  p.a₃ * r.α^3 - r.α^2 + p.a₁ * r.α - 7 = 0 ∧
  p.a₃ * r.β^3 - r.β^2 + p.a₁ * r.β - 7 = 0 ∧
  p.a₃ * r.γ^3 - r.γ^2 + p.a₁ * r.γ - 7 = 0

theorem cubic_polynomial_theorem (p : CubicPolynomial) (r : Roots) 
  (h1 : satisfiesCondition r)
  (h2 : arePositive r)
  (h3 : areValidRoots p r) :
  abs (p.a₁ - 130.6667) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_theorem_l1762_176272


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l1762_176245

-- Problem 1
theorem problem_one : -1^4 - 7 / (2 - (-3)^2) = 0 := by sorry

-- Problem 2
-- Define a custom type for degrees and minutes
structure DegreeMinute where
  degrees : Int
  minutes : Int

-- Define addition for DegreeMinute
def add_degree_minute (a b : DegreeMinute) : DegreeMinute :=
  let total_minutes := a.minutes + b.minutes
  let extra_degrees := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  ⟨a.degrees + b.degrees + extra_degrees, remaining_minutes⟩

-- Define subtraction for DegreeMinute
def sub_degree_minute (a b : DegreeMinute) : DegreeMinute :=
  let total_minutes_a := a.degrees * 60 + a.minutes
  let total_minutes_b := b.degrees * 60 + b.minutes
  let diff_minutes := total_minutes_a - total_minutes_b
  ⟨diff_minutes / 60, diff_minutes % 60⟩

-- Define multiplication of DegreeMinute by Int
def mul_degree_minute (a : DegreeMinute) (n : Int) : DegreeMinute :=
  let total_minutes := (a.degrees * 60 + a.minutes) * n
  ⟨total_minutes / 60, total_minutes % 60⟩

theorem problem_two :
  sub_degree_minute
    (add_degree_minute ⟨56, 17⟩ ⟨12, 45⟩)
    (mul_degree_minute ⟨16, 21⟩ 4) = ⟨3, 38⟩ := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l1762_176245


namespace NUMINAMATH_CALUDE_conference_handshakes_l1762_176290

/-- The number of handshakes in a conference with n participants -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The theorem stating that a conference with 10 participants results in 45 handshakes -/
theorem conference_handshakes : handshakes 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1762_176290


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l1762_176296

theorem sum_of_reciprocal_equations (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : 1/x - 1/y = -1) : 
  x + y = 5/6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l1762_176296


namespace NUMINAMATH_CALUDE_inequality_solution_l1762_176274

theorem inequality_solution (x : ℝ) : 
  (x^2 - 4) / (x^2 - 1) > 0 ↔ x > 2 ∨ x < -2 ∨ (-1 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1762_176274


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l1762_176288

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_unique :
  ∀ a b c : ℝ,
  (f a b c (-1) = 0) →
  (∀ x : ℝ, x ≤ f a b c x) →
  (∀ x : ℝ, f a b c x ≤ (1 + x^2) / 2) →
  (a = 1/4 ∧ b = 1/2 ∧ c = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l1762_176288


namespace NUMINAMATH_CALUDE_nara_height_l1762_176221

/-- Given the heights of Sangheon, Chiho, and Nara, prove Nara's height -/
theorem nara_height (sangheon_height : ℝ) (chiho_diff : ℝ) (nara_diff : ℝ) :
  sangheon_height = 1.56 →
  chiho_diff = 0.14 →
  nara_diff = 0.27 →
  sangheon_height - chiho_diff + nara_diff = 1.69 := by
sorry

end NUMINAMATH_CALUDE_nara_height_l1762_176221


namespace NUMINAMATH_CALUDE_alloy_composition_l1762_176218

theorem alloy_composition (m₁ m₂ m₃ m₄ : ℝ) 
  (total_mass : m₁ + m₂ + m₃ + m₄ = 20)
  (first_second_relation : m₁ = 1.5 * m₂)
  (second_third_ratio : m₂ = (3/4) * m₃)
  (third_fourth_ratio : m₃ = (5/6) * m₄) :
  m₄ = 960 / 123 := by
  sorry

end NUMINAMATH_CALUDE_alloy_composition_l1762_176218


namespace NUMINAMATH_CALUDE_sams_initial_dimes_l1762_176299

/-- The problem of determining Sam's initial number of dimes -/
theorem sams_initial_dimes : 
  ∀ (initial final given : ℕ), 
  given = 7 →                 -- Sam's dad gave him 7 dimes
  final = 16 →                -- After receiving the dimes, Sam has 16 dimes
  final = initial + given →   -- The final amount is the sum of initial and given
  initial = 9 :=              -- Prove that the initial amount was 9 dimes
by sorry

end NUMINAMATH_CALUDE_sams_initial_dimes_l1762_176299


namespace NUMINAMATH_CALUDE_problem_solution_l1762_176256

-- Define the complex square root function
noncomputable def complexSqrt (x : ℂ) : ℂ := sorry

-- Define the statements
def statement_I : Prop :=
  complexSqrt (-4) * complexSqrt (-16) = complexSqrt ((-4) * (-16))

def statement_II : Prop :=
  complexSqrt ((-4) * (-16)) = Real.sqrt 64

def statement_III : Prop :=
  Real.sqrt 64 = 8

-- Theorem to prove
theorem problem_solution :
  (¬statement_I ∧ statement_II ∧ statement_III) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1762_176256


namespace NUMINAMATH_CALUDE_military_unit_march_speeds_l1762_176298

/-- Proves that given the conditions of the military unit's march, the average speeds on the first and second days are 12 km/h and 10 km/h respectively. -/
theorem military_unit_march_speeds :
  ∀ (speed_day1 speed_day2 : ℝ),
    4 * speed_day1 + 5 * speed_day2 = 98 →
    4 * speed_day1 = 5 * speed_day2 - 2 →
    speed_day1 = 12 ∧ speed_day2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_military_unit_march_speeds_l1762_176298


namespace NUMINAMATH_CALUDE_system_solution_pairs_l1762_176294

theorem system_solution_pairs :
  ∀ x y : ℝ, x > 0 ∧ y > 0 →
  (x - 3 * Real.sqrt (x * y) - 2 * Real.sqrt (x / y) = 0 ∧
   x^2 * y^2 + x^4 = 82) →
  ((x = 3 ∧ y = 1/3) ∨ (x = Real.rpow 66 (1/4) ∧ y = 4 / Real.rpow 66 (1/4))) := by
sorry

end NUMINAMATH_CALUDE_system_solution_pairs_l1762_176294


namespace NUMINAMATH_CALUDE_concert_ticket_revenue_l1762_176278

/-- Calculates the total revenue from concert ticket sales given specific discount conditions -/
theorem concert_ticket_revenue : 
  let ticket_price : ℝ := 20
  let first_group_size : ℕ := 10
  let second_group_size : ℕ := 20
  let first_discount : ℝ := 0.4
  let second_discount : ℝ := 0.15
  let total_attendees : ℕ := 48

  let first_group_revenue := first_group_size * (ticket_price * (1 - first_discount))
  let second_group_revenue := second_group_size * (ticket_price * (1 - second_discount))
  let remaining_attendees := total_attendees - first_group_size - second_group_size
  let full_price_revenue := remaining_attendees * ticket_price

  first_group_revenue + second_group_revenue + full_price_revenue = 820 :=
by
  sorry


end NUMINAMATH_CALUDE_concert_ticket_revenue_l1762_176278


namespace NUMINAMATH_CALUDE_everton_calculator_count_l1762_176295

/-- Represents the order of calculators by Everton college -/
structure CalculatorOrder where
  totalCost : ℕ
  scientificCost : ℕ
  graphingCost : ℕ
  scientificCount : ℕ

/-- Calculates the total number of calculators in an order -/
def totalCalculators (order : CalculatorOrder) : ℕ :=
  let graphingCount := (order.totalCost - order.scientificCount * order.scientificCost) / order.graphingCost
  order.scientificCount + graphingCount

/-- Theorem: The total number of calculators in Everton college's order is 45 -/
theorem everton_calculator_count :
  let order : CalculatorOrder := {
    totalCost := 1625,
    scientificCost := 10,
    graphingCost := 57,
    scientificCount := 20
  }
  totalCalculators order = 45 := by
  sorry

end NUMINAMATH_CALUDE_everton_calculator_count_l1762_176295


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1762_176231

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 6 + a 5 = 4 →
  a 4 + a 3 - a 2 - a 1 = 1 →
  a 1 = Real.sqrt 2 - 1 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l1762_176231


namespace NUMINAMATH_CALUDE_salary_D_value_l1762_176259

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_E : ℕ := 9000
def average_salary : ℕ := 8000
def num_people : ℕ := 5

theorem salary_D_value :
  ∃ (salary_D : ℕ),
    (salary_A + salary_B + salary_C + salary_D + salary_E) / num_people = average_salary ∧
    salary_D = 9000 := by
  sorry

end NUMINAMATH_CALUDE_salary_D_value_l1762_176259


namespace NUMINAMATH_CALUDE_min_total_cost_l1762_176262

/-- Represents a salon with prices for haircut, facial cleaning, and nails -/
structure Salon where
  name : String
  haircut : ℕ
  facial : ℕ
  nails : ℕ

/-- Calculates the total cost of services at a salon -/
def totalCost (s : Salon) : ℕ := s.haircut + s.facial + s.nails

/-- The list of salons with their prices -/
def salonList : List Salon := [
  { name := "Gustran Salon", haircut := 45, facial := 22, nails := 30 },
  { name := "Barbara's Shop", haircut := 30, facial := 28, nails := 40 },
  { name := "The Fancy Salon", haircut := 34, facial := 30, nails := 20 }
]

/-- Theorem: The minimum total cost among the salons is 84 -/
theorem min_total_cost : 
  (salonList.map totalCost).minimum? = some 84 := by
  sorry

end NUMINAMATH_CALUDE_min_total_cost_l1762_176262


namespace NUMINAMATH_CALUDE_log_ratio_squared_l1762_176236

theorem log_ratio_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1)
  (h1 : Real.log x / Real.log 3 = Real.log 81 / Real.log x)
  (h2 : x * y = 27) :
  ((Real.log x - Real.log y) / Real.log 3) ^ 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l1762_176236


namespace NUMINAMATH_CALUDE_unique_six_digit_reverse_multiple_l1762_176241

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.reverse.foldl (λ acc d => acc * 10 + d) 0

theorem unique_six_digit_reverse_multiple : 
  ∃! n : ℕ, is_six_digit n ∧ n * 9 = reverse_digits n :=
by sorry

end NUMINAMATH_CALUDE_unique_six_digit_reverse_multiple_l1762_176241
