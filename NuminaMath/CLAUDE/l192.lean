import Mathlib

namespace NUMINAMATH_CALUDE_line_through_point_l192_19231

theorem line_through_point (a : ℝ) (h : a ≠ 0) :
  (∀ x y : ℝ, y = 5 * x + a → (x = a → y = a^2)) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l192_19231


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l192_19239

theorem inequality_and_equality_conditions (a b : ℝ) :
  (a^2 + b^2 - a - b - a*b + 0.25 ≥ 0) ∧
  (a^2 + b^2 - a - b - a*b + 0.25 = 0 ↔ (a = 0 ∧ b = 0.5) ∨ (a = 0.5 ∧ b = 0)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l192_19239


namespace NUMINAMATH_CALUDE_math_grade_calculation_l192_19256

theorem math_grade_calculation (history_grade third_subject_grade : ℝ) 
  (h1 : history_grade = 84)
  (h2 : third_subject_grade = 67)
  (h3 : (math_grade + history_grade + third_subject_grade) / 3 = 75) :
  math_grade = 74 := by
sorry

end NUMINAMATH_CALUDE_math_grade_calculation_l192_19256


namespace NUMINAMATH_CALUDE_complex_absolute_value_l192_19203

theorem complex_absolute_value : Complex.abs ((2 : ℂ) + (2 * Complex.I * Real.sqrt 2)) ^ 6 = 576 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l192_19203


namespace NUMINAMATH_CALUDE_square_minus_eight_equals_power_of_three_l192_19250

theorem square_minus_eight_equals_power_of_three (b n : ℕ) :
  b^2 - 8 = 3^n ↔ b = 3 ∧ n = 0 := by sorry

end NUMINAMATH_CALUDE_square_minus_eight_equals_power_of_three_l192_19250


namespace NUMINAMATH_CALUDE_cosine_product_eleven_l192_19255

theorem cosine_product_eleven : 
  Real.cos (π / 11) * Real.cos (2 * π / 11) * Real.cos (3 * π / 11) * 
  Real.cos (4 * π / 11) * Real.cos (5 * π / 11) = 1 / 32 :=
by sorry

end NUMINAMATH_CALUDE_cosine_product_eleven_l192_19255


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l192_19276

/-- An arithmetic sequence with given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  a_3_eq_9 : a 3 = 9
  S_3_eq_33 : (a 1 + a 2 + a 3) = 33

/-- Properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  ∃ (d : ℝ),
    (∀ n, seq.a n = seq.a 1 + (n - 1) * d) ∧
    d = -2 ∧
    (∀ n, seq.a n = 15 - 2 * n) ∧
    (∃ n_max : ℕ, ∀ n : ℕ, 
      (n * (seq.a 1 + seq.a n) / 2) ≤ (n_max * (seq.a 1 + seq.a n_max) / 2) ∧
      n_max * (seq.a 1 + seq.a n_max) / 2 = 49) :=
by sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l192_19276


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_84_l192_19252

theorem distinct_prime_factors_of_84 : ∃ (p q r : Nat), 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  84 = p * q * r := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_84_l192_19252


namespace NUMINAMATH_CALUDE_distribution_methods_count_l192_19270

/-- The number of ways to distribute 4 out of 7 different books to 4 students -/
def distribute_books (total_books : ℕ) (books_to_distribute : ℕ) (students : ℕ) 
  (restricted_books : ℕ) (restricted_student : ℕ) : ℕ :=
  (total_books - restricted_books) * 
  (Nat.factorial (total_books - 1) / (Nat.factorial (total_books - books_to_distribute) * 
   Nat.factorial (books_to_distribute - 1)))

/-- Theorem stating the number of distribution methods -/
theorem distribution_methods_count : 
  distribute_books 7 4 4 2 1 = 600 := by
  sorry

end NUMINAMATH_CALUDE_distribution_methods_count_l192_19270


namespace NUMINAMATH_CALUDE_equation_solution_l192_19293

theorem equation_solution : ∃ x : ℝ, 6 * x + 12 * x = 558 - 9 * (x - 4) ∧ x = 22 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l192_19293


namespace NUMINAMATH_CALUDE_picture_placement_l192_19257

theorem picture_placement (wall_width picture_width : ℝ) 
  (hw : wall_width = 22)
  (hp : picture_width = 4) : 
  (wall_width - picture_width) / 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_picture_placement_l192_19257


namespace NUMINAMATH_CALUDE_initial_pencils_count_l192_19220

/-- The number of pencils Sara added to the drawer -/
def pencils_added : ℕ := 100

/-- The total number of pencils in the drawer after Sara's addition -/
def total_pencils : ℕ := 215

/-- The initial number of pencils in the drawer -/
def initial_pencils : ℕ := total_pencils - pencils_added

theorem initial_pencils_count : initial_pencils = 115 := by
  sorry

end NUMINAMATH_CALUDE_initial_pencils_count_l192_19220


namespace NUMINAMATH_CALUDE_birds_in_tree_l192_19222

theorem birds_in_tree (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 29)
  (h2 : final_birds = 42) :
  final_birds - initial_birds = 13 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l192_19222


namespace NUMINAMATH_CALUDE_inverse_relation_l192_19261

theorem inverse_relation (x : ℝ) (h : 1 / x = 40) : x = 1 / 40 := by
  sorry

end NUMINAMATH_CALUDE_inverse_relation_l192_19261


namespace NUMINAMATH_CALUDE_nabla_square_l192_19259

theorem nabla_square (odot nabla : ℕ) : 
  odot ≠ nabla → 
  0 < odot → odot < 20 → 
  0 < nabla → nabla < 20 → 
  nabla * nabla * nabla = nabla →
  nabla * nabla = 64 := by
sorry

end NUMINAMATH_CALUDE_nabla_square_l192_19259


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l192_19227

theorem simplify_and_evaluate (x y : ℝ) (hx : x = -2) (hy : y = -1) :
  x + (1/3) * y^2 - 2 * (x - (1/3) * y^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l192_19227


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l192_19225

-- Define the given quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 5 * x - 2

-- Define the solution set of the given inequality
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | 1/2 < x ∧ x < 2}

-- Define the second quadratic function
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 5 * x + a^2 - 1

theorem quadratic_inequality_problem (a : ℝ) :
  (∀ x, f a x > 0 ↔ x ∈ solution_set a) →
  (a = -2 ∧
   ∀ x, g a x > 0 ↔ -3 < x ∧ x < 1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l192_19225


namespace NUMINAMATH_CALUDE_triangle_angle_inequalities_l192_19275

-- Define a structure for a triangle's angles
structure TriangleAngles where
  a : Real
  b : Real
  c : Real
  sum_is_pi : a + b + c = Real.pi

-- Theorem statement
theorem triangle_angle_inequalities (t : TriangleAngles) :
  (Real.sin t.a + Real.sin t.b + Real.sin t.c ≤ 3 * Real.sqrt 3 / 2) ∧
  (Real.cos (t.a / 2) + Real.cos (t.b / 2) + Real.cos (t.c / 2) ≤ 3 * Real.sqrt 3 / 2) ∧
  (Real.cos t.a * Real.cos t.b * Real.cos t.c ≤ 1 / 8) ∧
  (Real.sin (2 * t.a) + Real.sin (2 * t.b) + Real.sin (2 * t.c) ≤ Real.sin t.a + Real.sin t.b + Real.sin t.c) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_inequalities_l192_19275


namespace NUMINAMATH_CALUDE_intersection_equality_l192_19208

def A : Set ℝ := {x | x^2 - 2*x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem intersection_equality (a : ℝ) : (A ∩ B a = B a) → (a = 0 ∨ a = -1 ∨ a = 1/3) :=
sorry

end NUMINAMATH_CALUDE_intersection_equality_l192_19208


namespace NUMINAMATH_CALUDE_existence_of_coprime_sum_l192_19229

theorem existence_of_coprime_sum (n k : ℕ+) 
  (h : n.val % 2 = 1 ∨ (n.val % 2 = 0 ∧ k.val % 2 = 0)) :
  ∃ a b : ℤ, Nat.gcd a.natAbs n.val = 1 ∧ 
             Nat.gcd b.natAbs n.val = 1 ∧ 
             k.val = a + b := by
  sorry

end NUMINAMATH_CALUDE_existence_of_coprime_sum_l192_19229


namespace NUMINAMATH_CALUDE_call_center_theorem_l192_19207

/-- Represents the ratio of team A's size to team B's size -/
def team_size_ratio : ℚ := 5/8

/-- Represents the fraction of total calls processed by team B -/
def team_b_call_fraction : ℚ := 4/5

/-- Represents the ratio of calls processed by each member of team A to each member of team B -/
def member_call_ratio : ℚ := 2/5

theorem call_center_theorem :
  let total_calls : ℚ := 1
  let team_a_call_fraction : ℚ := total_calls - team_b_call_fraction
  team_size_ratio * (team_a_call_fraction / team_b_call_fraction) = member_call_ratio := by
  sorry

end NUMINAMATH_CALUDE_call_center_theorem_l192_19207


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l192_19200

/-- A circle with center (1, 0) and radius √m is tangent to the line x + y = 1 if and only if m = 1/2 -/
theorem circle_tangent_to_line (m : ℝ) : 
  (∃ (x y : ℝ), (x - 1)^2 + y^2 = m ∧ x + y = 1 ∧ 
    ∀ (x' y' : ℝ), (x' - 1)^2 + y'^2 = m → x' + y' ≥ 1) ↔ 
  m = 1/2 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l192_19200


namespace NUMINAMATH_CALUDE_min_value_trig_function_l192_19240

open Real

theorem min_value_trig_function (θ : Real) (h₁ : θ > 0) (h₂ : θ < π / 2) :
  ∀ y : Real, y = 1 / (sin θ)^2 + 9 / (cos θ)^2 → y ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_function_l192_19240


namespace NUMINAMATH_CALUDE_negation_of_all_squared_nonnegative_l192_19272

theorem negation_of_all_squared_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_all_squared_nonnegative_l192_19272


namespace NUMINAMATH_CALUDE_multiplication_division_equality_l192_19209

theorem multiplication_division_equality : (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5) = 800 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_division_equality_l192_19209


namespace NUMINAMATH_CALUDE_inequality_solution_l192_19235

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 3) ↔ 
  (x < -2 ∨ (-1 < x ∧ x < 0) ∨ 1 < x) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l192_19235


namespace NUMINAMATH_CALUDE_average_of_DEF_l192_19206

theorem average_of_DEF (D E F : ℚ) 
  (eq1 : 2003 * F - 4006 * D = 8012)
  (eq2 : 2003 * E + 6009 * D = 10010) :
  (D + E + F) / 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_average_of_DEF_l192_19206


namespace NUMINAMATH_CALUDE_max_profit_multimedia_devices_l192_19232

/-- Represents the profit function for multimedia devices -/
def profit_function (x : ℝ) : ℝ := -0.1 * x + 20

/-- Represents the constraint on the number of devices -/
def device_constraint (x : ℝ) : Prop := 10 ≤ x ∧ x ≤ 50

theorem max_profit_multimedia_devices :
  ∃ (x : ℝ), device_constraint x ∧
    (∀ y, device_constraint y → profit_function x ≥ profit_function y) ∧
    profit_function x = 19 ∧
    x = 10 := by sorry

end NUMINAMATH_CALUDE_max_profit_multimedia_devices_l192_19232


namespace NUMINAMATH_CALUDE_friends_journey_time_l192_19266

/-- Represents the journey of three friends with a bicycle --/
theorem friends_journey_time :
  -- Define the walking speed of the friends
  ∀ (walking_speed : ℝ),
  -- Define the bicycle speed
  ∀ (bicycle_speed : ℝ),
  -- Conditions
  (walking_speed > 0) →
  (bicycle_speed > 0) →
  -- Second friend walks 6 km in the first hour
  (walking_speed * 1 = 6) →
  -- Third friend rides 12 km in 2/3 hour
  (bicycle_speed * (2/3) = 12) →
  -- Total journey time
  ∃ (total_time : ℝ),
  total_time = 2 + 2/3 :=
by sorry

end NUMINAMATH_CALUDE_friends_journey_time_l192_19266


namespace NUMINAMATH_CALUDE_work_completion_time_l192_19221

/-- Given a work that can be completed by A in 14 days and by A and B together in 10 days,
    prove that B can complete the work alone in 35 days. -/
theorem work_completion_time (work : ℝ) (A B : ℝ → ℝ) : 
  (A work = work / 14) →
  (A work + B work = work / 10) →
  B work = work / 35 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l192_19221


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l192_19219

theorem quadratic_equation_result (m : ℝ) (h : m^2 + 2*m = 3) : 4*m^2 + 8*m - 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l192_19219


namespace NUMINAMATH_CALUDE_cube_painting_theorem_l192_19291

theorem cube_painting_theorem (n : ℕ) (h : n > 4) :
  (6 * (n - 4)^2 : ℕ) = (n - 4)^3 ↔ n = 10 := by sorry

end NUMINAMATH_CALUDE_cube_painting_theorem_l192_19291


namespace NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l192_19287

theorem tan_theta_minus_pi_fourth (θ : ℝ) (z : ℂ) : 
  z = (Real.cos θ - 4/5) + (Real.sin θ - 3/5) * Complex.I ∧ 
  z.re = 0 ∧ z.im ≠ 0 → 
  Real.tan (θ - π/4) = -7 :=
by sorry

end NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l192_19287


namespace NUMINAMATH_CALUDE_product_of_sums_equals_3280_l192_19238

theorem product_of_sums_equals_3280 :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_3280_l192_19238


namespace NUMINAMATH_CALUDE_group_size_proof_l192_19233

theorem group_size_proof (total : ℕ) 
  (h1 : (2 : ℚ) / 5 * total = (28 : ℚ) / 100 * total + 96) 
  (h2 : (28 : ℚ) / 100 * total = total - ((2 : ℚ) / 5 * total - 96)) : 
  total = 800 := by
sorry

end NUMINAMATH_CALUDE_group_size_proof_l192_19233


namespace NUMINAMATH_CALUDE_digit_2009_is_zero_l192_19271

/-- Represents the sequence of consecutive natural numbers starting from 1 -/
def consecutiveNaturals : ℕ → ℕ
  | 0 => 1
  | n + 1 => consecutiveNaturals n + 1

/-- Returns the nth digit in the sequence of consecutive natural numbers -/
def nthDigit (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 2009th digit in the sequence is 0 -/
theorem digit_2009_is_zero : nthDigit 2009 = 0 := by sorry

end NUMINAMATH_CALUDE_digit_2009_is_zero_l192_19271


namespace NUMINAMATH_CALUDE_stack_weight_error_l192_19282

/-- The weight of a disc with exactly 1 meter diameter in kg -/
def standard_weight : ℝ := 100

/-- The nominal radius of a disc in meters -/
def nominal_radius : ℝ := 0.5

/-- The standard deviation of the radius in meters -/
def radius_std_dev : ℝ := 0.01

/-- The number of discs in the stack -/
def num_discs : ℕ := 100

/-- The expected weight of a single disc given the manufacturing variation -/
def expected_single_disc_weight : ℝ := sorry

/-- The expected weight of the stack of discs -/
def expected_stack_weight : ℝ := sorry

/-- Engineer Sidorov's estimate of the stack weight -/
def sidorov_estimate : ℝ := 10000

theorem stack_weight_error :
  expected_stack_weight - sidorov_estimate = 4 := by sorry

end NUMINAMATH_CALUDE_stack_weight_error_l192_19282


namespace NUMINAMATH_CALUDE_cricket_players_l192_19265

theorem cricket_players (total : ℕ) (football : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 460)
  (h2 : football = 325)
  (h3 : neither = 50)
  (h4 : both = 90) :
  ∃ cricket : ℕ, cricket = 175 ∧ 
  cricket = total - neither - (football - both) := by
  sorry

end NUMINAMATH_CALUDE_cricket_players_l192_19265


namespace NUMINAMATH_CALUDE_cabinet_can_pass_through_door_l192_19253

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a room with given dimensions -/
def Room := Dimensions

/-- Represents a cabinet with given dimensions -/
def Cabinet := Dimensions

/-- Represents a door with given dimensions -/
structure Door where
  width : ℝ
  height : ℝ

/-- Checks if a cabinet can pass through a door -/
def can_pass_through (c : Cabinet) (d : Door) : Prop :=
  (c.width ≤ d.width ∧ c.height ≤ d.height) ∨
  (c.width ≤ d.height ∧ c.height ≤ d.width) ∨
  (c.length ≤ d.width ∧ c.height ≤ d.height) ∨
  (c.length ≤ d.height ∧ c.height ≤ d.width)

theorem cabinet_can_pass_through_door 
  (room : Room)
  (cabinet : Cabinet)
  (door : Door)
  (h_room : room = ⟨4, 2.5, 2.3⟩)
  (h_cabinet : cabinet = ⟨1.8, 0.6, 2.1⟩)
  (h_door : door = ⟨0.8, 1.9⟩) :
  can_pass_through cabinet door :=
sorry

end NUMINAMATH_CALUDE_cabinet_can_pass_through_door_l192_19253


namespace NUMINAMATH_CALUDE_inequalities_always_true_l192_19296

theorem inequalities_always_true 
  (x y a b : ℝ) 
  (hx : x > 0) (hy : y > 0) (ha : a > 0) (hb : b > 0)
  (hxa : x ≤ a) (hyb : y ≤ b) : 
  (x + y ≤ a + b) ∧ 
  (x - y ≤ a - b) ∧ 
  (x * y ≤ a * b) ∧ 
  (x / y ≤ a / b) := by
sorry

end NUMINAMATH_CALUDE_inequalities_always_true_l192_19296


namespace NUMINAMATH_CALUDE_work_completion_theorem_l192_19280

/-- The number of days it takes the first group to complete the work -/
def days_first_group : ℕ := 25

/-- The number of men in the second group -/
def men_second_group : ℕ := 20

/-- The number of days it takes the second group to complete the work -/
def days_second_group : ℕ := 20

/-- The number of men in the first group -/
def men_first_group : ℕ := men_second_group * days_second_group / days_first_group

theorem work_completion_theorem :
  men_first_group * days_first_group = men_second_group * days_second_group ∧
  men_first_group = 16 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l192_19280


namespace NUMINAMATH_CALUDE_jerry_earnings_duration_l192_19295

def jerry_earnings : ℕ := 14 + 31 + 20

def jerry_weekly_expenses : ℕ := 5 + 10 + 8

theorem jerry_earnings_duration : 
  ⌊(jerry_earnings : ℚ) / jerry_weekly_expenses⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_jerry_earnings_duration_l192_19295


namespace NUMINAMATH_CALUDE_article_cost_l192_19214

theorem article_cost (sell_price_high : ℝ) (sell_price_low : ℝ) (gain_percentage : ℝ) :
  sell_price_high = 350 →
  sell_price_low = 340 →
  gain_percentage = 0.04 →
  ∃ (cost : ℝ),
    sell_price_high - cost = (1 + gain_percentage) * (sell_price_low - cost) ∧
    cost = 90 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l192_19214


namespace NUMINAMATH_CALUDE_calculate_mixed_number_l192_19284

theorem calculate_mixed_number : 7 * (9 + 2/5) - 3 = 62 + 4/5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_mixed_number_l192_19284


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l192_19274

/-- Given a geometric sequence {a_n} with common ratio q = 1/2, prove that S_4 / a_2 = 15/4,
    where S_n is the sum of the first n terms. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = (1 / 2) * a n) →  -- Common ratio q = 1/2
  (∀ n, S n = a 1 * (1 - (1 / 2)^n) / (1 - (1 / 2))) →  -- Definition of S_n
  S 4 / a 2 = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l192_19274


namespace NUMINAMATH_CALUDE_ticket_draw_theorem_l192_19258

theorem ticket_draw_theorem (total : ℕ) (blue green red yellow orange : ℕ) : 
  total = 400 ∧ 
  blue + green + red + yellow + orange = total ∧
  blue * 2 = green ∧ 
  green * 2 = red ∧
  green * 3 = yellow ∧
  yellow * 2 = orange →
  (∃ n : ℕ, n ≤ 196 ∧ 
    (∀ m : ℕ, m < n → 
      (m ≤ blue ∨ m ≤ green ∨ m ≤ red ∨ m ≤ yellow ∨ m ≤ orange) ∧ 
      m < 50)) ∧
  (∃ color : ℕ, color ≥ 50 ∧ 
    (color = blue ∨ color = green ∨ color = red ∨ color = yellow ∨ color = orange) ∧
    color ≤ 196) := by
  sorry

end NUMINAMATH_CALUDE_ticket_draw_theorem_l192_19258


namespace NUMINAMATH_CALUDE_overhead_percentage_l192_19279

theorem overhead_percentage (purchase_price markup net_profit : ℝ) :
  purchase_price = 48 →
  markup = 45 →
  net_profit = 12 →
  (((purchase_price + markup - net_profit) - purchase_price) / purchase_price) * 100 = 68.75 := by
  sorry

end NUMINAMATH_CALUDE_overhead_percentage_l192_19279


namespace NUMINAMATH_CALUDE_function_shift_l192_19288

theorem function_shift (f : ℝ → ℝ) : 
  (∀ x, f (x - 1) = x^2) → (∀ x, f x = (x + 1)^2) := by
sorry

end NUMINAMATH_CALUDE_function_shift_l192_19288


namespace NUMINAMATH_CALUDE_mass_of_man_on_boat_l192_19228

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth boat_sinking water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sinking * water_density

/-- Theorem stating the mass of the man in the given scenario. -/
theorem mass_of_man_on_boat :
  let boat_length : ℝ := 4
  let boat_breadth : ℝ := 3
  let boat_sinking : ℝ := 0.01  -- 1 cm in meters
  let water_density : ℝ := 1000 -- kg/m³
  mass_of_man boat_length boat_breadth boat_sinking water_density = 120 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_man_on_boat_l192_19228


namespace NUMINAMATH_CALUDE_inverse_matrices_l192_19254

/-- Two 2x2 matrices are inverses if their product is the identity matrix -/
def are_inverses (A B : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  A * B = !![1, 0; 0, 1]

/-- Matrix A definition -/
def A (x : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![x, 3; 1, 5]

/-- Matrix B definition -/
def B (y : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![-5/31, 1/31; y, 3/31]

/-- The theorem to be proved -/
theorem inverse_matrices :
  are_inverses (A (-9)) (B (1/31)) := by sorry

end NUMINAMATH_CALUDE_inverse_matrices_l192_19254


namespace NUMINAMATH_CALUDE_sum_of_digits_divisible_by_11_l192_19234

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Among 39 sequential natural numbers, there's always one with sum of digits divisible by 11 -/
theorem sum_of_digits_divisible_by_11 (N : ℕ) : 
  ∃ k : ℕ, k ∈ Finset.range 39 ∧ (sum_of_digits (N + k) % 11 = 0) := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_divisible_by_11_l192_19234


namespace NUMINAMATH_CALUDE_parabola_point_x_coordinate_l192_19277

/-- Given a parabola y² = 4x and a point M on the parabola whose distance to the focus is 3,
    prove that the x-coordinate of M is 2. -/
theorem parabola_point_x_coordinate (x y : ℝ) : 
  y^2 = 4*x →  -- M is on the parabola y² = 4x
  (x - 1)^2 + y^2 = 3^2 →  -- Distance from M to focus (1, 0) is 3
  x = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_point_x_coordinate_l192_19277


namespace NUMINAMATH_CALUDE_number_of_workers_l192_19216

theorem number_of_workers (
  avg_salary_with_first_supervisor : ℝ)
  (first_supervisor_salary : ℝ)
  (avg_salary_with_new_supervisor : ℝ)
  (new_supervisor_salary : ℝ)
  (h1 : avg_salary_with_first_supervisor = 430)
  (h2 : first_supervisor_salary = 870)
  (h3 : avg_salary_with_new_supervisor = 440)
  (h4 : new_supervisor_salary = 960) :
  ∃ (w : ℕ), w = 8 ∧
  (w + 1) * avg_salary_with_first_supervisor - first_supervisor_salary =
  9 * avg_salary_with_new_supervisor - new_supervisor_salary :=
by sorry

end NUMINAMATH_CALUDE_number_of_workers_l192_19216


namespace NUMINAMATH_CALUDE_circle_reflection_translation_l192_19283

/-- Given a point (3, -4), prove that reflecting it across the x-axis
    and then translating it 5 units to the left results in the point (-2, 4) -/
theorem circle_reflection_translation :
  let initial_point : ℝ × ℝ := (3, -4)
  let reflected_point : ℝ × ℝ := (initial_point.1, -initial_point.2)
  let final_point : ℝ × ℝ := (reflected_point.1 - 5, reflected_point.2)
  final_point = (-2, 4) := by
sorry

end NUMINAMATH_CALUDE_circle_reflection_translation_l192_19283


namespace NUMINAMATH_CALUDE_soccer_team_lineups_l192_19245

/-- The number of ways to select a starting lineup from a soccer team -/
def numStartingLineups (totalPlayers : ℕ) (regularPlayers : ℕ) : ℕ :=
  totalPlayers * Nat.choose (totalPlayers - 1) regularPlayers

/-- Theorem stating that the number of starting lineups for a team of 16 players,
    with 1 goalie and 10 regular players, is 48,048 -/
theorem soccer_team_lineups :
  numStartingLineups 16 10 = 48048 := by
  sorry

#eval numStartingLineups 16 10

end NUMINAMATH_CALUDE_soccer_team_lineups_l192_19245


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l192_19290

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^5 + 2 = (x^2 - 2*x + 3) * q + (-4*x^2 - 3*x + 2) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l192_19290


namespace NUMINAMATH_CALUDE_factorial_ratio_equality_l192_19247

theorem factorial_ratio_equality : (Nat.factorial 9)^2 / (Nat.factorial 4 * Nat.factorial 5) = 45760000 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equality_l192_19247


namespace NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l192_19285

-- Define the circle
def circle_radius : ℝ := 8

-- Define the diameter
def diameter : ℝ := 2 * circle_radius

-- Define the height of the triangle (equal to the radius)
def triangle_height : ℝ := circle_radius

-- Theorem statement
theorem largest_inscribed_triangle_area :
  let triangle_area := (1 / 2) * diameter * triangle_height
  triangle_area = 64 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l192_19285


namespace NUMINAMATH_CALUDE_ships_initial_distance_l192_19212

/-- The initial distance between two ships moving towards a port -/
def initial_distance : ℝ := 240

/-- The distance traveled by the second ship when a right triangle is formed -/
def right_triangle_distance : ℝ := 80

/-- The remaining distance for the second ship when the first ship reaches the port -/
def remaining_distance : ℝ := 120

theorem ships_initial_distance :
  ∃ (v₁ v₂ : ℝ), v₁ > 0 ∧ v₂ > 0 ∧
  (initial_distance - v₁ * (right_triangle_distance / v₂))^2 + right_triangle_distance^2 = initial_distance^2 ∧
  (initial_distance / v₁) * v₂ = initial_distance - remaining_distance :=
by sorry

#check ships_initial_distance

end NUMINAMATH_CALUDE_ships_initial_distance_l192_19212


namespace NUMINAMATH_CALUDE_distance_at_least_diameter_time_l192_19213

/-- Represents a circular track -/
structure Track where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a car on a track -/
structure Car where
  track : Track
  clockwise : Bool
  position : ℝ → ℝ × ℝ

/-- The setup of the problem -/
def problem_setup : ℝ × Track × Track × Car × Car := sorry

/-- The time during which the distance between the cars is at least the diameter of each track -/
def time_at_least_diameter (setup : ℝ × Track × Track × Car × Car) : ℝ := sorry

/-- The main theorem stating that the time during which the distance between the cars 
    is at least the diameter of each track is 1/2 hour -/
theorem distance_at_least_diameter_time 
  (setup : ℝ × Track × Track × Car × Car) 
  (h_setup : setup = problem_setup) : 
  time_at_least_diameter setup = 1/2 := by sorry

end NUMINAMATH_CALUDE_distance_at_least_diameter_time_l192_19213


namespace NUMINAMATH_CALUDE_triangle_inequality_l192_19298

theorem triangle_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_sum : a + b + c = 1) : 
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l192_19298


namespace NUMINAMATH_CALUDE_box_areas_product_l192_19264

/-- For a rectangular box with dimensions a, b, and c, and a constant k,
    where the areas of the bottom, side, and front are kab, kbc, and kca respectively,
    the product of these areas is equal to k^3 × (abc)^2. -/
theorem box_areas_product (a b c k : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ k > 0) :
  (k * a * b) * (k * b * c) * (k * c * a) = k^3 * (a * b * c)^2 := by
  sorry

end NUMINAMATH_CALUDE_box_areas_product_l192_19264


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l192_19289

/-- An arithmetic sequence. -/
structure ArithmeticSequence (α : Type*) [Add α] [SMul ℕ α] :=
  (a : ℕ → α)
  (d : α)
  (h : ∀ n, a (n + 1) = a n + d)

/-- Theorem: In an arithmetic sequence where a₂ + a₈ = 12, a₅ = 6. -/
theorem arithmetic_sequence_property 
  (a : ArithmeticSequence ℝ) 
  (h : a.a 2 + a.a 8 = 12) : 
  a.a 5 = 6 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l192_19289


namespace NUMINAMATH_CALUDE_son_age_problem_l192_19244

theorem son_age_problem (son_age father_age : ℕ) : 
  father_age = son_age + 35 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 33 := by
sorry

end NUMINAMATH_CALUDE_son_age_problem_l192_19244


namespace NUMINAMATH_CALUDE_trip_charges_eq_14_l192_19292

/-- Represents the daily mileage and charging capacity for a 7-day trip. -/
structure TripData where
  daily_mileage : Fin 7 → ℕ
  initial_charging_capacity : ℕ
  daily_capacity_increment : ℕ
  weather_reduction_days : Finset (Fin 7)
  weather_reduction_percent : ℚ
  stop_interval : ℕ
  stop_days : Finset (Fin 7)

/-- Calculates the number of charges needed for a given day. -/
def charges_needed (data : TripData) (day : Fin 7) : ℕ :=
  sorry

/-- Calculates the total number of charges needed for the entire trip. -/
def total_charges (data : TripData) : ℕ :=
  sorry

/-- The main theorem stating that the total number of charges for the given trip data is 14. -/
theorem trip_charges_eq_14 : ∃ (data : TripData),
  data.daily_mileage = ![135, 259, 159, 189, 210, 156, 240] ∧
  data.initial_charging_capacity = 106 ∧
  data.daily_capacity_increment = 15 ∧
  data.weather_reduction_days = {3, 6} ∧
  data.weather_reduction_percent = 5 / 100 ∧
  data.stop_interval = 55 ∧
  data.stop_days = {1, 5} ∧
  total_charges data = 14 :=
  sorry

end NUMINAMATH_CALUDE_trip_charges_eq_14_l192_19292


namespace NUMINAMATH_CALUDE_g_1994_of_4_l192_19242

def g (x : ℚ) : ℚ := (2 + x) / (2 - 4 * x)

def g_n : ℕ → (ℚ → ℚ)
  | 0 => id
  | n + 1 => λ x => g (g_n n x)

theorem g_1994_of_4 : g_n 1994 4 = 87 / 50 := by
  sorry

end NUMINAMATH_CALUDE_g_1994_of_4_l192_19242


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l192_19205

/-- A taxi fare system with a fixed starting fee and a proportional amount per mile -/
structure TaxiFare where
  startingFee : ℝ
  costPerMile : ℝ

/-- Calculate the total fare for a given distance -/
def calculateFare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.startingFee + tf.costPerMile * distance

theorem taxi_fare_calculation (tf : TaxiFare) 
  (h1 : tf.startingFee = 20)
  (h2 : calculateFare tf 60 = 150)
  : calculateFare tf 80 = 193.33 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_calculation_l192_19205


namespace NUMINAMATH_CALUDE_unique_solution_for_A_l192_19201

/-- Given an equation 1A + 4B3 = 469, where A and B are single digits and 4B3 is a three-digit number,
    prove that A = 6 is the unique solution for A. -/
theorem unique_solution_for_A : ∃! (A : ℕ), ∃ (B : ℕ),
  (A < 10) ∧ (B < 10) ∧ (400 ≤ 4 * 10 * B + 3) ∧ (4 * 10 * B + 3 < 1000) ∧
  (10 * A + 4 * 10 * B + 3 = 469) ∧ A = 6 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_for_A_l192_19201


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l192_19204

theorem right_triangle_side_length 
  (A B C : ℝ) (AB BC AC : ℝ) :
  -- Triangle ABC is right-angled at A
  A + B + C = π / 2 →
  -- BC = 10
  BC = 10 →
  -- tan C = 3cos B
  Real.tan C = 3 * Real.cos B →
  -- AB² + AC² = BC²
  AB^2 + AC^2 = BC^2 →
  -- AB = (20√2)/3
  AB = 20 * Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l192_19204


namespace NUMINAMATH_CALUDE_larger_number_of_sum_and_difference_l192_19215

theorem larger_number_of_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 45) 
  (diff_eq : x - y = 7) : 
  max x y = 26 := by
sorry

end NUMINAMATH_CALUDE_larger_number_of_sum_and_difference_l192_19215


namespace NUMINAMATH_CALUDE_derek_rides_more_than_carla_l192_19241

-- Define the speeds and times
def carla_speed : ℝ := 12
def derek_speed : ℝ := 15
def derek_time : ℝ := 3
def time_difference : ℝ := 0.5

-- Theorem statement
theorem derek_rides_more_than_carla :
  derek_speed * derek_time - carla_speed * (derek_time + time_difference) = 3 := by
  sorry

end NUMINAMATH_CALUDE_derek_rides_more_than_carla_l192_19241


namespace NUMINAMATH_CALUDE_positive_y_intercept_l192_19267

/-- A line that intersects the y-axis in the positive half-plane -/
structure PositiveYInterceptLine where
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line equation is y = 2x + b -/
  equation : ∀ (x y : ℝ), y = 2 * x + b
  /-- The line intersects the y-axis in the positive half-plane -/
  positive_intercept : ∃ (y : ℝ), y > 0 ∧ y = b

/-- The y-intercept of a line that intersects the y-axis in the positive half-plane is positive -/
theorem positive_y_intercept (l : PositiveYInterceptLine) : l.b > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_y_intercept_l192_19267


namespace NUMINAMATH_CALUDE_first_statue_weight_l192_19251

/-- Given the weights of a marble block and its carved statues, prove the weight of the first statue -/
theorem first_statue_weight
  (total_weight : ℝ)
  (second_statue : ℝ)
  (third_statue : ℝ)
  (fourth_statue : ℝ)
  (discarded : ℝ)
  (h1 : total_weight = 80)
  (h2 : second_statue = 18)
  (h3 : third_statue = 15)
  (h4 : fourth_statue = 15)
  (h5 : discarded = 22)
  : ∃ (first_statue : ℝ),
    first_statue + second_statue + third_statue + fourth_statue + discarded = total_weight ∧
    first_statue = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_statue_weight_l192_19251


namespace NUMINAMATH_CALUDE_complex_rectangle_perimeter_l192_19299

/-- A structure representing a rectangle with an internal complex shape. -/
structure ComplexRectangle where
  width : ℝ
  height : ℝ
  enclosed_area : ℝ

/-- The perimeter of a ComplexRectangle is equal to 2 * (width + height) -/
def perimeter (r : ComplexRectangle) : ℝ := 2 * (r.width + r.height)

theorem complex_rectangle_perimeter :
  ∀ (r : ComplexRectangle),
    r.width = 15 ∧ r.height = 10 ∧ r.enclosed_area = 108 →
    perimeter r = 50 := by
  sorry

end NUMINAMATH_CALUDE_complex_rectangle_perimeter_l192_19299


namespace NUMINAMATH_CALUDE_george_total_earnings_l192_19249

/-- The total amount earned by George from selling toys -/
def george_earnings (num_cars : ℕ) (price_per_car : ℕ) (lego_price : ℕ) : ℕ :=
  num_cars * price_per_car + lego_price

/-- Theorem: George earned $45 from selling 3 cars at $5 each and a set of Legos for $30 -/
theorem george_total_earnings : george_earnings 3 5 30 = 45 := by
  sorry

end NUMINAMATH_CALUDE_george_total_earnings_l192_19249


namespace NUMINAMATH_CALUDE_intersection_singleton_k_value_l192_19269

theorem intersection_singleton_k_value (k : ℝ) : 
  let A : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2 * (p.1 + p.2)}
  let B : Set (ℝ × ℝ) := {p | k * p.1 - p.2 + k + 3 ≥ 0}
  (Set.Subsingleton (A ∩ B)) → k = -2 - Real.sqrt 3 :=
by sorry

#check intersection_singleton_k_value

end NUMINAMATH_CALUDE_intersection_singleton_k_value_l192_19269


namespace NUMINAMATH_CALUDE_largest_circle_area_l192_19262

/-- The area of the largest circle formed from a string that fits exactly around a rectangle -/
theorem largest_circle_area (string_length : ℝ) (rectangle_area : ℝ) : 
  string_length = 60 →
  rectangle_area = 200 →
  (∃ (x y : ℝ), x * y = rectangle_area ∧ 2 * (x + y) = string_length) →
  (π * (string_length / (2 * π))^2 : ℝ) = 900 / π :=
by sorry

end NUMINAMATH_CALUDE_largest_circle_area_l192_19262


namespace NUMINAMATH_CALUDE_x_equals_n_l192_19263

def x : ℕ → ℚ
  | 0 => 0
  | n + 1 => ((n^2 + n + 1) * x n + 1) / (n^2 + n + 1 - x n)

theorem x_equals_n (n : ℕ) : x n = n := by
  sorry

end NUMINAMATH_CALUDE_x_equals_n_l192_19263


namespace NUMINAMATH_CALUDE_infinite_series_sum_l192_19236

theorem infinite_series_sum : 
  (∑' n : ℕ+, (1 : ℝ) / (n * (n + 3))) = 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l192_19236


namespace NUMINAMATH_CALUDE_monic_quadratic_polynomial_l192_19246

theorem monic_quadratic_polynomial (f : ℝ → ℝ) : 
  (∀ x, f x = x^2 + 5*x + 6) → 
  f 0 = 6 ∧ f 1 = 12 := by
sorry

end NUMINAMATH_CALUDE_monic_quadratic_polynomial_l192_19246


namespace NUMINAMATH_CALUDE_iris_shopping_l192_19243

theorem iris_shopping (jacket_price shorts_price pants_price total_spent : ℕ)
  (jacket_quantity pants_quantity : ℕ) :
  jacket_price = 10 →
  shorts_price = 6 →
  pants_price = 12 →
  jacket_quantity = 3 →
  pants_quantity = 4 →
  total_spent = 90 →
  ∃ shorts_quantity : ℕ, 
    total_spent = jacket_price * jacket_quantity + 
                  shorts_price * shorts_quantity + 
                  pants_price * pants_quantity ∧
    shorts_quantity = 2 :=
by sorry

end NUMINAMATH_CALUDE_iris_shopping_l192_19243


namespace NUMINAMATH_CALUDE_sum_of_segments_equals_radius_l192_19230

/-- A regular (4k+2)-gon inscribed in a circle -/
structure RegularPolygon (k : ℕ) :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (vertices : Fin (4*k+2) → ℝ × ℝ)

/-- The segments cut by angle A₍ₖ₎OA₍ₖ₊₁₎ on the lines A₁A₍₂ₖ₎, A₂A₍₂ₖ₋₁₎, ..., A₍ₖ₎A₍ₖ₊₁₎ -/
def cut_segments (p : RegularPolygon k) : List (ℝ × ℝ) :=
  sorry

/-- The sum of the lengths of the cut segments -/
def sum_of_segment_lengths (segments : List (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: The sum of the lengths of the cut segments is equal to the radius -/
theorem sum_of_segments_equals_radius (k : ℕ) (p : RegularPolygon k) :
  sum_of_segment_lengths (cut_segments p) = p.radius :=
sorry

end NUMINAMATH_CALUDE_sum_of_segments_equals_radius_l192_19230


namespace NUMINAMATH_CALUDE_distance_between_docks_l192_19297

/-- The distance between docks A and B in kilometers. -/
def distance : ℝ := 105

/-- The speed of the water flow in kilometers per hour. -/
def water_speed : ℝ := 3

/-- The time taken to travel downstream in hours. -/
def downstream_time : ℝ := 5

/-- The time taken to travel upstream in hours. -/
def upstream_time : ℝ := 7

/-- Theorem stating that the distance between docks A and B is 105 kilometers. -/
theorem distance_between_docks :
  distance = 105 ∧
  water_speed = 3 ∧
  downstream_time = 5 ∧
  upstream_time = 7 ∧
  (distance / downstream_time - water_speed = distance / upstream_time + water_speed) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_docks_l192_19297


namespace NUMINAMATH_CALUDE_intersection_complement_eq_set_l192_19237

def R : Set ℝ := Set.univ

def M : Set ℝ := {-1, 1, 2, 4}

def N : Set ℝ := {x : ℝ | x^2 - 2*x > 3}

theorem intersection_complement_eq_set : M ∩ (R \ N) = {-1, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_eq_set_l192_19237


namespace NUMINAMATH_CALUDE_number_puzzle_l192_19281

theorem number_puzzle (x : ℤ) : x + (x - 1) = 33 → 6 * x - 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l192_19281


namespace NUMINAMATH_CALUDE_challenge_points_40_l192_19294

/-- Calculates the number of activities required for a given number of challenge points. -/
def activities_required (n : ℕ) : ℕ :=
  let segment := (n - 1) / 10
  (n.min 10) * 1 + 
  ((n - 10).max 0).min 10 * 2 + 
  ((n - 20).max 0).min 10 * 3 + 
  ((n - 30).max 0).min 10 * 4 +
  ((n - 40).max 0) * (segment + 1)

/-- Proves that 40 challenge points require 100 activities. -/
theorem challenge_points_40 : activities_required 40 = 100 := by
  sorry

#eval activities_required 40

end NUMINAMATH_CALUDE_challenge_points_40_l192_19294


namespace NUMINAMATH_CALUDE_art_club_artworks_l192_19273

/-- The number of artworks collected by an art club over multiple school years -/
def artworks_collected (num_students : ℕ) (artworks_per_quarter : ℕ) (quarters_per_year : ℕ) (num_years : ℕ) : ℕ :=
  num_students * artworks_per_quarter * quarters_per_year * num_years

/-- Theorem: The art club collects 900 artworks in 3 school years -/
theorem art_club_artworks :
  artworks_collected 25 3 4 3 = 900 := by
  sorry

end NUMINAMATH_CALUDE_art_club_artworks_l192_19273


namespace NUMINAMATH_CALUDE_original_price_after_discounts_l192_19268

theorem original_price_after_discounts (final_price : ℝ) 
  (discount1 : ℝ) (discount2 : ℝ) (original_price : ℝ) : 
  final_price = 144 ∧ 
  discount1 = 0.1 ∧ 
  discount2 = 0.2 ∧ 
  final_price = original_price * (1 - discount1) * (1 - discount2) → 
  original_price = 200 := by sorry

end NUMINAMATH_CALUDE_original_price_after_discounts_l192_19268


namespace NUMINAMATH_CALUDE_tickets_found_is_zero_l192_19223

/-- The number of carnival games --/
def num_games : ℕ := 5

/-- The value of each ticket in dollars --/
def ticket_value : ℕ := 3

/-- The total value of all tickets in dollars --/
def total_value : ℕ := 30

/-- The number of tickets won from each game --/
def tickets_per_game : ℕ := total_value / (num_games * ticket_value)

/-- The number of tickets found on the floor --/
def tickets_found : ℕ := total_value - (num_games * tickets_per_game * ticket_value)

theorem tickets_found_is_zero : tickets_found = 0 := by
  sorry

end NUMINAMATH_CALUDE_tickets_found_is_zero_l192_19223


namespace NUMINAMATH_CALUDE_real_sum_greater_than_two_l192_19224

theorem real_sum_greater_than_two (x y : ℝ) : x + y > 2 → x > 1 ∨ y > 1 := by
  sorry

end NUMINAMATH_CALUDE_real_sum_greater_than_two_l192_19224


namespace NUMINAMATH_CALUDE_part_one_part_two_l192_19202

-- Define the function f
def f (x a b : ℝ) : ℝ := |2*x + a| + |2*x - 2*b| + 3

-- Part I
theorem part_one (x : ℝ) :
  let a : ℝ := 1
  let b : ℝ := 1
  f x a b > 8 ↔ (x < -1 ∨ x > 1.5) :=
sorry

-- Part II
theorem part_two (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x, f x a b ≥ 5) ∧ (∃ x, f x a b = 5) →
  (1/a + 1/b ≥ (3 + 2*Real.sqrt 2) / 2) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ (∀ x, f x a b ≥ 5) ∧ (∃ x, f x a b = 5) ∧ 1/a + 1/b = (3 + 2*Real.sqrt 2) / 2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l192_19202


namespace NUMINAMATH_CALUDE_remainder_9387_div_11_l192_19210

theorem remainder_9387_div_11 : 9387 % 11 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_9387_div_11_l192_19210


namespace NUMINAMATH_CALUDE_paper_distribution_l192_19218

theorem paper_distribution (x y : ℕ+) : 
  x * y = 221 ↔ (x = 1 ∧ y = 221) ∨ (x = 221 ∧ y = 1) ∨ (x = 13 ∧ y = 17) ∨ (x = 17 ∧ y = 13) := by
  sorry

end NUMINAMATH_CALUDE_paper_distribution_l192_19218


namespace NUMINAMATH_CALUDE_o2_moles_combined_l192_19226

-- Define the molecules and their molar ratios in the reaction
structure Reaction :=
  (C2H6_ratio : ℚ)
  (O2_ratio : ℚ)
  (C2H4O_ratio : ℚ)
  (H2O_ratio : ℚ)

-- Define the balanced reaction
def balanced_reaction : Reaction :=
  { C2H6_ratio := 1
  , O2_ratio := 1/2
  , C2H4O_ratio := 1
  , H2O_ratio := 1 }

-- Theorem statement
theorem o2_moles_combined 
  (r : Reaction) 
  (h1 : r.C2H6_ratio = 1) 
  (h2 : r.C2H4O_ratio = 1) 
  (h3 : r = balanced_reaction) : 
  r.O2_ratio = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_o2_moles_combined_l192_19226


namespace NUMINAMATH_CALUDE_total_interest_after_ten_years_l192_19248

/-- Calculate the total interest after 10 years given:
  * The simple interest on the initial principal for 10 years is 1400
  * The principal is trebled after 5 years
-/
theorem total_interest_after_ten_years (P R : ℝ) 
  (h1 : P * R * 10 / 100 = 1400) : 
  (P * R * 5 / 100) + (3 * P * R * 5 / 100) = 280 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_after_ten_years_l192_19248


namespace NUMINAMATH_CALUDE_solution_to_equation_l192_19260

theorem solution_to_equation (x : ℝ) (hx : x ≠ 0) :
  (7 * x)^5 = (14 * x)^4 ↔ x = 16/7 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l192_19260


namespace NUMINAMATH_CALUDE_monotone_increasing_quadratic_l192_19278

/-- A function f(x) = 4x^2 - kx - 8 is monotonically increasing on [5, +∞) if and only if k ≤ 40 -/
theorem monotone_increasing_quadratic (k : ℝ) :
  (∀ x ≥ 5, Monotone (fun x => 4 * x^2 - k * x - 8)) ↔ k ≤ 40 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_quadratic_l192_19278


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l192_19211

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a+b)*(b+c)*(c+d)*(d+a) :=
by sorry

theorem equality_condition (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 = (a+b)*(b+c)*(c+d)*(d+a) ↔ a = b ∧ b = c ∧ c = d :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l192_19211


namespace NUMINAMATH_CALUDE_city_water_consumption_most_suitable_l192_19286

/-- Represents a survey scenario -/
structure SurveyScenario where
  description : String
  population_size : Nat
  practicality_of_sampling : Bool

/-- Determines if a survey scenario is suitable for sampling -/
def is_suitable_for_sampling (scenario : SurveyScenario) : Bool :=
  scenario.population_size > 1000 && scenario.practicality_of_sampling

/-- The list of survey scenarios -/
def survey_scenarios : List SurveyScenario := [
  { description := "Security check for passengers before boarding a plane",
    population_size := 300,
    practicality_of_sampling := false },
  { description := "Survey of the vision of students in Grade 8, Class 1 of a certain school",
    population_size := 40,
    practicality_of_sampling := false },
  { description := "Survey of the average daily water consumption in a certain city",
    population_size := 100000,
    practicality_of_sampling := true },
  { description := "Survey of the sleep time of 20 centenarians in a certain county",
    population_size := 20,
    practicality_of_sampling := false }
]

theorem city_water_consumption_most_suitable :
  ∃ (scenario : SurveyScenario),
    scenario ∈ survey_scenarios ∧
    scenario.description = "Survey of the average daily water consumption in a certain city" ∧
    is_suitable_for_sampling scenario ∧
    ∀ (other : SurveyScenario),
      other ∈ survey_scenarios →
      other ≠ scenario →
      ¬(is_suitable_for_sampling other) :=
by sorry

end NUMINAMATH_CALUDE_city_water_consumption_most_suitable_l192_19286


namespace NUMINAMATH_CALUDE_magnitude_of_z_l192_19217

-- Define the complex number
def z : ℂ := 7 - 24 * Complex.I

-- State the theorem
theorem magnitude_of_z : Complex.abs z = 25 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l192_19217
