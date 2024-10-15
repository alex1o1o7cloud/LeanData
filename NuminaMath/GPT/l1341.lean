import Mathlib

namespace NUMINAMATH_GPT_probability_properties_l1341_134184

noncomputable def P1 : ℝ := 1 / 4
noncomputable def P2 : ℝ := 1 / 4
noncomputable def P3 : ℝ := 1 / 2

theorem probability_properties :
  (P1 ≠ P3) ∧
  (P1 + P2 = P3) ∧
  (P1 + P2 + P3 = 1) ∧
  (P3 = 2 * P1) ∧
  (P3 = 2 * P2) :=
by
  sorry

end NUMINAMATH_GPT_probability_properties_l1341_134184


namespace NUMINAMATH_GPT_ones_digit_of_9_pow_27_l1341_134182

-- Definitions representing the cyclical pattern
def ones_digit_of_9_power (n : ℕ) : ℕ :=
  if n % 2 = 1 then 9 else 1

-- The problem statement to be proven
theorem ones_digit_of_9_pow_27 : ones_digit_of_9_power 27 = 9 := 
by
  -- the detailed proof steps are omitted
  sorry

end NUMINAMATH_GPT_ones_digit_of_9_pow_27_l1341_134182


namespace NUMINAMATH_GPT_square_of_binomial_is_25_l1341_134193

theorem square_of_binomial_is_25 (a : ℝ)
  (h : ∃ b : ℝ, (4 * (x : ℝ) + b)^2 = 16 * x^2 + 40 * x + a) : a = 25 :=
sorry

end NUMINAMATH_GPT_square_of_binomial_is_25_l1341_134193


namespace NUMINAMATH_GPT_find_numbers_l1341_134133

theorem find_numbers
  (X Y : ℕ)
  (h1 : 10 ≤ X ∧ X < 100)
  (h2 : 10 ≤ Y ∧ Y < 100)
  (h3 : X = 2 * Y)
  (h4 : ∃ a b c d, X = 10 * a + b ∧ Y = 10 * c + d ∧ (c + d = a + b) ∧ (c = a - b ∨ d = a - b)) :
  X = 34 ∧ Y = 17 :=
sorry

end NUMINAMATH_GPT_find_numbers_l1341_134133


namespace NUMINAMATH_GPT_shopkeeper_gain_percent_l1341_134199

theorem shopkeeper_gain_percent
    (SP₁ SP₂ CP : ℝ)
    (h₁ : SP₁ = 187)
    (h₂ : SP₂ = 264)
    (h₃ : SP₁ = 0.85 * CP) :
    ((SP₂ - CP) / CP) * 100 = 20 := by 
  sorry

end NUMINAMATH_GPT_shopkeeper_gain_percent_l1341_134199


namespace NUMINAMATH_GPT_cube_coloring_schemes_l1341_134107

theorem cube_coloring_schemes (colors : Finset ℕ) (h : colors.card = 6) :
  ∃ schemes : Nat, schemes = 230 :=
by
  sorry

end NUMINAMATH_GPT_cube_coloring_schemes_l1341_134107


namespace NUMINAMATH_GPT_values_are_equal_and_differ_in_precision_l1341_134176

-- We define the decimal values
def val1 : ℝ := 4.5
def val2 : ℝ := 4.50

-- We define the counting units
def unit1 : ℝ := 0.1
def unit2 : ℝ := 0.01

-- Now, we state our theorem
theorem values_are_equal_and_differ_in_precision : 
  val1 = val2 ∧ unit1 ≠ unit2 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_values_are_equal_and_differ_in_precision_l1341_134176


namespace NUMINAMATH_GPT_inequality_solution_l1341_134163

-- Definitions
variables {a b : ℝ}

-- Hypothesis
variable (h : a > b)

-- Theorem
theorem inequality_solution : -2 * a < -2 * b :=
sorry

end NUMINAMATH_GPT_inequality_solution_l1341_134163


namespace NUMINAMATH_GPT_geometric_sequence_a4_l1341_134146

variable (a : ℕ → ℝ) (q : ℝ)

-- Conditions
def condition1 : Prop := 3 * a 5 = a 6
def condition2 : Prop := a 2 = 1

-- Question
def question : Prop := a 4 = 9

theorem geometric_sequence_a4 (h1 : condition1 a) (h2 : condition2 a) : question a :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a4_l1341_134146


namespace NUMINAMATH_GPT_height_of_parallelogram_l1341_134198

theorem height_of_parallelogram (A B H : ℕ) (hA : A = 308) (hB : B = 22) (h_eq : H = A / B) : H = 14 := 
by sorry

end NUMINAMATH_GPT_height_of_parallelogram_l1341_134198


namespace NUMINAMATH_GPT_solution_sets_equiv_solve_l1341_134142

theorem solution_sets_equiv_solve (a b : ℝ) :
  (∀ x : ℝ, (4 * x + 1) / (x + 2) < 0 ↔ -2 < x ∧ x < -1 / 4) →
  (∀ x : ℝ, a * x^2 + b * x - 2 > 0 ↔ -2 < x ∧ x < -1 / 4) →
  a = -4 ∧ b = -9 := by
  sorry

end NUMINAMATH_GPT_solution_sets_equiv_solve_l1341_134142


namespace NUMINAMATH_GPT_sector_arc_length_l1341_134164

theorem sector_arc_length (r : ℝ) (θ : ℝ) (L : ℝ) (h₁ : r = 1) (h₂ : θ = 60 * π / 180) : L = π / 3 :=
by
  sorry

end NUMINAMATH_GPT_sector_arc_length_l1341_134164


namespace NUMINAMATH_GPT_squirrel_count_l1341_134150

theorem squirrel_count (n m : ℕ) (h1 : n = 12) (h2 : m = 12 + 12 / 3) : n + m = 28 := by
  sorry

end NUMINAMATH_GPT_squirrel_count_l1341_134150


namespace NUMINAMATH_GPT_john_total_climb_height_l1341_134194

-- Define the heights and conditions
def num_flights : ℕ := 3
def height_per_flight : ℕ := 10
def total_stairs_height : ℕ := num_flights * height_per_flight
def rope_height : ℕ := total_stairs_height / 2
def ladder_height : ℕ := rope_height + 10

-- Prove that the total height John climbed is 70 feet
theorem john_total_climb_height : 
  total_stairs_height + rope_height + ladder_height = 70 := by
  sorry

end NUMINAMATH_GPT_john_total_climb_height_l1341_134194


namespace NUMINAMATH_GPT_evaluate_expression_l1341_134144

theorem evaluate_expression : 2^3 + 2^3 + 2^3 + 2^3 = 2^5 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1341_134144


namespace NUMINAMATH_GPT_sleep_hours_l1341_134106

-- Define the times Isaac wakes up, goes to sleep, and takes naps
def monday : ℝ := 16 - 9
def tuesday_night : ℝ := 12 - 6.5
def tuesday_nap : ℝ := 1
def wednesday : ℝ := 9.75 - 7.75
def thursday_night : ℝ := 15.5 - 8
def thursday_nap : ℝ := 1.5
def friday : ℝ := 12 - 7.25
def saturday : ℝ := 12.75 - 9
def sunday_night : ℝ := 10.5 - 8.5
def sunday_nap : ℝ := 2

noncomputable def total_sleep : ℝ := 
  monday +
  (tuesday_night + tuesday_nap) +
  wednesday +
  (thursday_night + thursday_nap) +
  friday +
  saturday +
  (sunday_night + sunday_nap)

theorem sleep_hours (total_sleep : ℝ) : total_sleep = 36.75 := 
by
  -- Here, you would provide the steps used to add up the hours, but we will skip with sorry
  sorry

end NUMINAMATH_GPT_sleep_hours_l1341_134106


namespace NUMINAMATH_GPT_solve_fractional_equation_for_c_l1341_134131

theorem solve_fractional_equation_for_c :
  (∃ c : ℝ, (c - 37) / 3 = (3 * c + 7) / 8) → c = -317 := by
sorry

end NUMINAMATH_GPT_solve_fractional_equation_for_c_l1341_134131


namespace NUMINAMATH_GPT_stanley_run_walk_difference_l1341_134175

theorem stanley_run_walk_difference :
  ∀ (ran walked : ℝ), ran = 0.4 → walked = 0.2 → ran - walked = 0.2 :=
by
  intros ran walked h_ran h_walk
  rw [h_ran, h_walk]
  norm_num

end NUMINAMATH_GPT_stanley_run_walk_difference_l1341_134175


namespace NUMINAMATH_GPT_max_value_of_sum_max_value_achievable_l1341_134177

theorem max_value_of_sum (a b c d : ℝ) 
  (h : a^6 + b^6 + c^6 + d^6 = 64) : a^7 + b^7 + c^7 + d^7 ≤ 128 :=
sorry

theorem max_value_achievable : ∃ a b c d : ℝ,
  a^6 + b^6 + c^6 + d^6 = 64 ∧ a^7 + b^7 + c^7 + d^7 = 128 :=
sorry

end NUMINAMATH_GPT_max_value_of_sum_max_value_achievable_l1341_134177


namespace NUMINAMATH_GPT_circles_common_point_l1341_134151

theorem circles_common_point {n : ℕ} (hn : n ≥ 5) (circles : Fin n → Set Point)
  (hcommon : ∀ (a b c : Fin n), (circles a ∩ circles b ∩ circles c).Nonempty) :
  ∃ p : Point, ∀ i : Fin n, p ∈ circles i :=
sorry

end NUMINAMATH_GPT_circles_common_point_l1341_134151


namespace NUMINAMATH_GPT_discount_percentage_is_25_l1341_134108

def piano_cost := 500
def lessons_count := 20
def lesson_price := 40
def total_paid := 1100

def lessons_cost := lessons_count * lesson_price
def total_cost := piano_cost + lessons_cost
def discount_amount := total_cost - total_paid
def discount_percentage := (discount_amount / lessons_cost) * 100

theorem discount_percentage_is_25 : discount_percentage = 25 := by
  sorry

end NUMINAMATH_GPT_discount_percentage_is_25_l1341_134108


namespace NUMINAMATH_GPT_problem_l1341_134166

-- Define first terms
def a_1 : ℕ := 12
def b_1 : ℕ := 48

-- Define the 100th term condition
def a_100 (d_a : ℚ) := 12 + 99 * d_a
def b_100 (d_b : ℚ) := 48 + 99 * d_b

-- Condition that the sum of the 100th terms is 200
def condition (d_a d_b : ℚ) := a_100 d_a + b_100 d_b = 200

-- Define the value of the sum of the first 100 terms
def sequence_sum (d_a d_b : ℚ) := 100 * 60 + (140 / 99) * ((99 * 100) / 2)

-- The proof theorem
theorem problem : ∀ d_a d_b : ℚ, condition d_a d_b → sequence_sum d_a d_b = 13000 :=
by
  intros d_a d_b h_cond
  sorry

end NUMINAMATH_GPT_problem_l1341_134166


namespace NUMINAMATH_GPT_Zilla_savings_l1341_134129

/-- Zilla's monthly savings based on her spending distributions -/
theorem Zilla_savings
  (rent : ℚ) (monthly_earnings_percentage : ℚ)
  (other_expenses_fraction : ℚ) (monthly_rent : ℚ)
  (monthly_expenses : ℚ) (total_monthly_earnings : ℚ)
  (half_monthly_earnings : ℚ) (savings : ℚ)
  (h1 : rent = 133)
  (h2 : monthly_earnings_percentage = 0.07)
  (h3 : other_expenses_fraction = 0.5)
  (h4 : total_monthly_earnings = monthly_rent / monthly_earnings_percentage)
  (h5 : half_monthly_earnings = total_monthly_earnings * other_expenses_fraction)
  (h6 : savings = total_monthly_earnings - (monthly_rent + half_monthly_earnings))
  : savings = 817 :=
sorry

end NUMINAMATH_GPT_Zilla_savings_l1341_134129


namespace NUMINAMATH_GPT_total_students_l1341_134113

/-- Definition of the problem's conditions as Lean statements -/
def left_col := 8
def right_col := 14
def front_row := 7
def back_row := 15

/-- The total number of columns calculated from Eunji's column positions -/
def total_columns := left_col + right_col - 1
/-- The total number of rows calculated from Eunji's row positions -/
def total_rows := front_row + back_row - 1

/-- Lean statement showing the total number of students given the conditions -/
theorem total_students : total_columns * total_rows = 441 := by
  sorry

end NUMINAMATH_GPT_total_students_l1341_134113


namespace NUMINAMATH_GPT_inverse_proposition_true_l1341_134136

-- Define a rectangle and a square
structure Rectangle where
  length : ℝ
  width  : ℝ

def is_square (r : Rectangle) : Prop :=
  r.length = r.width ∧ r.length > 0 ∧ r.width > 0

-- Define the condition that a rectangle with equal adjacent sides is a square
def rectangle_with_equal_adjacent_sides_is_square : Prop :=
  ∀ r : Rectangle, r.length = r.width → is_square r

-- Define the inverse proposition that a square is a rectangle with equal adjacent sides
def square_is_rectangle_with_equal_adjacent_sides : Prop :=
  ∀ r : Rectangle, is_square r → r.length = r.width

-- The proof statement
theorem inverse_proposition_true :
  rectangle_with_equal_adjacent_sides_is_square → square_is_rectangle_with_equal_adjacent_sides :=
by
  sorry

end NUMINAMATH_GPT_inverse_proposition_true_l1341_134136


namespace NUMINAMATH_GPT_find_x_l1341_134123

-- Definitions of the conditions in Lean 4
def angle_sum_180 (A B C : ℝ) : Prop := A + B + C = 180
def angle_BAC_eq_90 (A : ℝ) : Prop := A = 90
def angle_BCA_eq_2x (C x : ℝ) : Prop := C = 2 * x
def angle_ABC_eq_3x (B x : ℝ) : Prop := B = 3 * x

-- The theorem we need to prove
theorem find_x (A B C x : ℝ) 
  (h1 : angle_sum_180 A B C) 
  (h2 : angle_BAC_eq_90 A)
  (h3 : angle_BCA_eq_2x C x) 
  (h4 : angle_ABC_eq_3x B x) : x = 18 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l1341_134123


namespace NUMINAMATH_GPT_base_b_arithmetic_l1341_134125

theorem base_b_arithmetic (b : ℕ) (h1 : 4 + 3 = 7) (h2 : 6 + 2 = 8) (h3 : 4 + 6 = 10) (h4 : 3 + 4 + 1 = 8) : b = 9 :=
  sorry

end NUMINAMATH_GPT_base_b_arithmetic_l1341_134125


namespace NUMINAMATH_GPT_mother_present_age_l1341_134169

def person_present_age (P M : ℕ) : Prop :=
  P = (2 / 5) * M

def person_age_in_10_years (P M : ℕ) : Prop :=
  P + 10 = (1 / 2) * (M + 10)

theorem mother_present_age (P M : ℕ) (h1 : person_present_age P M) (h2 : person_age_in_10_years P M) : M = 50 :=
sorry

end NUMINAMATH_GPT_mother_present_age_l1341_134169


namespace NUMINAMATH_GPT_roots_imply_sum_l1341_134121

theorem roots_imply_sum (a b c x1 x2 : ℝ) (hneq : a ≠ 0) (hroots : a * x1 ^ 2 + b * x1 + c = 0 ∧ a * x2 ^ 2 + b * x2 + c = 0) :
  x1 + x2 = -b / a :=
sorry

end NUMINAMATH_GPT_roots_imply_sum_l1341_134121


namespace NUMINAMATH_GPT_radius_of_roots_circle_l1341_134143

theorem radius_of_roots_circle (z : ℂ) (hz : (z - 2)^6 = 64 * z^6) : ∃ r : ℝ, r = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_roots_circle_l1341_134143


namespace NUMINAMATH_GPT_ratio_paid_back_to_initial_debt_l1341_134179

def initial_debt : ℕ := 40
def still_owed : ℕ := 30
def paid_back (initial_debt still_owed : ℕ) : ℕ := initial_debt - still_owed

theorem ratio_paid_back_to_initial_debt
  (initial_debt still_owed : ℕ) :
  (paid_back initial_debt still_owed : ℚ) / initial_debt = 1 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_paid_back_to_initial_debt_l1341_134179


namespace NUMINAMATH_GPT_estimated_red_balls_l1341_134119

theorem estimated_red_balls
  (total_balls : ℕ)
  (total_draws : ℕ)
  (red_draws : ℕ)
  (h_total_balls : total_balls = 12)
  (h_total_draws : total_draws = 200)
  (h_red_draws : red_draws = 50) :
  red_draws * total_balls = total_draws * 3 :=
by
  sorry

end NUMINAMATH_GPT_estimated_red_balls_l1341_134119


namespace NUMINAMATH_GPT_problem_seven_integers_l1341_134195

theorem problem_seven_integers (a b c d e f g : ℕ) 
  (h1 : b = a + 1) 
  (h2 : c = b + 1) 
  (h3 : d = c + 1) 
  (h4 : e = d + 1) 
  (h5 : f = e + 1) 
  (h6 : g = f + 1) 
  (h_sum : a + b + c + d + e + f + g = 2017) : 
  a = 286 ∨ g = 286 :=
sorry

end NUMINAMATH_GPT_problem_seven_integers_l1341_134195


namespace NUMINAMATH_GPT_sum_divisible_by_5_and_7_remainder_12_l1341_134120

theorem sum_divisible_by_5_and_7_remainder_12 :
  let a := 105
  let d := 35
  let n := 2013
  let S := (n * (2 * a + (n - 1) * d)) / 2
  S % 12 = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_divisible_by_5_and_7_remainder_12_l1341_134120


namespace NUMINAMATH_GPT_smallest_n_l1341_134138

theorem smallest_n (n : ℕ) (h : 5 * n % 26 = 220 % 26) : n = 18 :=
by
  -- Initial congruence simplification
  have h1 : 220 % 26 = 12 := by norm_num
  rw [h1] at h
  -- Reformulation of the problem
  have h2 : 5 * n % 26 = 12 := h
  -- Conclude the smallest n
  sorry

end NUMINAMATH_GPT_smallest_n_l1341_134138


namespace NUMINAMATH_GPT_tree_initial_height_l1341_134111

theorem tree_initial_height (H : ℝ) (C : ℝ) (P : H + 6 = (H + 4) + 1/4 * (H + 4) ∧ C = 1) : H = 4 :=
by
  let H := 4
  sorry

end NUMINAMATH_GPT_tree_initial_height_l1341_134111


namespace NUMINAMATH_GPT_books_cost_l1341_134134

theorem books_cost (total_cost_three_books cost_seven_books : ℕ) 
  (h₁ : total_cost_three_books = 45)
  (h₂ : cost_seven_books = 7 * (total_cost_three_books / 3)) : 
  cost_seven_books = 105 :=
  sorry

end NUMINAMATH_GPT_books_cost_l1341_134134


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1341_134102

theorem sufficient_but_not_necessary (x : ℝ) :
  (x^2 > 1) → (1 / x < 1) ∧ ¬(1 / x < 1 → x^2 > 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1341_134102


namespace NUMINAMATH_GPT_spoiled_milk_percentage_l1341_134185

theorem spoiled_milk_percentage (p_egg p_flour p_all_good : ℝ) (h_egg : p_egg = 0.40) (h_flour : p_flour = 0.75) (h_all_good : p_all_good = 0.24) : 
  (1 - (p_all_good / (p_egg * p_flour))) = 0.20 :=
by
  sorry

end NUMINAMATH_GPT_spoiled_milk_percentage_l1341_134185


namespace NUMINAMATH_GPT_train_pass_bridge_time_l1341_134174

noncomputable def length_of_train : ℝ := 485
noncomputable def length_of_bridge : ℝ := 140
noncomputable def speed_of_train_kmph : ℝ := 45 
noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmph * (1000 / 3600)

theorem train_pass_bridge_time :
  (length_of_train + length_of_bridge) / speed_of_train_mps = 50 :=
by
  sorry

end NUMINAMATH_GPT_train_pass_bridge_time_l1341_134174


namespace NUMINAMATH_GPT_false_implies_not_all_ripe_l1341_134152

def all_ripe (basket : Type) [Nonempty basket] (P : basket → Prop) : Prop :=
  ∀ x : basket, P x

theorem false_implies_not_all_ripe
  (basket : Type)
  [Nonempty basket]
  (P : basket → Prop)
  (h : ¬ all_ripe basket P) :
  (∃ x, ¬ P x) ∧ ¬ all_ripe basket P :=
by
  sorry

end NUMINAMATH_GPT_false_implies_not_all_ripe_l1341_134152


namespace NUMINAMATH_GPT_quadratic_intersects_xaxis_once_l1341_134167

theorem quadratic_intersects_xaxis_once (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k = 0) ↔ k = 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_intersects_xaxis_once_l1341_134167


namespace NUMINAMATH_GPT_repeating_decimal_sum_l1341_134101

open Real

noncomputable def repeating_decimal_to_fraction (d: ℕ) : ℚ :=
  if d = 3 then 1/3 else if d = 7 then 7/99 else if d = 9 then 1/111 else 0 -- specific case of 3, 7, 9.

theorem repeating_decimal_sum:
  let x := repeating_decimal_to_fraction 3
  let y := repeating_decimal_to_fraction 7
  let z := repeating_decimal_to_fraction 9
  x + y + z = 499 / 1189 :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_repeating_decimal_sum_l1341_134101


namespace NUMINAMATH_GPT_sum_of_series_l1341_134170

theorem sum_of_series : (1 / (1 * 2 * 3) + 1 / (2 * 3 * 4) + 1 / (3 * 4 * 5) + 1 / (4 * 5 * 6)) = 7 / 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_series_l1341_134170


namespace NUMINAMATH_GPT_sum_of_three_numbers_l1341_134162

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a + b = 35) (h2 : b + c = 42) (h3 : c + a = 58) :
  a + b + c = 67.5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l1341_134162


namespace NUMINAMATH_GPT_mean_median_difference_l1341_134172

open Real

/-- In a class of 100 students, these are the distributions of scores:
  - 10% scored 60 points
  - 30% scored 75 points
  - 25% scored 80 points
  - 20% scored 90 points
  - 15% scored 100 points

Prove that the difference between the mean and the median scores is 1.5. -/
theorem mean_median_difference :
  let total_students := 100 
  let score_60 := 0.10 * total_students
  let score_75 := 0.30 * total_students
  let score_80 := 0.25 * total_students
  let score_90 := 0.20 * total_students
  let score_100 := (100 - (score_60 + score_75 + score_80 + score_90))
  let median := 80
  let mean := (60 * score_60 + 75 * score_75 + 80 * score_80 + 90 * score_90 + 100 * score_100) / total_students
  mean - median = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_mean_median_difference_l1341_134172


namespace NUMINAMATH_GPT_problem_solution_l1341_134115

theorem problem_solution
  (a1 a2 a3: ℝ)
  (a_arith_seq : ∃ d, a1 = 1 + d ∧ a2 = a1 + d ∧ a3 = a2 + d ∧ 9 = a3 + d)
  (b1 b2 b3: ℝ)
  (b_geo_seq : ∃ r, r > 0 ∧ b1 = -9 * r ∧ b2 = b1 * r ∧ b3 = b2 * r ∧ -1 = b3 * r) :
  (b2 / (a1 + a3) = -3 / 10) :=
by
  -- Placeholder for the proof, not required in this context
  sorry

end NUMINAMATH_GPT_problem_solution_l1341_134115


namespace NUMINAMATH_GPT_max_questions_wrong_to_succeed_l1341_134183

theorem max_questions_wrong_to_succeed :
  ∀ (total_questions : ℕ) (passing_percentage : ℚ),
  total_questions = 50 →
  passing_percentage = 0.75 →
  ∃ (max_wrong : ℕ), max_wrong = 12 ∧
    (total_questions - max_wrong) ≥ passing_percentage * total_questions := by
  intro total_questions passing_percentage h1 h2
  use 12
  constructor
  . rfl
  . sorry  -- Proof omitted

end NUMINAMATH_GPT_max_questions_wrong_to_succeed_l1341_134183


namespace NUMINAMATH_GPT_contestants_order_l1341_134141

variables (G E H F : ℕ) -- Scores of the participants, given that they are nonnegative

theorem contestants_order (h1 : E + G = F + H) (h2 : F + E = H + G) (h3 : G > E + F) : 
  G ≥ E ∧ G ≥ H ∧ G ≥ F ∧ E = H ∧ E ≥ F :=
by {
  sorry
}

end NUMINAMATH_GPT_contestants_order_l1341_134141


namespace NUMINAMATH_GPT_value_of_y_l1341_134127

theorem value_of_y (x y : ℝ) (h1 : x * y = 9) (h2 : x / y = 36) : y = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_l1341_134127


namespace NUMINAMATH_GPT_Pria_drove_372_miles_l1341_134161

theorem Pria_drove_372_miles (advertisement_mileage : ℕ) (tank_capacity : ℕ) (mileage_difference : ℕ) 
(h1 : advertisement_mileage = 35) 
(h2 : tank_capacity = 12) 
(h3 : mileage_difference = 4) : 
(advertisement_mileage - mileage_difference) * tank_capacity = 372 :=
by sorry

end NUMINAMATH_GPT_Pria_drove_372_miles_l1341_134161


namespace NUMINAMATH_GPT_kylie_total_beads_used_l1341_134160

noncomputable def beads_monday_necklaces : ℕ := 10 * 20
noncomputable def beads_tuesday_necklaces : ℕ := 2 * 20
noncomputable def beads_wednesday_bracelets : ℕ := 5 * 10
noncomputable def beads_thursday_earrings : ℕ := 3 * 5
noncomputable def beads_friday_anklets : ℕ := 4 * 8
noncomputable def beads_friday_rings : ℕ := 6 * 7

noncomputable def total_beads_used : ℕ :=
  beads_monday_necklaces +
  beads_tuesday_necklaces +
  beads_wednesday_bracelets +
  beads_thursday_earrings +
  beads_friday_anklets +
  beads_friday_rings

theorem kylie_total_beads_used : total_beads_used = 379 := by
  sorry

end NUMINAMATH_GPT_kylie_total_beads_used_l1341_134160


namespace NUMINAMATH_GPT_water_percentage_in_fresh_grapes_l1341_134191

theorem water_percentage_in_fresh_grapes 
  (P : ℝ) -- the percentage of water in fresh grapes
  (fresh_grapes_weight : ℝ := 40) -- weight of fresh grapes in kg
  (dry_grapes_weight : ℝ := 5) -- weight of dry grapes in kg
  (dried_grapes_water_percentage : ℝ := 20) -- percentage of water in dried grapes
  (solid_content : ℝ := 4) -- solid content in both fresh and dried grapes in kg
  : P = 90 :=
by
  sorry

end NUMINAMATH_GPT_water_percentage_in_fresh_grapes_l1341_134191


namespace NUMINAMATH_GPT_value_of_S_l1341_134153

def pseudocode_value : ℕ := 1
def increment (S I : ℕ) : ℕ := S + I

def loop_steps : ℕ :=
  let S := pseudocode_value
  let S := increment S 1
  let S := increment S 3
  let S := increment S 5
  let S := increment S 7
  S

theorem value_of_S : loop_steps = 17 :=
  by sorry

end NUMINAMATH_GPT_value_of_S_l1341_134153


namespace NUMINAMATH_GPT_common_term_sequence_7n_l1341_134128

theorem common_term_sequence_7n (n : ℕ) : 
  ∃ a_n : ℕ, a_n = (7 / 9) * (10^n - 1) :=
by
  sorry

end NUMINAMATH_GPT_common_term_sequence_7n_l1341_134128


namespace NUMINAMATH_GPT_paint_room_together_l1341_134192

variable (t : ℚ)
variable (Doug_rate : ℚ := 1/5)
variable (Dave_rate : ℚ := 1/7)
variable (Diana_rate : ℚ := 1/6)
variable (Combined_rate : ℚ := Doug_rate + Dave_rate + Diana_rate)
variable (break_time : ℚ := 2)

theorem paint_room_together:
  Combined_rate * (t - break_time) = 1 :=
sorry

end NUMINAMATH_GPT_paint_room_together_l1341_134192


namespace NUMINAMATH_GPT_book_total_pages_l1341_134100

theorem book_total_pages (x : ℝ) 
  (h1 : ∀ d1 : ℝ, d1 = x * (1/6) + 10)
  (h2 : ∀ remaining1 : ℝ, remaining1 = x - d1)
  (h3 : ∀ d2 : ℝ, d2 = remaining1 * (1/5) + 12)
  (h4 : ∀ remaining2 : ℝ, remaining2 = remaining1 - d2)
  (h5 : ∀ d3 : ℝ, d3 = remaining2 * (1/4) + 14)
  (h6 : ∀ remaining3 : ℝ, remaining3 = remaining2 - d3)
  (h7 : remaining3 = 52) : x = 169 := sorry

end NUMINAMATH_GPT_book_total_pages_l1341_134100


namespace NUMINAMATH_GPT_hcf_of_two_numbers_l1341_134126

theorem hcf_of_two_numbers (A B : ℕ) (h1 : Nat.lcm A B = 750) (h2 : A * B = 18750) : Nat.gcd A B = 25 :=
by
  sorry

end NUMINAMATH_GPT_hcf_of_two_numbers_l1341_134126


namespace NUMINAMATH_GPT_range_of_k_l1341_134189

theorem range_of_k (k : ℝ) : (∃ x y : ℝ, x^2 + k * y^2 = 2) ∧ (∀ x y : ℝ, y ≠ 0 → x^2 + k * y^2 = 2 → (x = 0 ∧ (∃ a : ℝ, a > 1 ∧ y = a))) → 0 < k ∧ k < 1 :=
sorry

end NUMINAMATH_GPT_range_of_k_l1341_134189


namespace NUMINAMATH_GPT_count_positive_even_multiples_of_3_less_than_5000_perfect_squares_l1341_134158

theorem count_positive_even_multiples_of_3_less_than_5000_perfect_squares :
  ∃ n : ℕ, (n = 11) ∧ ∀ k : ℕ, (k < 5000) → (k % 2 = 0) → (k % 3 = 0) → (∃ m : ℕ, k = m * m) → k ≤ 36 * 11 * 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_count_positive_even_multiples_of_3_less_than_5000_perfect_squares_l1341_134158


namespace NUMINAMATH_GPT_points_player_1_after_13_rotations_l1341_134105

variable (table : List ℕ) (players : Fin 16 → ℕ)

axiom round_rotating_table : table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
axiom points_player_5 : players 5 = 72
axiom points_player_9 : players 9 = 84

theorem points_player_1_after_13_rotations : players 1 = 20 := 
  sorry

end NUMINAMATH_GPT_points_player_1_after_13_rotations_l1341_134105


namespace NUMINAMATH_GPT_cost_price_per_meter_l1341_134139

theorem cost_price_per_meter (selling_price : ℝ) (total_meters : ℕ) (profit_per_meter : ℝ)
  (h1 : selling_price = 8925)
  (h2 : total_meters = 85)
  (h3 : profit_per_meter = 5) :
  (selling_price - total_meters * profit_per_meter) / total_meters = 100 := by
  sorry

end NUMINAMATH_GPT_cost_price_per_meter_l1341_134139


namespace NUMINAMATH_GPT_last_two_videos_length_l1341_134188

noncomputable def ad1 : ℕ := 45
noncomputable def ad2 : ℕ := 30
noncomputable def pause1 : ℕ := 45
noncomputable def pause2 : ℕ := 30
noncomputable def video1 : ℕ := 120
noncomputable def video2 : ℕ := 270
noncomputable def total_time : ℕ := 960

theorem last_two_videos_length : 
    ∃ v : ℕ, 
    v = 210 ∧ 
    total_time = ad1 + ad2 + video1 + video2 + pause1 + pause2 + 2 * v :=
by
  sorry

end NUMINAMATH_GPT_last_two_videos_length_l1341_134188


namespace NUMINAMATH_GPT_system_no_solution_iff_n_eq_neg_half_l1341_134178

theorem system_no_solution_iff_n_eq_neg_half (x y z n : ℝ) :
  (¬ ∃ x y z, 2 * n * x + y = 2 ∧ n * y + 2 * z = 2 ∧ x + 2 * n * z = 2) ↔ n = -1/2 := by
  sorry

end NUMINAMATH_GPT_system_no_solution_iff_n_eq_neg_half_l1341_134178


namespace NUMINAMATH_GPT_tangent_line_circle_l1341_134103

theorem tangent_line_circle (m : ℝ) (h : m > 0) : 
  (∀ x y : ℝ, x + y = 2 ↔ x^2 + y^2 = m) → m = 2 :=
by
  intro h_tangent
  sorry

end NUMINAMATH_GPT_tangent_line_circle_l1341_134103


namespace NUMINAMATH_GPT_cards_from_country_correct_l1341_134173

def total_cards : ℝ := 403.0
def cards_from_home : ℝ := 287.0
def cards_from_country : ℝ := total_cards - cards_from_home

theorem cards_from_country_correct : cards_from_country = 116.0 := by
  -- proof to be added
  sorry

end NUMINAMATH_GPT_cards_from_country_correct_l1341_134173


namespace NUMINAMATH_GPT_apples_total_l1341_134124

theorem apples_total
    (cecile_apples : ℕ := 15)
    (diane_apples_more : ℕ := 20) :
    (cecile_apples + (cecile_apples + diane_apples_more)) = 50 :=
by
  sorry

end NUMINAMATH_GPT_apples_total_l1341_134124


namespace NUMINAMATH_GPT_min_k_value_l1341_134155

noncomputable def minimum_k_condition (x y z k : ℝ) : Prop :=
  k * (x^2 - x + 1) * (y^2 - y + 1) * (z^2 - z + 1) ≥ (x * y * z)^2 - (x * y * z) + 1

theorem min_k_value :
  ∀ x y z : ℝ, x ≤ 0 → y ≤ 0 → z ≤ 0 → minimum_k_condition x y z (16 / 9) :=
by
  sorry

end NUMINAMATH_GPT_min_k_value_l1341_134155


namespace NUMINAMATH_GPT_max_cubes_fit_l1341_134147

-- Define the conditions
def box_volume (length : ℕ) (width : ℕ) (height : ℕ) : ℕ := length * width * height
def cube_volume : ℕ := 27
def total_cubes (V_box : ℕ) (V_cube : ℕ) : ℕ := V_box / V_cube

-- Statement of the problem
theorem max_cubes_fit (length width height : ℕ) (V_box : ℕ) (V_cube q : ℕ) :
  length = 8 → width = 9 → height = 12 → V_box = box_volume length width height →
  V_cube = cube_volume → q = total_cubes V_box V_cube → q = 32 :=
by sorry

end NUMINAMATH_GPT_max_cubes_fit_l1341_134147


namespace NUMINAMATH_GPT_find_star_l1341_134117

theorem find_star :
  ∃ (star : ℤ), 45 - ( 28 - ( 37 - ( 15 - star ) ) ) = 56 ∧ star = 17 :=
by
  sorry

end NUMINAMATH_GPT_find_star_l1341_134117


namespace NUMINAMATH_GPT_range_of_a_l1341_134118

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + 2 * a > 0) ↔ (0 < a ∧ a < 8) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1341_134118


namespace NUMINAMATH_GPT_find_red_coin_l1341_134157

/- Define the function f(n) as the minimum number of scans required to determine the red coin
   - out of n coins with the given conditions.
   - Seyed has 998 white coins, 1 red coin, and 1 red-white coin.
-/

def f (n : Nat) : Nat := sorry

/- The main theorem to be proved: There exists an algorithm that can find the red coin using 
   the scanner at most 17 times for 1000 coins.
-/

theorem find_red_coin (n : Nat) (h : n = 1000) : f n ≤ 17 := sorry

end NUMINAMATH_GPT_find_red_coin_l1341_134157


namespace NUMINAMATH_GPT_factor_correct_l1341_134171

-- Define the polynomial p(x)
def p (x : ℤ) : ℤ := 6 * (x + 4) * (x + 7) * (x + 9) * (x + 11) - 5 * x^2

-- Define the potential factors of p(x)
def f1 (x : ℤ) : ℤ := 3 * x^2 + 93 * x
def f2 (x : ℤ) : ℤ := 2 * x^2 + 178 * x + 5432

theorem factor_correct : ∀ x : ℤ, p x = f1 x * f2 x := by
  sorry

end NUMINAMATH_GPT_factor_correct_l1341_134171


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_not_necessary_condition_l1341_134156

theorem sufficient_but_not_necessary 
  (α : ℝ) (h : Real.sin α = Real.cos α) :
  Real.cos (2 * α) = 0 :=
by sorry

theorem not_necessary_condition 
  (α : ℝ) (h : Real.cos (2 * α) = 0) :
  ∃ β : ℝ, Real.sin β ≠ Real.cos β :=
by sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_not_necessary_condition_l1341_134156


namespace NUMINAMATH_GPT_cube_volume_l1341_134165

theorem cube_volume (A : ℝ) (hA : A = 96) (s : ℝ) (hS : A = 6 * s^2) : s^3 = 64 := by
  sorry

end NUMINAMATH_GPT_cube_volume_l1341_134165


namespace NUMINAMATH_GPT_retailer_discount_problem_l1341_134140

theorem retailer_discount_problem
  (CP MP SP : ℝ) 
  (h1 : CP = 100)
  (h2 : MP = CP + (0.65 * CP))
  (h3 : SP = CP + (0.2375 * CP)) :
  (MP - SP) / MP * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_retailer_discount_problem_l1341_134140


namespace NUMINAMATH_GPT_company_a_taxis_l1341_134190

variable (a b : ℕ)

theorem company_a_taxis
  (h1 : 5 * a < 56)
  (h2 : 6 * a > 56)
  (h3 : 4 * b < 56)
  (h4 : 5 * b > 56)
  (h5 : b = a + 3) :
  a = 10 := by
  sorry

end NUMINAMATH_GPT_company_a_taxis_l1341_134190


namespace NUMINAMATH_GPT_verify_digits_l1341_134109

theorem verify_digits :
  ∀ (a b c d e f g h : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
  f ≠ g ∧ f ≠ h ∧
  g ≠ h ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 →
  (10 * a + b) - (10 * c + d) = 10 * e + d →
  e * f = 10 * d + c →
  (10 * g + d) + (10 * g + b) = 10 * h + c →
  a = 9 ∧ b = 8 ∧ c = 2 ∧ d = 4 ∧ e = 7 ∧ f = 6 ∧ g = 1 ∧ h = 3 :=
by
  intros a b c d e f g h
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_verify_digits_l1341_134109


namespace NUMINAMATH_GPT_odd_prime_divisibility_two_prime_divisibility_l1341_134180

theorem odd_prime_divisibility (p a n : ℕ) (hp : p % 2 = 1) (hp_prime : Nat.Prime p)
  (ha : a > 0) (hn : n > 0) (div_cond : p^n ∣ a^p - 1) : p^(n-1) ∣ a - 1 :=
sorry

theorem two_prime_divisibility (a n : ℕ) (ha : a > 0) (hn : n > 0) (div_cond : 2^n ∣ a^2 - 1) : ¬ 2^(n-1) ∣ a - 1 :=
sorry

end NUMINAMATH_GPT_odd_prime_divisibility_two_prime_divisibility_l1341_134180


namespace NUMINAMATH_GPT_subsequent_flights_requirements_l1341_134149

-- Define the initial conditions
def late_flights : ℕ := 1
def on_time_flights : ℕ := 3
def total_initial_flights : ℕ := late_flights + on_time_flights

-- Define the number of subsequent flights needed
def subsequent_flights_needed (x : ℕ) : Prop :=
  let total_flights := total_initial_flights + x
  let on_time_total := on_time_flights + x
  (on_time_total : ℚ) / (total_flights : ℚ) > 0.40

-- State the theorem to prove
theorem subsequent_flights_requirements:
  ∃ x : ℕ, subsequent_flights_needed x := sorry

end NUMINAMATH_GPT_subsequent_flights_requirements_l1341_134149


namespace NUMINAMATH_GPT_ratio_of_u_to_v_l1341_134137

theorem ratio_of_u_to_v {b u v : ℝ} 
  (h1 : b ≠ 0)
  (h2 : 0 = 12 * u + b)
  (h3 : 0 = 8 * v + b) : 
  u / v = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_u_to_v_l1341_134137


namespace NUMINAMATH_GPT_range_of_a_l1341_134186

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + (a + 1) * x + 1 < 0) → (a < -3 ∨ a > 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1341_134186


namespace NUMINAMATH_GPT_symmetric_circle_equation_l1341_134154

-- Define the original circle and the line of symmetry
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 2
def line_of_symmetry (x y : ℝ) : Prop := 2 * x - y + 3 = 0

-- Proving the equation of the symmetric circle
theorem symmetric_circle_equation :
  (∀ x y : ℝ, original_circle x y ↔ (x + 3)^2 + (y - 2)^2 = 2) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_circle_equation_l1341_134154


namespace NUMINAMATH_GPT_smallest_integer_cube_ends_in_576_l1341_134116

theorem smallest_integer_cube_ends_in_576 : ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 576 ∧ ∀ m : ℕ, m > 0 → m^3 % 1000 = 576 → m ≥ n := 
by
  sorry

end NUMINAMATH_GPT_smallest_integer_cube_ends_in_576_l1341_134116


namespace NUMINAMATH_GPT_find_y_l1341_134168

theorem find_y (x y : ℤ) (h1 : x = -4) (h2 : x^2 + 3 * x + 7 = y - 5) : y = 16 := 
by
  sorry

end NUMINAMATH_GPT_find_y_l1341_134168


namespace NUMINAMATH_GPT_degree_of_resulting_poly_l1341_134132

-- Define the polynomials involved in the problem
noncomputable def poly_1 : Polynomial ℝ := 3 * Polynomial.X ^ 5 + 2 * Polynomial.X ^ 3 - Polynomial.X - 16
noncomputable def poly_2 : Polynomial ℝ := 4 * Polynomial.X ^ 11 - 8 * Polynomial.X ^ 8 + 6 * Polynomial.X ^ 5 + 35
noncomputable def poly_3 : Polynomial ℝ := (Polynomial.X ^ 2 + 4) ^ 8

-- Define the resulting polynomial
noncomputable def resulting_poly : Polynomial ℝ :=
  poly_1 * poly_2 - poly_3

-- The goal is to prove that the degree of the resulting polynomial is 16
theorem degree_of_resulting_poly : resulting_poly.degree = 16 := 
sorry

end NUMINAMATH_GPT_degree_of_resulting_poly_l1341_134132


namespace NUMINAMATH_GPT_leo_weight_proof_l1341_134104

def Leo_s_current_weight (L K : ℝ) := 
  L + 10 = 1.5 * K ∧ L + K = 170 → L = 98

theorem leo_weight_proof : ∀ (L K : ℝ), L + 10 = 1.5 * K ∧ L + K = 170 → L = 98 := 
by 
  intros L K h
  sorry

end NUMINAMATH_GPT_leo_weight_proof_l1341_134104


namespace NUMINAMATH_GPT_area_of_triangle_ADE_l1341_134187

noncomputable def triangle_area (A B C: ℝ × ℝ) : ℝ :=
  1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem area_of_triangle_ADE (A B D E F : ℝ × ℝ) (h₁ : A.1 = 0 ∧ A.2 = 0) (h₂ : B.1 = 8 ∧ B.2 = 0)
  (h₃ : D.1 = 8 ∧ D.2= 8) (h₄ : E.1 = 4 * 3 / 5 ∧ E.2 = 0) 
  (h₅ : F.1 = 0 ∧ F.2 = 12) :
  triangle_area A D E = 288 / 25 := 
sorry

end NUMINAMATH_GPT_area_of_triangle_ADE_l1341_134187


namespace NUMINAMATH_GPT_rectangle_width_l1341_134110

theorem rectangle_width (L W : ℝ) 
  (h1 : L * W = 300)
  (h2 : 2 * L + 2 * W = 70) : 
  W = 15 :=
by 
  -- We prove the width W of the rectangle is 15 meters.
  sorry

end NUMINAMATH_GPT_rectangle_width_l1341_134110


namespace NUMINAMATH_GPT_books_in_special_collection_l1341_134112

theorem books_in_special_collection (B : ℕ) :
  (∃ returned not_returned loaned_out_end  : ℝ, 
    loaned_out_end = 54 ∧ 
    returned = 0.65 * 60.00000000000001 ∧ 
    not_returned = 60.00000000000001 - returned ∧ 
    B = loaned_out_end + not_returned) → 
  B = 75 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_books_in_special_collection_l1341_134112


namespace NUMINAMATH_GPT_find_dividend_l1341_134135

theorem find_dividend (q : ℕ) (d : ℕ) (r : ℕ) (D : ℕ) 
  (h_q : q = 15000)
  (h_d : d = 82675)
  (h_r : r = 57801)
  (h_D : D = 1240182801) :
  D = d * q + r := by 
  sorry

end NUMINAMATH_GPT_find_dividend_l1341_134135


namespace NUMINAMATH_GPT_find_constants_l1341_134122

theorem find_constants
  (k m n : ℝ)
  (h : -x^3 + (k + 7) * x^2 + m * x - 8 = -(x - 2) * (x - 4) * (x - n)) :
  k = 7 ∧ m = 2 ∧ n = 1 :=
sorry

end NUMINAMATH_GPT_find_constants_l1341_134122


namespace NUMINAMATH_GPT_p_plus_q_eq_10_l1341_134130

theorem p_plus_q_eq_10 (p q : ℕ) (hp : p > q) (hpq1 : p < 10) (hpq2 : q < 10)
  (h : p.factorial / q.factorial = 840) : p + q = 10 :=
by
  sorry

end NUMINAMATH_GPT_p_plus_q_eq_10_l1341_134130


namespace NUMINAMATH_GPT_percentage_of_masters_l1341_134145

-- Definition of given conditions
def average_points_juniors := 22
def average_points_masters := 47
def overall_average_points := 41

-- Problem statement
theorem percentage_of_masters (x y : ℕ) (hx : x ≥ 0) (hy : y ≥ 0) 
    (h_avg_juniors : 22 * x = average_points_juniors * x)
    (h_avg_masters : 47 * y = average_points_masters * y)
    (h_overall_average : 22 * x + 47 * y = overall_average_points * (x + y)) : 
    (y : ℚ) / (x + y) * 100 = 76 := 
sorry

end NUMINAMATH_GPT_percentage_of_masters_l1341_134145


namespace NUMINAMATH_GPT_triangle_side_lengths_log_l1341_134181

theorem triangle_side_lengths_log (m : ℕ) (log15 log81 logm : ℝ)
  (h1 : log15 = Real.log 15 / Real.log 10)
  (h2 : log81 = Real.log 81 / Real.log 10)
  (h3 : logm = Real.log m / Real.log 10)
  (h4 : 0 < log15 ∧ 0 < log81 ∧ 0 < logm)
  (h5 : log15 + log81 > logm)
  (h6 : log15 + logm > log81)
  (h7 : log81 + logm > log15)
  (h8 : m > 0) :
  6 ≤ m ∧ m < 1215 → 
  ∃ n : ℕ, n = 1215 - 6 ∧ n = 1209 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_lengths_log_l1341_134181


namespace NUMINAMATH_GPT_store_loss_l1341_134114

noncomputable def calculation (x y : ℕ) : ℤ :=
  let revenue : ℕ := 60 * 2
  let cost : ℕ := x + y
  revenue - cost

theorem store_loss (x y : ℕ) (hx : (60 - x) * 2 = x) (hy : (y - 60) * 2 = y) :
  calculation x y = -40 := by
    sorry

end NUMINAMATH_GPT_store_loss_l1341_134114


namespace NUMINAMATH_GPT_marcie_and_martin_in_picture_l1341_134197

noncomputable def marcie_prob_in_picture : ℚ :=
  let marcie_lap_time := 100
  let martin_lap_time := 75
  let start_time := 720
  let end_time := 780
  let picture_duration := 60
  let marcie_position_720 := (720 % marcie_lap_time) / marcie_lap_time
  let marcie_in_pic_start := 0
  let marcie_in_pic_end := 20 + 33 + 1/3
  let martin_position_720 := (720 % martin_lap_time) / martin_lap_time
  let martin_in_pic_start := 20
  let martin_in_pic_end := 45 + 25
  let overlap_start := max marcie_in_pic_start martin_in_pic_start
  let overlap_end := min marcie_in_pic_end martin_in_pic_end
  let overlap_duration := overlap_end - overlap_start
  overlap_duration / picture_duration

theorem marcie_and_martin_in_picture :
  marcie_prob_in_picture = 111 / 200 :=
by
  sorry

end NUMINAMATH_GPT_marcie_and_martin_in_picture_l1341_134197


namespace NUMINAMATH_GPT_average_salary_decrease_l1341_134148

theorem average_salary_decrease 
    (avg_wage_illiterate_initial : ℝ)
    (avg_wage_illiterate_new : ℝ)
    (num_illiterate : ℕ)
    (num_literate : ℕ)
    (num_total : ℕ)
    (total_decrease : ℝ) :
    avg_wage_illiterate_initial = 25 →
    avg_wage_illiterate_new = 10 →
    num_illiterate = 20 →
    num_literate = 10 →
    num_total = num_illiterate + num_literate →
    total_decrease = (avg_wage_illiterate_initial - avg_wage_illiterate_new) * num_illiterate →
    total_decrease / num_total = 10 :=
by
  intros avg_wage_illiterate_initial_eq avg_wage_illiterate_new_eq num_illiterate_eq num_literate_eq num_total_eq total_decrease_eq
  sorry

end NUMINAMATH_GPT_average_salary_decrease_l1341_134148


namespace NUMINAMATH_GPT_notebooks_distributed_l1341_134159

theorem notebooks_distributed  (C : ℕ) (N : ℕ) 
  (h1 : N = C^2 / 8) 
  (h2 : N = 8 * C) : 
  N = 512 :=
by 
  sorry

end NUMINAMATH_GPT_notebooks_distributed_l1341_134159


namespace NUMINAMATH_GPT_eat_both_veg_nonveg_l1341_134196

theorem eat_both_veg_nonveg (total_veg only_veg : ℕ) (h1 : total_veg = 31) (h2 : only_veg = 19) :
  (total_veg - only_veg) = 12 :=
by
  have h3 : total_veg - only_veg = 31 - 19 := by rw [h1, h2]
  exact h3

end NUMINAMATH_GPT_eat_both_veg_nonveg_l1341_134196
