import Mathlib

namespace NUMINAMATH_GPT_cobs_count_l1273_127349

theorem cobs_count (bushel_weight : ℝ) (ear_weight : ℝ) (num_bushels : ℕ)
  (h1 : bushel_weight = 56) (h2 : ear_weight = 0.5) (h3 : num_bushels = 2) : 
  ((num_bushels * bushel_weight) / ear_weight) = 224 :=
by 
  sorry

end NUMINAMATH_GPT_cobs_count_l1273_127349


namespace NUMINAMATH_GPT_inequality_ay_bz_cx_lt_k_squared_l1273_127364

theorem inequality_ay_bz_cx_lt_k_squared
  (a b c x y z k : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hk1 : a + x = k) (hk2 : b + y = k) (hk3 : c + z = k) :
  (a * y + b * z + c * x) < k^2 :=
sorry

end NUMINAMATH_GPT_inequality_ay_bz_cx_lt_k_squared_l1273_127364


namespace NUMINAMATH_GPT_height_of_stack_correct_l1273_127382

namespace PaperStack

-- Define the problem conditions
def sheets_per_package : ℕ := 500
def thickness_per_sheet_mm : ℝ := 0.1
def packages_per_stack : ℕ := 60
def mm_to_m : ℝ := 1000.0

-- Statement: the height of the stack of 60 paper packages
theorem height_of_stack_correct :
  (sheets_per_package * thickness_per_sheet_mm * packages_per_stack) / mm_to_m = 3 :=
sorry

end PaperStack

end NUMINAMATH_GPT_height_of_stack_correct_l1273_127382


namespace NUMINAMATH_GPT_cistern_empty_time_l1273_127390

noncomputable def time_to_empty_cistern (fill_no_leak_time fill_with_leak_time : ℝ) (filled_cistern : ℝ) : ℝ :=
  let R := filled_cistern / fill_no_leak_time
  let L := (R - filled_cistern / fill_with_leak_time)
  filled_cistern / L

theorem cistern_empty_time :
  time_to_empty_cistern 12 14 1 = 84 :=
by
  unfold time_to_empty_cistern
  simp
  sorry

end NUMINAMATH_GPT_cistern_empty_time_l1273_127390


namespace NUMINAMATH_GPT_monotonically_increasing_power_function_l1273_127347

theorem monotonically_increasing_power_function (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m ^ 2 - 2 * m - 2) * x ^ (m - 2) > 0 → (m ^ 2 - 2 * m - 2) > 0 ∧ (m - 2) > 0) ↔ m = 3 := 
sorry

end NUMINAMATH_GPT_monotonically_increasing_power_function_l1273_127347


namespace NUMINAMATH_GPT_trees_planted_tomorrow_l1273_127302

-- Definitions from the conditions
def current_trees := 39
def trees_planted_today := 41
def total_trees := 100

-- Theorem statement matching the proof problem
theorem trees_planted_tomorrow : 
  ∃ (trees_planted_tomorrow : ℕ), current_trees + trees_planted_today + trees_planted_tomorrow = total_trees ∧ trees_planted_tomorrow = 20 := 
by
  sorry

end NUMINAMATH_GPT_trees_planted_tomorrow_l1273_127302


namespace NUMINAMATH_GPT_remainder_when_divided_by_13_l1273_127313

theorem remainder_when_divided_by_13 (N k : ℤ) (h : N = 39 * k + 20) : N % 13 = 7 := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_13_l1273_127313


namespace NUMINAMATH_GPT_dvds_left_l1273_127334

-- Define the initial conditions
def owned_dvds : Nat := 13
def sold_dvds : Nat := 6

-- Define the goal
theorem dvds_left (owned_dvds : Nat) (sold_dvds : Nat) : owned_dvds - sold_dvds = 7 :=
by
  sorry

end NUMINAMATH_GPT_dvds_left_l1273_127334


namespace NUMINAMATH_GPT_smallest_positive_b_l1273_127332

theorem smallest_positive_b (b : ℕ) : 
  (b % 3 = 2) ∧ 
  (b % 4 = 3) ∧ 
  (b % 5 = 4) ∧ 
  (b % 6 = 5) ↔ 
  b = 59 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_b_l1273_127332


namespace NUMINAMATH_GPT_set_equality_l1273_127316

theorem set_equality (A : Set ℕ) (h : {1} ∪ A = {1, 3, 5}) : 
  A = {1, 3, 5} ∨ A = {3, 5} :=
  sorry

end NUMINAMATH_GPT_set_equality_l1273_127316


namespace NUMINAMATH_GPT_Lisa_quiz_goal_l1273_127370

theorem Lisa_quiz_goal (total_quizzes : ℕ) (required_percentage : ℝ) (a_scored : ℕ) (completed_quizzes : ℕ) : 
  total_quizzes = 60 → 
  required_percentage = 0.75 → 
  a_scored = 30 → 
  completed_quizzes = 40 → 
  ∃ lower_than_a_quizzes : ℕ, lower_than_a_quizzes = 5 :=
by
  intros total_quizzes_eq req_percent_eq a_scored_eq completed_quizzes_eq
  sorry

end NUMINAMATH_GPT_Lisa_quiz_goal_l1273_127370


namespace NUMINAMATH_GPT_find_angle_A_l1273_127319

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h1 : 2 * Real.sin B = Real.sqrt 3 * b) 
  (h2 : a = 2) (h3 : ∃ area : ℝ, area = Real.sqrt 3 ∧ area = (1 / 2) * b * c * Real.sin A) :
  A = Real.pi / 3 ∧ b = 2 ∧ c = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_A_l1273_127319


namespace NUMINAMATH_GPT_eighteen_mnp_eq_P_np_Q_2mp_l1273_127365

theorem eighteen_mnp_eq_P_np_Q_2mp (m n p : ℕ) (P Q : ℕ) (hP : P = 2 ^ m) (hQ : Q = 3 ^ n) :
  18 ^ (m * n * p) = P ^ (n * p) * Q ^ (2 * m * p) :=
by
  sorry

end NUMINAMATH_GPT_eighteen_mnp_eq_P_np_Q_2mp_l1273_127365


namespace NUMINAMATH_GPT_sum_of_terms_7_8_9_l1273_127338

namespace ArithmeticSequence

-- Define the sequence and its properties
variables (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * a 0 + n * (n - 1) / 2 * (a 1 - a 0)

def condition3 (S : ℕ → ℤ) : Prop :=
  S 3 = 9

def condition5 (S : ℕ → ℤ) : Prop :=
  S 5 = 30

-- Main statement to prove
theorem sum_of_terms_7_8_9 :
  is_arithmetic_sequence a →
  (∀ n, S n = sum_first_n_terms a n) →
  condition3 S →
  condition5 S →
  a 7 + a 8 + a 9 = 63 :=
by
  sorry

end ArithmeticSequence

end NUMINAMATH_GPT_sum_of_terms_7_8_9_l1273_127338


namespace NUMINAMATH_GPT_values_of_m_l1273_127350

theorem values_of_m (m n : ℕ) (hmn : m * n = 900) (hm: m > 1) (hn: n ≥ 1) : 
  (∃ (k : ℕ), ∀ (m : ℕ), (1 < m ∧ (900 / m) ≥ 1 ∧ 900 % m = 0) ↔ k = 25) :=
sorry

end NUMINAMATH_GPT_values_of_m_l1273_127350


namespace NUMINAMATH_GPT_isosceles_triangle_largest_angle_l1273_127372

theorem isosceles_triangle_largest_angle (a b c : ℝ) 
  (h1 : a = b)
  (h2 : c + 50 + 50 = 180) : 
  c = 80 :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_largest_angle_l1273_127372


namespace NUMINAMATH_GPT_even_function_a_value_monotonicity_on_neg_infinity_l1273_127305

noncomputable def f (x a : ℝ) : ℝ := ((x + 1) * (x + a)) / (x^2)

-- (1) Proving f(x) is even implies a = -1
theorem even_function_a_value (a : ℝ) : (∀ x : ℝ, f x a = f (-x) a) ↔ a = -1 :=
by
  sorry

-- (2) Proving monotonicity on (-∞, 0) for f(x) with a = -1
theorem monotonicity_on_neg_infinity (x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : x₂ < 0) :
  (f x₁ (-1) > f x₂ (-1)) :=
by
  sorry

end NUMINAMATH_GPT_even_function_a_value_monotonicity_on_neg_infinity_l1273_127305


namespace NUMINAMATH_GPT_root_expression_value_l1273_127343

-- Define the root condition
def is_root (a : ℝ) : Prop := 2 * a^2 - 3 * a - 5 = 0

-- The main theorem statement
theorem root_expression_value {a : ℝ} (h : is_root a) : -4 * a^2 + 6 * a = -10 := by
  sorry

end NUMINAMATH_GPT_root_expression_value_l1273_127343


namespace NUMINAMATH_GPT_find_certain_number_l1273_127303

theorem find_certain_number
  (t b c : ℝ)
  (average1 : (t + b + c + 14 + 15) / 5 = 12)
  (average2 : (t + b + c + x) / 4 = 15)
  (x : ℝ) :
  x = 29 :=
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l1273_127303


namespace NUMINAMATH_GPT_find_smallest_n_l1273_127396

-- Definitions of the condition that m and n are relatively prime and that the fraction includes the digits 4, 5, and 6 consecutively
def is_coprime (m n : ℕ) : Prop := Nat.gcd m n = 1

def has_digits_456 (m n : ℕ) : Prop := 
  ∃ k : ℕ, ∃ c : ℕ, 10^k * m % (10^k * n) = 456 * 10^c

-- The theorem to prove the smallest value of n
theorem find_smallest_n (m n : ℕ) (h1 : is_coprime m n) (h2 : m < n) (h3 : has_digits_456 m n) : n = 230 :=
sorry

end NUMINAMATH_GPT_find_smallest_n_l1273_127396


namespace NUMINAMATH_GPT_inscribed_polygon_sides_l1273_127329

-- We start by defining the conditions of the problem in Lean.
def radius := 1
def side_length_condition (n : ℕ) : Prop :=
  1 < 2 * Real.sin (Real.pi / n) ∧ 2 * Real.sin (Real.pi / n) < Real.sqrt 2

-- Now we state the main theorem.
theorem inscribed_polygon_sides (n : ℕ) (h1 : side_length_condition n) : n = 5 :=
  sorry

end NUMINAMATH_GPT_inscribed_polygon_sides_l1273_127329


namespace NUMINAMATH_GPT_chessboard_movement_l1273_127336

-- Defining the problem as described in the transformed proof problem

theorem chessboard_movement (pieces : Nat) (adjacent_empty_square : Nat → Nat → Bool) (visited_all_squares : Nat → Bool)
  (returns_to_starting_square : Nat → Bool) :
  (∃ (moment : Nat), ∀ (piece : Nat), ¬ returns_to_starting_square piece) :=
by
  -- Here we state that there exists a moment when each piece (checker) is not on its starting square
  sorry

end NUMINAMATH_GPT_chessboard_movement_l1273_127336


namespace NUMINAMATH_GPT_sum_of_abs_values_eq_12_l1273_127326

theorem sum_of_abs_values_eq_12 (a b c d : ℝ) (h : 6 * x^2 + x - 12 = (a * x + b) * (c * x + d)) :
  abs a + abs b + abs c + abs d = 12 := sorry

end NUMINAMATH_GPT_sum_of_abs_values_eq_12_l1273_127326


namespace NUMINAMATH_GPT_complement_of_M_in_U_l1273_127321

noncomputable def U : Set ℝ := { x | x^2 - 2 * x - 3 ≤ 0 }
noncomputable def M : Set ℝ := { y | ∃ x, x^2 + y^2 = 1 }

theorem complement_of_M_in_U :
  (U \ M) = { x | 1 < x ∧ x ≤ 3 } :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_in_U_l1273_127321


namespace NUMINAMATH_GPT_find_a_range_l1273_127333

-- Define propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- The main theorem stating the range of a
theorem find_a_range (a : ℝ) (h : ¬(∃ x : ℝ, p a x) → ¬(∃ x : ℝ, q x) ∧ ¬(¬(∃ x : ℝ, q x) → ¬(∃ x : ℝ, p a x))) : 1 < a ∧ a ≤ 2 := sorry

end NUMINAMATH_GPT_find_a_range_l1273_127333


namespace NUMINAMATH_GPT_find_num_managers_l1273_127301

variable (num_associates : ℕ) (avg_salary_managers avg_salary_associates avg_salary_company : ℚ)
variable (num_managers : ℚ)

-- Define conditions based on given problem
def conditions := 
  num_associates = 75 ∧
  avg_salary_managers = 90000 ∧
  avg_salary_associates = 30000 ∧
  avg_salary_company = 40000

-- Proof problem statement
theorem find_num_managers (h : conditions num_associates avg_salary_managers avg_salary_associates avg_salary_company) :
  num_managers = 15 :=
sorry

end NUMINAMATH_GPT_find_num_managers_l1273_127301


namespace NUMINAMATH_GPT_probability_correct_l1273_127318

-- Definitions and conditions
def G : List Char := ['A', 'B', 'C', 'D']

-- Number of favorable arrangements where A is adjacent to B and C
def favorable_arrangements : ℕ := 4  -- ABCD, BCDA, DABC, and CDAB

-- Total possible arrangements of 4 people
def total_arrangements : ℕ := 24  -- 4!

-- Probability calculation
def probability_A_adjacent_B_C : ℚ := favorable_arrangements / total_arrangements

-- Prove that this probability equals 1/6
theorem probability_correct : probability_A_adjacent_B_C = 1 / 6 := by
  sorry

end NUMINAMATH_GPT_probability_correct_l1273_127318


namespace NUMINAMATH_GPT_log_inequality_l1273_127353

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 2 / Real.log 5
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem log_inequality : c > a ∧ a > b := 
by
  sorry

end NUMINAMATH_GPT_log_inequality_l1273_127353


namespace NUMINAMATH_GPT_smallest_positive_integer_for_divisibility_l1273_127378

def is_divisible_by (a b : ℕ) : Prop :=
  ∃ k, a = b * k

def smallest_n (n : ℕ) : Prop :=
  (is_divisible_by (n^2) 50) ∧ (is_divisible_by (n^3) 288) ∧ (∀ m : ℕ, m > 0 → m < n → ¬ (is_divisible_by (m^2) 50 ∧ is_divisible_by (m^3) 288))

theorem smallest_positive_integer_for_divisibility : smallest_n 60 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_for_divisibility_l1273_127378


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l1273_127351

def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

theorem part1_solution (x : ℝ) : 
  f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2 := sorry

theorem part2_solution (a : ℝ) :
  (∃ x : ℝ, f x a < 2 * a) ↔ 3 < a := sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l1273_127351


namespace NUMINAMATH_GPT_original_speed_l1273_127395

noncomputable def circumference_feet := 10
noncomputable def feet_to_miles := 5280
noncomputable def seconds_to_hours := 3600
noncomputable def shortened_time := 1 / 18000
noncomputable def speed_increase := 6

theorem original_speed (r : ℝ) (t : ℝ) : 
  r * t = (circumference_feet / feet_to_miles) * seconds_to_hours ∧ 
  (r + speed_increase) * (t - shortened_time) = (circumference_feet / feet_to_miles) * seconds_to_hours
  → r = 6 := 
by
  sorry

end NUMINAMATH_GPT_original_speed_l1273_127395


namespace NUMINAMATH_GPT_factorize_2x2_minus_8_factorize_ax2_minus_2ax_plus_a_l1273_127330

variable {α : Type*} [CommRing α]

-- Problem 1
theorem factorize_2x2_minus_8 (x : α) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) :=
sorry

-- Problem 2
theorem factorize_ax2_minus_2ax_plus_a (a x : α) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
sorry

end NUMINAMATH_GPT_factorize_2x2_minus_8_factorize_ax2_minus_2ax_plus_a_l1273_127330


namespace NUMINAMATH_GPT_value_of_c_l1273_127314

noncomputable def f (x a b c : ℤ) : ℤ := x^3 + a*x^2 + b*x + c

theorem value_of_c (a b c : ℤ) (ha: a ≠ 0) (hb: b ≠ 0) (hc: c ≠ 0)
  (hfa: f a a b c = a^3) (hfb: f b a b c = b^3) : c = 16 := by
    sorry

end NUMINAMATH_GPT_value_of_c_l1273_127314


namespace NUMINAMATH_GPT_rank_A_second_l1273_127358

-- We define the conditions provided in the problem
variables (a b c : ℕ) -- defining the scores of A, B, and C as natural numbers

-- Conditions given
def A_said (a b c : ℕ) := b < a ∧ c < a
def B_said (b c : ℕ) := b > c
def C_said (a b c : ℕ) := a > c ∧ b > c

-- Conditions as hypotheses
variable (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c) -- the scores are different
variable (h2 : A_said a b c ∨ B_said b c ∨ C_said a b c) -- exactly one of the statements is incorrect

-- The theorem to prove
theorem rank_A_second : ∃ (rankA : ℕ), rankA = 2 := by
  sorry

end NUMINAMATH_GPT_rank_A_second_l1273_127358


namespace NUMINAMATH_GPT_people_with_fewer_than_7_cards_l1273_127388

theorem people_with_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (remainder : ℕ) 
  (h : total_cards = total_people * cards_per_person + remainder)
  (h_cards : total_cards = 60)
  (h_people : total_people = 9)
  (h_cards_per_person : cards_per_person = 6)
  (h_remainder : remainder = 6) :
  (total_people - remainder) = 3 :=
by
  sorry

end NUMINAMATH_GPT_people_with_fewer_than_7_cards_l1273_127388


namespace NUMINAMATH_GPT_centroid_midpoint_triangle_eq_centroid_original_triangle_l1273_127392

/-
Prove that the centroid of the triangle formed by the midpoints of the sides of another triangle
is the same as the centroid of the original triangle.
-/
theorem centroid_midpoint_triangle_eq_centroid_original_triangle
  (A B C M N P : ℝ × ℝ)
  (hM : M = (A + B) / 2)
  (hN : N = (A + C) / 2)
  (hP : P = (B + C) / 2) :
  (M.1 + N.1 + P.1) / 3 = (A.1 + B.1 + C.1) / 3 ∧
  (M.2 + N.2 + P.2) / 3 = (A.2 + B.2 + C.2) / 3 :=
by
  sorry

end NUMINAMATH_GPT_centroid_midpoint_triangle_eq_centroid_original_triangle_l1273_127392


namespace NUMINAMATH_GPT_eval_expression_l1273_127384

theorem eval_expression : 2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3 = 4019 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1273_127384


namespace NUMINAMATH_GPT_michelle_will_have_four_crayons_l1273_127356

def michelle_crayons (m j : ℕ) : ℕ := m + j

theorem michelle_will_have_four_crayons (H₁ : michelle_crayons 2 2 = 4) : michelle_crayons 2 2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_michelle_will_have_four_crayons_l1273_127356


namespace NUMINAMATH_GPT_trout_ratio_l1273_127359

theorem trout_ratio (caleb_trouts dad_trouts : ℕ) (h_c : caleb_trouts = 2) (h_d : dad_trouts = caleb_trouts + 4) :
  dad_trouts / (Nat.gcd dad_trouts caleb_trouts) = 3 ∧ caleb_trouts / (Nat.gcd dad_trouts caleb_trouts) = 1 :=
by
  sorry

end NUMINAMATH_GPT_trout_ratio_l1273_127359


namespace NUMINAMATH_GPT_find_numbers_l1273_127375

theorem find_numbers (x y : ℤ) (h_sum : x + y = 40) (h_diff : x - y = 12) : x = 26 ∧ y = 14 :=
sorry

end NUMINAMATH_GPT_find_numbers_l1273_127375


namespace NUMINAMATH_GPT_least_three_digit_product_18_l1273_127352

theorem least_three_digit_product_18 : ∃ N : ℕ, 100 ≤ N ∧ N ≤ 999 ∧ (∃ (H T U : ℕ), H ≠ 0 ∧ N = 100 * H + 10 * T + U ∧ H * T * U = 18) ∧ ∀ M : ℕ, (100 ≤ M ∧ M ≤ 999 ∧ (∃ (H T U : ℕ), H ≠ 0 ∧ M = 100 * H + 10 * T + U ∧ H * T * U = 18)) → N ≤ M :=
    sorry

end NUMINAMATH_GPT_least_three_digit_product_18_l1273_127352


namespace NUMINAMATH_GPT_determine_house_numbers_l1273_127306

-- Definitions based on the conditions given
def even_numbered_side (n : ℕ) : Prop :=
  n % 2 = 0

def sum_balanced (n : ℕ) (house_numbers : List ℕ) : Prop :=
  let left_sum := house_numbers.take n |>.sum
  let right_sum := house_numbers.drop (n + 1) |>.sum
  left_sum = right_sum

def house_constraints (n : ℕ) : Prop :=
  50 < n ∧ n < 500

-- Main theorem statement
theorem determine_house_numbers : 
  ∃ (n : ℕ) (house_numbers : List ℕ), 
    even_numbered_side n ∧ 
    house_constraints n ∧ 
    sum_balanced n house_numbers :=
  sorry

end NUMINAMATH_GPT_determine_house_numbers_l1273_127306


namespace NUMINAMATH_GPT_maria_waist_size_in_cm_l1273_127323

noncomputable def waist_size_in_cm (waist_size_inches : ℕ) (extra_inch : ℕ) (inches_per_foot : ℕ) (cm_per_foot : ℕ) : ℚ :=
  let total_inches := waist_size_inches + extra_inch
  let total_feet := (total_inches : ℚ) / inches_per_foot
  total_feet * cm_per_foot

theorem maria_waist_size_in_cm :
  waist_size_in_cm 28 1 12 31 = 74.9 :=
by
  sorry

end NUMINAMATH_GPT_maria_waist_size_in_cm_l1273_127323


namespace NUMINAMATH_GPT_probability_two_asian_countries_probability_A1_not_B1_l1273_127386

-- Scope: Definitions for the problem context
def countries : List String := ["A1", "A2", "A3", "B1", "B2", "B3"]

-- Probability of picking two Asian countries from a pool of six (three Asian, three European)
theorem probability_two_asian_countries : 
  (3 / 15) = (1 / 5) := by
  sorry

-- Probability of picking one country from the Asian group and 
-- one from the European group, including A1 but not B1
theorem probability_A1_not_B1 : 
  (2 / 9) = (2 / 9) := by
  sorry

end NUMINAMATH_GPT_probability_two_asian_countries_probability_A1_not_B1_l1273_127386


namespace NUMINAMATH_GPT_find_n_l1273_127342

theorem find_n :
  ∃ n : ℤ, 3 ^ 3 - 7 = 4 ^ 2 + 2 + n ∧ n = 2 :=
by
  use 2
  sorry

end NUMINAMATH_GPT_find_n_l1273_127342


namespace NUMINAMATH_GPT_nancy_total_spending_l1273_127324

theorem nancy_total_spending :
  let this_month_games := 9
  let this_month_price := 5
  let last_month_games := 8
  let last_month_price := 4
  let next_month_games := 7
  let next_month_price := 6
  let total_cost := (this_month_games * this_month_price) +
                    (last_month_games * last_month_price) +
                    (next_month_games * next_month_price)
  total_cost = 119 :=
by
  sorry

end NUMINAMATH_GPT_nancy_total_spending_l1273_127324


namespace NUMINAMATH_GPT_total_time_pushing_car_l1273_127325

theorem total_time_pushing_car :
  let d1 := 3
  let s1 := 6
  let d2 := 3
  let s2 := 3
  let d3 := 4
  let s3 := 8
  let t1 := d1 / s1
  let t2 := d2 / s2
  let t3 := d3 / s3
  (t1 + t2 + t3) = 2 :=
by
  sorry

end NUMINAMATH_GPT_total_time_pushing_car_l1273_127325


namespace NUMINAMATH_GPT_pirate_15_gets_coins_l1273_127348

def coins_required_for_pirates : ℕ :=
  Nat.factorial 14 * ((2 ^ 4) * (3 ^ 9)) / 15 ^ 14

theorem pirate_15_gets_coins :
  coins_required_for_pirates = 314928 := 
by sorry

end NUMINAMATH_GPT_pirate_15_gets_coins_l1273_127348


namespace NUMINAMATH_GPT_total_bill_is_89_l1273_127309

-- Define the individual costs and quantities
def adult_meal_cost := 12
def child_meal_cost := 7
def fries_cost := 5
def drink_cost := 10

def num_adults := 4
def num_children := 3
def num_fries := 2
def num_drinks := 1

-- Calculate the total bill
def total_bill : Nat :=
  (num_adults * adult_meal_cost) + 
  (num_children * child_meal_cost) + 
  (num_fries * fries_cost) + 
  (num_drinks * drink_cost)

-- The proof statement
theorem total_bill_is_89 : total_bill = 89 := 
  by
  -- The proof will be provided here
  sorry

end NUMINAMATH_GPT_total_bill_is_89_l1273_127309


namespace NUMINAMATH_GPT_sum_arithmetic_series_l1273_127398

theorem sum_arithmetic_series : 
    let a₁ := 1
    let d := 2
    let n := 9
    let a_n := a₁ + (n - 1) * d
    let S_n := n * (a₁ + a_n) / 2
    a_n = 17 → S_n = 81 :=
by intros
   sorry

end NUMINAMATH_GPT_sum_arithmetic_series_l1273_127398


namespace NUMINAMATH_GPT_boys_collected_in_all_l1273_127393

-- Definition of the problem’s conditions
variables (solomon juwan levi : ℕ)

-- Given conditions as assumptions
def conditions : Prop :=
  solomon = 66 ∧
  solomon = 3 * juwan ∧
  levi = juwan / 2

-- Total cans collected by all boys
def total_cans (solomon juwan levi : ℕ) : ℕ := solomon + juwan + levi

theorem boys_collected_in_all : ∃ solomon juwan levi : ℕ, 
  conditions solomon juwan levi ∧ total_cans solomon juwan levi = 99 :=
by {
  sorry
}

end NUMINAMATH_GPT_boys_collected_in_all_l1273_127393


namespace NUMINAMATH_GPT_seconds_hand_revolution_l1273_127355

theorem seconds_hand_revolution (revTimeSeconds revTimeMinutes : ℕ) : 
  (revTimeSeconds = 60) ∧ (revTimeMinutes = 1) :=
sorry

end NUMINAMATH_GPT_seconds_hand_revolution_l1273_127355


namespace NUMINAMATH_GPT_moles_of_C2H5Cl_l1273_127308

-- Define chemical entities as types
structure Molecule where
  name : String

-- Declare molecules involved in the reaction
def C2H6 := Molecule.mk "C2H6"
def Cl2  := Molecule.mk "Cl2"
def C2H5Cl := Molecule.mk "C2H5Cl"
def HCl := Molecule.mk "HCl"

-- Define number of moles as a non-negative integer
def moles (m : Molecule) : ℕ := sorry

-- Conditions
axiom initial_moles_C2H6 : moles C2H6 = 3
axiom initial_moles_Cl2 : moles Cl2 = 3

-- Balanced reaction equation: 1 mole of C2H6 reacts with 1 mole of Cl2 to form 1 mole of C2H5Cl
axiom reaction_stoichiometry : ∀ (x : ℕ), moles C2H6 = x → moles Cl2 = x → moles C2H5Cl = x

-- Proof problem
theorem moles_of_C2H5Cl : moles C2H5Cl = 3 := by
  apply reaction_stoichiometry
  exact initial_moles_C2H6
  exact initial_moles_Cl2

end NUMINAMATH_GPT_moles_of_C2H5Cl_l1273_127308


namespace NUMINAMATH_GPT_john_final_price_l1273_127337

theorem john_final_price : 
  let goodA_price := 2500
  let goodA_rebate := 0.06 * goodA_price
  let goodA_price_after_rebate := goodA_price - goodA_rebate
  let goodA_sales_tax := 0.10 * goodA_price_after_rebate
  let goodA_final_price := goodA_price_after_rebate + goodA_sales_tax
  
  let goodB_price := 3150
  let goodB_rebate := 0.08 * goodB_price
  let goodB_price_after_rebate := goodB_price - goodB_rebate
  let goodB_sales_tax := 0.12 * goodB_price_after_rebate
  let goodB_final_price := goodB_price_after_rebate + goodB_sales_tax

  let goodC_price := 1000
  let goodC_rebate := 0.05 * goodC_price
  let goodC_price_after_rebate := goodC_price - goodC_rebate
  let goodC_sales_tax := 0.07 * goodC_price_after_rebate
  let goodC_final_price := goodC_price_after_rebate + goodC_sales_tax

  let total_amount := goodA_final_price + goodB_final_price + goodC_final_price

  let special_voucher_discount := 0.03 * total_amount
  let final_price := total_amount - special_voucher_discount
  let rounded_final_price := Float.round final_price

  rounded_final_price = 6642 := by
  sorry

end NUMINAMATH_GPT_john_final_price_l1273_127337


namespace NUMINAMATH_GPT_compute_div_mul_l1273_127317

theorem compute_div_mul (x y z : Int) (h : y ≠ 0) (hx : x = -100) (hy : y = -25) (hz : z = -6) :
  (((-x) / (-y)) * -z) = -24 := by
  sorry

end NUMINAMATH_GPT_compute_div_mul_l1273_127317


namespace NUMINAMATH_GPT_animal_count_l1273_127304

variable (H C D : Nat)

theorem animal_count :
  (H + C + D = 72) → 
  (2 * H + 4 * C + 2 * D = 212) → 
  (C = 34) → 
  (H + D = 38) :=
by
  intros h1 h2 hc
  sorry

end NUMINAMATH_GPT_animal_count_l1273_127304


namespace NUMINAMATH_GPT_main_factor_is_D_l1273_127361

-- Let A, B, C, and D be the factors where A is influenced by 1, B by 2, C by 3, and D by 4
def A := 1
def B := 2
def C := 3
def D := 4

-- Defining the main factor influenced by the plan
def main_factor_influenced_by_plan := D

-- The problem statement translated to a Lean theorem statement
theorem main_factor_is_D : main_factor_influenced_by_plan = D := 
by sorry

end NUMINAMATH_GPT_main_factor_is_D_l1273_127361


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1273_127385

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f (1 + x) = f (1 - x)
axiom f_2 : f 2 = 1 / 2
axiom f_prime_lt_exp : ∀ x : ℝ, deriv f x < Real.exp x

theorem solution_set_of_inequality :
  {x : ℝ | f x < Real.exp x - 1 / 2} = {x : ℝ | 0 < x} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1273_127385


namespace NUMINAMATH_GPT_top_card_is_queen_probability_l1273_127369

-- Define the conditions of the problem
def standard_deck_size := 52
def number_of_queens := 4

-- Problem statement: The probability that the top card is a Queen
theorem top_card_is_queen_probability : 
  (number_of_queens : ℚ) / standard_deck_size = 1 / 13 := 
sorry

end NUMINAMATH_GPT_top_card_is_queen_probability_l1273_127369


namespace NUMINAMATH_GPT_bertha_descendants_no_children_l1273_127320

-- Definitions based on the conditions of the problem.
def bertha_daughters : ℕ := 10
def total_descendants : ℕ := 40
def granddaughters : ℕ := total_descendants - bertha_daughters
def daughters_with_children : ℕ := 8
def children_per_daughter_with_children : ℕ := 4
def number_of_granddaughters : ℕ := daughters_with_children * children_per_daughter_with_children
def total_daughters_and_granddaughters : ℕ := bertha_daughters + number_of_granddaughters
def without_children : ℕ := total_daughters_and_granddaughters - daughters_with_children

-- Lean statement to prove the main question given the definitions.
theorem bertha_descendants_no_children : without_children = 34 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_bertha_descendants_no_children_l1273_127320


namespace NUMINAMATH_GPT_sophia_lost_pawns_l1273_127335

theorem sophia_lost_pawns
    (total_pawns : ℕ := 16)
    (start_pawns_each : ℕ := 8)
    (chloe_lost : ℕ := 1)
    (pawns_left : ℕ := 10)
    (chloe_pawns_left : ℕ := start_pawns_each - chloe_lost) :
    total_pawns = 2 * start_pawns_each → 
    ∃ (sophia_lost : ℕ), sophia_lost = start_pawns_each - (pawns_left - chloe_pawns_left) :=
by 
    intros _ 
    use 5 
    sorry

end NUMINAMATH_GPT_sophia_lost_pawns_l1273_127335


namespace NUMINAMATH_GPT_jims_investment_l1273_127387

theorem jims_investment
  {total_investment : ℝ} 
  (h1 : total_investment = 127000)
  {john_ratio : ℕ} 
  (h2 : john_ratio = 8)
  {james_ratio : ℕ} 
  (h3 : james_ratio = 11)
  {jim_ratio : ℕ} 
  (h4 : jim_ratio = 15)
  {jordan_ratio : ℕ} 
  (h5 : jordan_ratio = 19) :
  jim_ratio / (john_ratio + james_ratio + jim_ratio + jordan_ratio) * total_investment = 35943.40 :=
by {
  sorry
}

end NUMINAMATH_GPT_jims_investment_l1273_127387


namespace NUMINAMATH_GPT_michael_meets_truck_once_l1273_127357

def michael_speed := 5  -- feet per second
def pail_distance := 150  -- feet
def truck_speed := 15  -- feet per second
def truck_stop_time := 20  -- seconds

def initial_michael_position (t : ℕ) : ℕ := t * michael_speed
def initial_truck_position (t : ℕ) : ℕ := pail_distance + t * truck_speed - (t / (truck_speed * truck_stop_time))

def distance (t : ℕ) : ℕ := initial_truck_position t - initial_michael_position t

theorem michael_meets_truck_once :
  ∃ t, (distance t = 0) :=  
sorry

end NUMINAMATH_GPT_michael_meets_truck_once_l1273_127357


namespace NUMINAMATH_GPT_principal_amount_l1273_127394

theorem principal_amount (A r t : ℝ) (hA : A = 1120) (hr : r = 0.11) (ht : t = 2.4) :
  abs ((A / (1 + r * t)) - 885.82) < 0.01 :=
by
  -- This theorem is stating that given A = 1120, r = 0.11, and t = 2.4,
  -- the principal amount (calculated using the simple interest formula)
  -- is approximately 885.82 with a margin of error less than 0.01.
  sorry

end NUMINAMATH_GPT_principal_amount_l1273_127394


namespace NUMINAMATH_GPT_vector_addition_l1273_127373

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (6, 2)
def vector_b : ℝ × ℝ := (-2, 4)

-- Theorem statement to prove the sum of vector_a and vector_b equals (4, 6)
theorem vector_addition :
  vector_a + vector_b = (4, 6) :=
sorry

end NUMINAMATH_GPT_vector_addition_l1273_127373


namespace NUMINAMATH_GPT_range_of_a_l1273_127346

variable (a : ℝ)

def p : Prop :=
  ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0

def q : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

theorem range_of_a :
  (p a ∧ q a) → a ≤ -1 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1273_127346


namespace NUMINAMATH_GPT_stewarts_theorem_l1273_127300

theorem stewarts_theorem 
  (a b b₁ a₁ d c : ℝ)
  (h₁ : b * b ≠ 0) 
  (h₂ : a * a ≠ 0) 
  (h₃ : b₁ * b₁ ≠ 0) 
  (h₄ : a₁ * a₁ ≠ 0) 
  (h₅ : d * d ≠ 0) 
  (h₆ : c = a₁ + b₁) :
  b * b * a₁ + a * a * b₁ - d * d * c = a₁ * b₁ * c :=
  sorry

end NUMINAMATH_GPT_stewarts_theorem_l1273_127300


namespace NUMINAMATH_GPT_calculate_x_l1273_127380

theorem calculate_x :
  529 + 2 * 23 * 11 + 121 = 1156 :=
by
  -- Begin the proof (which we won't complete here)
  -- The proof steps would go here
  sorry  -- placeholder for the actual proof steps

end NUMINAMATH_GPT_calculate_x_l1273_127380


namespace NUMINAMATH_GPT_first_movie_series_seasons_l1273_127360

theorem first_movie_series_seasons (S : ℕ) : 
  (∀ E : ℕ, E = 16) → 
  (∀ L : ℕ, L = 2) → 
  (∀ T : ℕ, T = 364) → 
  (∀ second_series_seasons : ℕ, second_series_seasons = 14) → 
  (∀ second_series_remaining : ℕ, second_series_remaining = second_series_seasons * (E - L)) → 
  (E - L = 14) → 
  (second_series_remaining = 196) → 
  (T - second_series_remaining = S * (E - L)) → 
  S = 12 :=
by 
  intros E_16 L_2 T_364 second_series_14 second_series_remaining_196 E_L second_series_total_episodes remaining_episodes
  sorry

end NUMINAMATH_GPT_first_movie_series_seasons_l1273_127360


namespace NUMINAMATH_GPT_no_distinct_natural_numbers_eq_sum_and_cubes_eq_l1273_127371

theorem no_distinct_natural_numbers_eq_sum_and_cubes_eq:
  ∀ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 
  → a^3 + b^3 = c^3 + d^3
  → a + b = c + d
  → false := 
by
  intros
  sorry

end NUMINAMATH_GPT_no_distinct_natural_numbers_eq_sum_and_cubes_eq_l1273_127371


namespace NUMINAMATH_GPT_school_club_profit_l1273_127312

def calculate_profit (bars_bought : ℕ) (cost_per_3_bars : ℚ) (bars_sold : ℕ) (price_per_4_bars : ℚ) : ℚ :=
  let cost_per_bar := cost_per_3_bars / 3
  let total_cost := bars_bought * cost_per_bar
  let price_per_bar := price_per_4_bars / 4
  let total_revenue := bars_sold * price_per_bar
  total_revenue - total_cost

theorem school_club_profit :
  calculate_profit 1200 1.50 1200 2.40 = 120 :=
by sorry

end NUMINAMATH_GPT_school_club_profit_l1273_127312


namespace NUMINAMATH_GPT_inverse_cos_plus_one_l1273_127311

noncomputable def f (x : ℝ) : ℝ := Real.cos x + 1

theorem inverse_cos_plus_one (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2) :
    f (-(Real.arccos (x - 1))) = x :=
by
  sorry

end NUMINAMATH_GPT_inverse_cos_plus_one_l1273_127311


namespace NUMINAMATH_GPT_decaf_percentage_total_l1273_127340

-- Defining the initial conditions
def initial_stock : ℝ := 400
def initial_decaf_percentage : ℝ := 0.30
def new_stock : ℝ := 100
def new_decaf_percentage : ℝ := 0.60

-- Given conditions
def amount_initial_decaf := initial_decaf_percentage * initial_stock
def amount_new_decaf := new_decaf_percentage * new_stock
def total_decaf := amount_initial_decaf + amount_new_decaf
def total_stock := initial_stock + new_stock

-- Prove the percentage of decaffeinated coffee in the total stock
theorem decaf_percentage_total : 
  (total_decaf / total_stock) * 100 = 36 := by
  sorry

end NUMINAMATH_GPT_decaf_percentage_total_l1273_127340


namespace NUMINAMATH_GPT_minimum_ribbon_length_l1273_127391

def side_length : ℚ := 13 / 12

def perimeter_of_equilateral_triangle (a : ℚ) : ℚ := 3 * a

theorem minimum_ribbon_length :
  perimeter_of_equilateral_triangle side_length = 3.25 := 
by
  sorry

end NUMINAMATH_GPT_minimum_ribbon_length_l1273_127391


namespace NUMINAMATH_GPT_contrapositive_equivalence_l1273_127379

variable (p q : Prop)

theorem contrapositive_equivalence : (p → ¬q) ↔ (q → ¬p) := by
  sorry

end NUMINAMATH_GPT_contrapositive_equivalence_l1273_127379


namespace NUMINAMATH_GPT_calculation_result_l1273_127397

theorem calculation_result : 
  (16 = 2^4) → 
  (8 = 2^3) → 
  (4 = 2^2) → 
  (16^6 * 8^3 / 4^10 = 8192) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_calculation_result_l1273_127397


namespace NUMINAMATH_GPT_probability_of_queen_is_correct_l1273_127310

def deck_size : ℕ := 52
def queen_count : ℕ := 4

-- This definition denotes the probability calculation.
def probability_drawing_queen : ℚ := queen_count / deck_size

theorem probability_of_queen_is_correct :
  probability_drawing_queen = 1 / 13 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_queen_is_correct_l1273_127310


namespace NUMINAMATH_GPT_joyce_pencils_given_l1273_127331

def original_pencils : ℕ := 51
def total_pencils_after : ℕ := 57

theorem joyce_pencils_given : total_pencils_after - original_pencils = 6 :=
by
  sorry

end NUMINAMATH_GPT_joyce_pencils_given_l1273_127331


namespace NUMINAMATH_GPT_two_roots_range_a_l1273_127345

noncomputable def piecewise_func (x : ℝ) : ℝ :=
if x ≤ 1 then (1/3) * x + 1 else Real.log x

theorem two_roots_range_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ piecewise_func x1 = a * x1 ∧ piecewise_func x2 = a * x2) ↔ (1/3 < a ∧ a < 1/Real.exp 1) :=
sorry

end NUMINAMATH_GPT_two_roots_range_a_l1273_127345


namespace NUMINAMATH_GPT_arithmetic_sequence_term_count_l1273_127363

theorem arithmetic_sequence_term_count (a d n an : ℕ) (h₀ : a = 5) (h₁ : d = 7) (h₂ : an = 126) (h₃ : an = a + (n - 1) * d) : n = 18 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_count_l1273_127363


namespace NUMINAMATH_GPT_sum_end_digit_7_l1273_127377

theorem sum_end_digit_7 (n : ℕ) : ¬ (n * (n + 1) ≡ 14 [MOD 20]) :=
by
  intro h
  -- Place where you'd continue the proof, but for now we use sorry
  sorry

end NUMINAMATH_GPT_sum_end_digit_7_l1273_127377


namespace NUMINAMATH_GPT_original_number_of_men_l1273_127376

theorem original_number_of_men (W : ℝ) (M : ℝ) (total_work : ℝ) :
  (M * W * 11 = (M + 10) * W * 8) → M = 27 :=
by
  sorry

end NUMINAMATH_GPT_original_number_of_men_l1273_127376


namespace NUMINAMATH_GPT_corn_height_growth_l1273_127344

theorem corn_height_growth (init_height week1_growth week2_growth week3_growth : ℕ)
  (h0 : init_height = 0)
  (h1 : week1_growth = 2)
  (h2 : week2_growth = 2 * week1_growth)
  (h3 : week3_growth = 4 * week2_growth) :
  init_height + week1_growth + week2_growth + week3_growth = 22 :=
by sorry

end NUMINAMATH_GPT_corn_height_growth_l1273_127344


namespace NUMINAMATH_GPT_percentage_of_circle_outside_triangle_l1273_127322

theorem percentage_of_circle_outside_triangle (A : ℝ)
  (h₁ : 0 < A) -- Total area A is positive
  (A_inter : ℝ) (A_outside_tri : ℝ) (A_total_circle : ℝ)
  (h₂ : A_inter = 0.45 * A)
  (h₃ : A_outside_tri = 0.40 * A)
  (h₄ : A_total_circle = 0.60 * A) :
  100 * (1 - A_inter / A_total_circle) = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_circle_outside_triangle_l1273_127322


namespace NUMINAMATH_GPT_find_solutions_l1273_127367

noncomputable def equation (x : ℝ) : ℝ :=
  (1 / (x^2 + 11*x - 8)) + (1 / (x^2 + 2*x - 8)) + (1 / (x^2 - 13*x - 8))

theorem find_solutions : 
  {x : ℝ | equation x = 0} = {1, -8, 8, -1} := by
  sorry

end NUMINAMATH_GPT_find_solutions_l1273_127367


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_with_product_272_l1273_127339

theorem sum_of_consecutive_integers_with_product_272 :
    ∃ (x y : ℕ), x * y = 272 ∧ y = x + 1 ∧ x + y = 33 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_with_product_272_l1273_127339


namespace NUMINAMATH_GPT_divide_segment_mean_proportional_l1273_127368

theorem divide_segment_mean_proportional (a : ℝ) (x : ℝ) : 
  ∃ H : ℝ, H > 0 ∧ H < a ∧ H = (a * (Real.sqrt 5 - 1) / 2) :=
sorry

end NUMINAMATH_GPT_divide_segment_mean_proportional_l1273_127368


namespace NUMINAMATH_GPT_number_of_registration_methods_l1273_127366

theorem number_of_registration_methods
  (students : ℕ) (groups : ℕ) (registration_methods : ℕ)
  (h_students : students = 4) (h_groups : groups = 3) :
  registration_methods = groups ^ students :=
by
  rw [h_students, h_groups]
  exact sorry

end NUMINAMATH_GPT_number_of_registration_methods_l1273_127366


namespace NUMINAMATH_GPT_right_angled_triangle_other_angle_isosceles_triangle_base_angle_l1273_127362

theorem right_angled_triangle_other_angle (a : ℝ) (h1 : 0 < a) (h2 : a < 90) (h3 : 40 = a) :
  50 = 90 - a :=
sorry

theorem isosceles_triangle_base_angle (v : ℝ) (h1 : 0 < v) (h2 : v < 180) (h3 : 80 = v) :
  50 = (180 - v) / 2 :=
sorry

end NUMINAMATH_GPT_right_angled_triangle_other_angle_isosceles_triangle_base_angle_l1273_127362


namespace NUMINAMATH_GPT_standard_equation_of_circle_l1273_127381

theorem standard_equation_of_circle :
  ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ (h - 2) / 2 = k / 1 + 3 / 2 ∧ 
  ((h - 2)^2 + (k + 3)^2 = r^2) ∧ ((h + 2)^2 + (k + 5)^2 = r^2) ∧ 
  h = -1 ∧ k = -2 ∧ r^2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_standard_equation_of_circle_l1273_127381


namespace NUMINAMATH_GPT_average_price_of_towels_l1273_127341

-- Definitions based on the conditions
def cost_of_three_towels := 3 * 100
def cost_of_five_towels := 5 * 150
def cost_of_two_towels := 550
def total_cost := cost_of_three_towels + cost_of_five_towels + cost_of_two_towels
def total_number_of_towels := 3 + 5 + 2
def average_price := total_cost / total_number_of_towels

-- The theorem statement
theorem average_price_of_towels :
  average_price = 160 :=
by
  sorry

end NUMINAMATH_GPT_average_price_of_towels_l1273_127341


namespace NUMINAMATH_GPT_lisa_flight_time_l1273_127383

theorem lisa_flight_time :
  ∀ (d s : ℕ), (d = 256) → (s = 32) → ((d / s) = 8) :=
by
  intros d s h_d h_s
  sorry

end NUMINAMATH_GPT_lisa_flight_time_l1273_127383


namespace NUMINAMATH_GPT_crayons_loss_difference_l1273_127307

theorem crayons_loss_difference (crayons_given crayons_lost : ℕ) 
  (h_given : crayons_given = 90) 
  (h_lost : crayons_lost = 412) : 
  crayons_lost - crayons_given = 322 :=
by
  sorry

end NUMINAMATH_GPT_crayons_loss_difference_l1273_127307


namespace NUMINAMATH_GPT_seventh_graders_problems_l1273_127399

theorem seventh_graders_problems (n : ℕ) (S : ℕ) (a : ℕ) (h1 : a > (S - a) / 5) (h2 : a < (S - a) / 3) : n = 5 :=
  sorry

end NUMINAMATH_GPT_seventh_graders_problems_l1273_127399


namespace NUMINAMATH_GPT_chess_group_players_l1273_127315

theorem chess_group_players (n : ℕ) (H : n * (n - 1) / 2 = 435) : n = 30 :=
by
  sorry

end NUMINAMATH_GPT_chess_group_players_l1273_127315


namespace NUMINAMATH_GPT_smallest_positive_value_l1273_127389

theorem smallest_positive_value (a b : ℤ) (h : a > b) : 
  ∃ (k : ℝ), k = 2 ∧ k = (↑(a - b) / ↑(a + b) + ↑(a + b) / ↑(a - b)) :=
sorry

end NUMINAMATH_GPT_smallest_positive_value_l1273_127389


namespace NUMINAMATH_GPT_union_sets_l1273_127327

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem union_sets : M ∪ N = {-1, 0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_union_sets_l1273_127327


namespace NUMINAMATH_GPT_intersection_of_M_and_N_is_12_l1273_127374

def M : Set ℤ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℤ := {1, 2, 3}

theorem intersection_of_M_and_N_is_12 : M ∩ N = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_is_12_l1273_127374


namespace NUMINAMATH_GPT_no_real_b_for_inequality_l1273_127354

theorem no_real_b_for_inequality : ¬ ∃ b : ℝ, (∃ x : ℝ, |x^2 + 3 * b * x + 4 * b| = 5 ∧ ∀ y : ℝ, y ≠ x → |y^2 + 3 * b * y + 4 * b| > 5) := sorry

end NUMINAMATH_GPT_no_real_b_for_inequality_l1273_127354


namespace NUMINAMATH_GPT_max_distance_m_l1273_127328

def circle_eq (x y : ℝ) := x^2 + y^2 - 4*x + 6*y - 3 = 0
def line_eq (m x y : ℝ) := m * x + y + m - 1 = 0
def center_circle (x y : ℝ) := circle_eq x y → (x = 2) ∧ (y = -3)

theorem max_distance_m :
  ∃ m : ℝ, line_eq m (-1) 1 ∧ ∀ x y t u : ℝ, center_circle x y → line_eq m t u → 
  -(4 / 3) * -m = -1 → m = -(3 / 4) :=
sorry

end NUMINAMATH_GPT_max_distance_m_l1273_127328
