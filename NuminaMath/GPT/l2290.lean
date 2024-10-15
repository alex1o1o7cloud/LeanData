import Mathlib

namespace NUMINAMATH_GPT_stratified_sampling_example_l2290_229085

theorem stratified_sampling_example 
    (high_school_students : ℕ)
    (junior_high_students : ℕ) 
    (sampled_high_school_students : ℕ)
    (sampling_ratio : ℚ)
    (total_students : ℕ)
    (n : ℕ)
    (h1 : high_school_students = 3500)
    (h2 : junior_high_students = 1500)
    (h3 : sampled_high_school_students = 70)
    (h4 : sampling_ratio = sampled_high_school_students / high_school_students)
    (h5 : total_students = high_school_students + junior_high_students) :
    n = total_students * sampling_ratio → 
    n = 100 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_example_l2290_229085


namespace NUMINAMATH_GPT_sum_of_next_17_consecutive_integers_l2290_229098

theorem sum_of_next_17_consecutive_integers (x : ℤ) (h₁ : (List.range 17).sum + 17 * x = 306) :
  (List.range 17).sum + 17 * (x + 17)  = 595 := 
sorry

end NUMINAMATH_GPT_sum_of_next_17_consecutive_integers_l2290_229098


namespace NUMINAMATH_GPT_palindrome_probability_divisible_by_11_l2290_229099

namespace PalindromeProbability

-- Define the concept of a five-digit palindrome and valid digits
def is_five_digit_palindrome (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ n = 10001 * a + 1010 * b + 100 * c

-- Define the condition for a number being divisible by 11
def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- Count all five-digit palindromes
def count_five_digit_palindromes : ℕ :=
  9 * 10 * 10  -- There are 9 choices for a (1-9), and 10 choices for b and c (0-9)

-- Count five-digit palindromes that are divisible by 11
def count_divisible_by_11_five_digit_palindromes : ℕ :=
  9 * 10  -- There are 9 choices for a, and 10 valid (b, c) pairs for divisibility by 11

-- Calculate the probability
theorem palindrome_probability_divisible_by_11 :
  (count_divisible_by_11_five_digit_palindromes : ℚ) / count_five_digit_palindromes = 1 / 10 :=
  by sorry -- Proof goes here

end PalindromeProbability

end NUMINAMATH_GPT_palindrome_probability_divisible_by_11_l2290_229099


namespace NUMINAMATH_GPT_four_p_plus_one_composite_l2290_229050

theorem four_p_plus_one_composite (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_five : p ≥ 5) (h2p_plus1_prime : Nat.Prime (2 * p + 1)) : ¬ Nat.Prime (4 * p + 1) :=
sorry

end NUMINAMATH_GPT_four_p_plus_one_composite_l2290_229050


namespace NUMINAMATH_GPT_train_cross_time_l2290_229057

open Real

noncomputable def length_train1 := 190 -- in meters
noncomputable def length_train2 := 160 -- in meters
noncomputable def speed_train1 := 60 * (5/18) --speed_kmhr_to_msec 60 km/hr to m/s
noncomputable def speed_train2 := 40 * (5/18) -- speed_kmhr_to_msec 40 km/hr to m/s
noncomputable def relative_speed := speed_train1 + speed_train2 -- relative speed

theorem train_cross_time :
  (length_train1 + length_train2) / relative_speed = 350 / ((60 * (5/18)) + (40 * (5/18))) :=
by
  sorry -- The proof will be here initially just to validate the Lean statement

end NUMINAMATH_GPT_train_cross_time_l2290_229057


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l2290_229095

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
def condition_1 (n : ℕ) : Prop := S n = 2 * a n - 2

theorem geometric_sequence_ratio (h : ∀ n, condition_1 a S n) : (a 8 / a 6 = 4) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l2290_229095


namespace NUMINAMATH_GPT_scenario_a_scenario_b_l2290_229029

-- Define the chessboard and the removal function
def is_adjacent (x1 y1 x2 y2 : ℕ) : Prop :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y1 + 1 = y2)) ∨ (y1 = y2 ∧ (x1 = x2 + 1 ∨ x1 + 1 = x2))

def is_square (x y : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 8 ∧ 1 ≤ y ∧ y ≤ 8

-- Define a Hamiltonian path on the chessboard
inductive HamiltonianPath : (ℕ × ℕ) → (ℕ → (ℕ × ℕ)) → ℕ → Prop
| empty : Π (start : ℕ × ℕ) (path : ℕ → (ℕ × ℕ)), HamiltonianPath start path 0
| step : Π (start : ℕ × ℕ) (path : ℕ → (ℕ × ℕ)) (n : ℕ),
    is_adjacent (path n).1 (path n).2 (path (n+1)).1 (path (n+1)).2 →
    HamiltonianPath start path n →
    (is_square (path (n + 1)).1 (path (n + 1)).2 ∧ ¬ (∃ m < n + 1, path m = path (n + 1))) →
    HamiltonianPath start path (n + 1)

-- State the main theorems
theorem scenario_a : 
  ∃ (path : ℕ → (ℕ × ℕ)),
    HamiltonianPath (3, 2) path 62 ∧
    (∀ n, path n ≠ (2, 2)) := sorry

theorem scenario_b :
  ¬ ∃ (path : ℕ → (ℕ × ℕ)),
    HamiltonianPath (3, 2) path 61 ∧
    (∀ n, path n ≠ (2, 2) ∧ path n ≠ (7, 7)) := sorry

end NUMINAMATH_GPT_scenario_a_scenario_b_l2290_229029


namespace NUMINAMATH_GPT_ratio_of_blue_marbles_l2290_229077

theorem ratio_of_blue_marbles {total_marbles red_marbles orange_marbles blue_marbles : ℕ} 
  (h_total : total_marbles = 24)
  (h_red : red_marbles = 6)
  (h_orange : orange_marbles = 6)
  (h_blue : blue_marbles = total_marbles - red_marbles - orange_marbles) : 
  (blue_marbles : ℚ) / (total_marbles : ℚ) = 1 / 2 := 
by
  sorry -- the proof is omitted as per instructions

end NUMINAMATH_GPT_ratio_of_blue_marbles_l2290_229077


namespace NUMINAMATH_GPT_hadassah_additional_paintings_l2290_229051

noncomputable def hadassah_initial_paintings : ℕ := 12
noncomputable def hadassah_initial_hours : ℕ := 6
noncomputable def hadassah_total_hours : ℕ := 16

theorem hadassah_additional_paintings 
  (initial_paintings : ℕ)
  (initial_hours : ℕ)
  (total_hours : ℕ) :
  initial_paintings = hadassah_initial_paintings →
  initial_hours = hadassah_initial_hours →
  total_hours = hadassah_total_hours →
  let additional_hours := total_hours - initial_hours
  let painting_rate := initial_paintings / initial_hours
  let additional_paintings := painting_rate * additional_hours
  additional_paintings = 20 :=
by
  sorry

end NUMINAMATH_GPT_hadassah_additional_paintings_l2290_229051


namespace NUMINAMATH_GPT_amount_after_2_years_l2290_229048

noncomputable def amount_after_n_years (present_value : ℝ) (rate_of_increase : ℝ) (years : ℕ) : ℝ :=
  present_value * (1 + rate_of_increase)^years

theorem amount_after_2_years :
  amount_after_n_years 6400 (1/8) 2 = 8100 :=
by
  sorry

end NUMINAMATH_GPT_amount_after_2_years_l2290_229048


namespace NUMINAMATH_GPT_common_number_is_eight_l2290_229032

theorem common_number_is_eight (a b c d e f g : ℝ) 
  (h1 : (a + b + c + d) / 4 = 7)
  (h2 : (d + e + f + g) / 4 = 9)
  (h3 : (a + b + c + d + e + f + g) / 7 = 8) :
  d = 8 :=
by
sorry

end NUMINAMATH_GPT_common_number_is_eight_l2290_229032


namespace NUMINAMATH_GPT_number_of_stacks_l2290_229039

theorem number_of_stacks (total_coins stacks coins_per_stack : ℕ) (h1 : coins_per_stack = 3) (h2 : total_coins = 15) (h3 : total_coins = stacks * coins_per_stack) : stacks = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_stacks_l2290_229039


namespace NUMINAMATH_GPT_ellipse_foci_coordinates_l2290_229066

/-- Define the parameters for the ellipse. -/
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 169 = 1

/-- Prove the coordinates of the foci of the given ellipse. -/
theorem ellipse_foci_coordinates :
  (∀ (x y : ℝ), ellipse_eq x y → False) →
  ∃ (c : ℝ), c = 12 ∧ 
  ((0, c) = (0, 12) ∧ (0, -c) = (0, -12)) := 
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_coordinates_l2290_229066


namespace NUMINAMATH_GPT_shirt_cost_l2290_229083

theorem shirt_cost (J S : ℝ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 86) : S = 24 :=
by
  sorry

end NUMINAMATH_GPT_shirt_cost_l2290_229083


namespace NUMINAMATH_GPT_sunday_price_correct_l2290_229071

def original_price : ℝ := 250
def first_discount_rate : ℝ := 0.60
def second_discount_rate : ℝ := 0.25
def discounted_price : ℝ := original_price * (1 - first_discount_rate)
def sunday_price : ℝ := discounted_price * (1 - second_discount_rate)

theorem sunday_price_correct :
  sunday_price = 75 := by
  sorry

end NUMINAMATH_GPT_sunday_price_correct_l2290_229071


namespace NUMINAMATH_GPT_vincent_total_laundry_loads_l2290_229014

theorem vincent_total_laundry_loads :
  let wednesday_loads := 6
  let thursday_loads := 2 * wednesday_loads
  let friday_loads := thursday_loads / 2
  let saturday_loads := wednesday_loads / 3
  let total_loads := wednesday_loads + thursday_loads + friday_loads + saturday_loads
  total_loads = 26 :=
by {
  let wednesday_loads := 6
  let thursday_loads := 2 * wednesday_loads
  let friday_loads := thursday_loads / 2
  let saturday_loads := wednesday_loads / 3
  let total_loads := wednesday_loads + thursday_loads + friday_loads + saturday_loads
  show total_loads = 26
  sorry
}

end NUMINAMATH_GPT_vincent_total_laundry_loads_l2290_229014


namespace NUMINAMATH_GPT_smallest_positive_integer_satisfying_conditions_l2290_229068

theorem smallest_positive_integer_satisfying_conditions :
  ∃ (x : ℕ),
    x % 4 = 1 ∧
    x % 5 = 2 ∧
    x % 7 = 3 ∧
    ∀ y : ℕ, (y % 4 = 1 ∧ y % 5 = 2 ∧ y % 7 = 3) → y ≥ x ∧ x = 93 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_satisfying_conditions_l2290_229068


namespace NUMINAMATH_GPT_find_c_d_l2290_229011

theorem find_c_d (c d : ℝ) (h1 : c ≠ 0) (h2 : d ≠ 0)
  (h3 : ∀ x : ℝ, x^2 + c * x + d = 0 → (x = c ∧ x = d)) :
  c = 1 ∧ d = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_c_d_l2290_229011


namespace NUMINAMATH_GPT_question_1_question_2_question_3_l2290_229035

theorem question_1 (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + (m - 1) < 1) ↔ 
    m < (1 - 2 * Real.sqrt 7) / 3 := sorry

theorem question_2 (m : ℝ) : 
  ∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + (m - 1) ≥ (m + 1) * x := sorry

theorem question_3 (m : ℝ) : 
  (∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2), (m + 1) * x^2 - (m - 1) * x + (m - 1) ≥ 0) ↔ 
    m ≥ 1 := sorry

end NUMINAMATH_GPT_question_1_question_2_question_3_l2290_229035


namespace NUMINAMATH_GPT_distinct_sum_l2290_229054

theorem distinct_sum (a b c d e : ℤ) (h1 : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 0)
  (h2 : a ≠ b) (h3 : a ≠ c) (h4 : a ≠ d) (h5 : a ≠ e) (h6 : b ≠ c) (h7 : b ≠ d) (h8 : b ≠ e) (h9 : c ≠ d) (h10 : c ≠ e) (h11 : d ≠ e) :
  a + b + c + d + e = 35 :=
sorry

end NUMINAMATH_GPT_distinct_sum_l2290_229054


namespace NUMINAMATH_GPT_quadratic_non_real_roots_l2290_229052

variable (b : ℝ)

theorem quadratic_non_real_roots : (b^2 - 64 < 0) → (-8 < b ∧ b < 8) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_non_real_roots_l2290_229052


namespace NUMINAMATH_GPT_total_candles_in_small_boxes_l2290_229070

-- Definitions of the conditions
def num_small_boxes_per_big_box := 4
def num_big_boxes := 50
def candles_per_small_box := 40

-- The total number of small boxes
def total_small_boxes : Nat := num_small_boxes_per_big_box * num_big_boxes

-- The statement to prove the total number of candles in all small boxes is 8000
theorem total_candles_in_small_boxes : candles_per_small_box * total_small_boxes = 8000 :=
by 
  sorry

end NUMINAMATH_GPT_total_candles_in_small_boxes_l2290_229070


namespace NUMINAMATH_GPT_find_k_l2290_229091

theorem find_k (k t : ℝ) (h1 : t = 5) (h2 : (1/2) * (t^2) / ((k-1) * (k+1)) = 10) : 
  k = 3/2 := 
  sorry

end NUMINAMATH_GPT_find_k_l2290_229091


namespace NUMINAMATH_GPT_bead_problem_l2290_229086

theorem bead_problem 
  (x y : ℕ) 
  (hx : 19 * x + 17 * y = 2017): 
  (x + y = 107) ∨ (x + y = 109) ∨ (x + y = 111) ∨ (x + y = 113) ∨ (x + y = 115) ∨ (x + y = 117) := 
sorry

end NUMINAMATH_GPT_bead_problem_l2290_229086


namespace NUMINAMATH_GPT_find_a_even_function_l2290_229080

theorem find_a_even_function (f : ℝ → ℝ) (a : ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_domain : ∀ x, 2 * a + 1 ≤ x ∧ x ≤ a + 5) :
  a = -2 :=
sorry

end NUMINAMATH_GPT_find_a_even_function_l2290_229080


namespace NUMINAMATH_GPT_range_of_x_if_p_and_q_range_of_a_if_not_p_sufficient_for_not_q_l2290_229093

variable (x a : ℝ)

-- Condition p
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a > 0

-- Condition q
def q (x : ℝ) : Prop :=
  (x^2 - x - 6 ≤ 0) ∧ (x^2 + 3*x - 10 > 0)

-- Proof problem for question (1)
theorem range_of_x_if_p_and_q (h1 : p x 1) (h2 : q x) : 2 < x ∧ x < 3 :=
  sorry

-- Proof problem for question (2)
theorem range_of_a_if_not_p_sufficient_for_not_q (h : (¬p x a) → (¬q x)) : 1 < a ∧ a ≤ 2 :=
  sorry

end NUMINAMATH_GPT_range_of_x_if_p_and_q_range_of_a_if_not_p_sufficient_for_not_q_l2290_229093


namespace NUMINAMATH_GPT_y_increase_by_20_l2290_229002

-- Define the conditions
def relationship (Δx Δy : ℕ) : Prop :=
  Δy = (11 * Δx) / 5

-- The proof problem statement
theorem y_increase_by_20 : relationship 5 11 → relationship 20 44 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_y_increase_by_20_l2290_229002


namespace NUMINAMATH_GPT_jane_exercises_per_day_l2290_229021

-- Conditions
variables (total_hours : ℕ) (total_weeks : ℕ) (days_per_week : ℕ)
variable (goal_achieved : total_hours = 40 ∧ total_weeks = 8 ∧ days_per_week = 5)

-- Statement
theorem jane_exercises_per_day : ∃ hours_per_day : ℕ, hours_per_day = (total_hours / total_weeks) / days_per_week :=
by
  sorry

end NUMINAMATH_GPT_jane_exercises_per_day_l2290_229021


namespace NUMINAMATH_GPT_fibonacci_periodicity_l2290_229017

-- Definitions for p-arithmetic and Fibonacci sequence
def is_prime (p : ℕ) := Nat.Prime p
def sqrt_5_extractable (p : ℕ) : Prop := ∃ k : ℕ, p = 5 * k + 1 ∨ p = 5 * k - 1

-- Definitions of Fibonacci sequences and properties
def fibonacci : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fibonacci n + fibonacci (n + 1)

-- Main theorem
theorem fibonacci_periodicity (p : ℕ) (r : ℕ) (h_prime : is_prime p) (h_not_2_or_5 : p ≠ 2 ∧ p ≠ 5)
    (h_period : r = (p+1) ∨ r = (p-1)) (h_div : (sqrt_5_extractable p → r ∣ (p - 1)) ∧ (¬ sqrt_5_extractable p → r ∣ (p + 1)))
    : (fibonacci (p+1) % p = 0 ∨ fibonacci (p-1) % p = 0) := by
          sorry

end NUMINAMATH_GPT_fibonacci_periodicity_l2290_229017


namespace NUMINAMATH_GPT_add_ab_values_l2290_229040

theorem add_ab_values (a b : ℝ) (h1 : ∀ x : ℝ, (x^2 + 4*x + 3) = (a*x + b)^2 + 4*(a*x + b) + 3) :
  a + b = -8 ∨ a + b = 4 :=
  by sorry

end NUMINAMATH_GPT_add_ab_values_l2290_229040


namespace NUMINAMATH_GPT_olivine_more_stones_l2290_229087

theorem olivine_more_stones (x O D : ℕ) (h1 : O = 30 + x) (h2 : D = O + 11)
  (h3 : 30 + O + D = 111) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_olivine_more_stones_l2290_229087


namespace NUMINAMATH_GPT_product_of_two_numbers_l2290_229089

theorem product_of_two_numbers (a b : ℝ)
  (h1 : a + b = 8 * (a - b))
  (h2 : a * b = 30 * (a - b)) :
  a * b = 400 / 7 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l2290_229089


namespace NUMINAMATH_GPT_factorize_expression_l2290_229025

theorem factorize_expression (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l2290_229025


namespace NUMINAMATH_GPT_negation_of_existential_statement_l2290_229015

theorem negation_of_existential_statement (x : ℚ) :
  ¬ (∃ x : ℚ, x^2 = 3) ↔ ∀ x : ℚ, x^2 ≠ 3 :=
by sorry

end NUMINAMATH_GPT_negation_of_existential_statement_l2290_229015


namespace NUMINAMATH_GPT_Tino_has_correct_jellybeans_total_jellybeans_l2290_229038

-- Define the individuals and their amounts of jellybeans
def Arnold_jellybeans := 5
def Lee_jellybeans := 2 * Arnold_jellybeans
def Tino_jellybeans := Lee_jellybeans + 24
def Joshua_jellybeans := 3 * Arnold_jellybeans

-- Verify Tino's jellybean count
theorem Tino_has_correct_jellybeans : Tino_jellybeans = 34 :=
by
  -- Unfold definitions and perform calculations
  sorry

-- Verify the total jellybean count
theorem total_jellybeans : (Arnold_jellybeans + Lee_jellybeans + Tino_jellybeans + Joshua_jellybeans) = 64 :=
by
  -- Unfold definitions and perform calculations
  sorry

end NUMINAMATH_GPT_Tino_has_correct_jellybeans_total_jellybeans_l2290_229038


namespace NUMINAMATH_GPT_number_of_rallies_l2290_229076

open Nat

def X_rallies : Nat := 10
def O_rallies : Nat := 100
def sequence_Os : Nat := 3
def sequence_Xs : Nat := 7

theorem number_of_rallies : 
  (sequence_Os * O_rallies + sequence_Xs * X_rallies ≤ 379) ∧ 
  (sequence_Os * O_rallies + sequence_Xs * X_rallies ≥ 370) := 
by
  sorry

end NUMINAMATH_GPT_number_of_rallies_l2290_229076


namespace NUMINAMATH_GPT_platform_length_l2290_229033

theorem platform_length (train_length : ℕ) (pole_time : ℕ) (platform_time : ℕ) (V : ℕ) (L : ℕ)
  (h_train_length : train_length = 500)
  (h_pole_time : pole_time = 50)
  (h_platform_time : platform_time = 100)
  (h_speed : V = train_length / pole_time)
  (h_platform_distance : V * platform_time = train_length + L) : 
  L = 500 := 
sorry

end NUMINAMATH_GPT_platform_length_l2290_229033


namespace NUMINAMATH_GPT_abs_diff_of_pq_eq_6_and_pq_sum_7_l2290_229047

variable (p q : ℝ)

noncomputable def abs_diff (a b : ℝ) := |a - b|

theorem abs_diff_of_pq_eq_6_and_pq_sum_7 (hpq : p * q = 6) (hpq_sum : p + q = 7) : abs_diff p q = 5 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_of_pq_eq_6_and_pq_sum_7_l2290_229047


namespace NUMINAMATH_GPT_total_distance_walked_l2290_229060

noncomputable def hazel_total_distance : ℕ := 3

def distance_first_hour := 2  -- The distance traveled in the first hour (in kilometers)
def distance_second_hour := distance_first_hour * 2  -- The distance traveled in the second hour
def distance_third_hour := distance_second_hour / 2  -- The distance traveled in the third hour, with a 50% speed decrease

theorem total_distance_walked :
  distance_first_hour + distance_second_hour + distance_third_hour = 8 :=
  by
    sorry

end NUMINAMATH_GPT_total_distance_walked_l2290_229060


namespace NUMINAMATH_GPT_fraction_subtraction_l2290_229019

theorem fraction_subtraction (a : ℝ) (h : a ≠ 0) : 1 / a - 3 / a = -2 / a := 
by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l2290_229019


namespace NUMINAMATH_GPT_time_between_ticks_at_6_oclock_l2290_229020

theorem time_between_ticks_at_6_oclock (ticks6 ticks12 intervals6 intervals12 total_time12: ℕ) (time_per_tick : ℕ) :
  ticks6 = 6 →
  ticks12 = 12 →
  total_time12 = 66 →
  intervals12 = ticks12 - 1 →
  time_per_tick = total_time12 / intervals12 →
  intervals6 = ticks6 - 1 →
  (time_per_tick * intervals6) = 30 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_time_between_ticks_at_6_oclock_l2290_229020


namespace NUMINAMATH_GPT_infinite_geometric_series_correct_l2290_229062

noncomputable def infinite_geometric_series_sum : ℚ :=
  let a : ℚ := 5 / 3
  let r : ℚ := -9 / 20
  a / (1 - r)

theorem infinite_geometric_series_correct : infinite_geometric_series_sum = 100 / 87 := 
by
  sorry

end NUMINAMATH_GPT_infinite_geometric_series_correct_l2290_229062


namespace NUMINAMATH_GPT_closest_point_to_line_l2290_229088

theorem closest_point_to_line {x y : ℝ} :
  (y = 2 * x - 7) → (∃ p : ℝ × ℝ, p.1 = 5 ∧ p.2 = 3 ∧ (p.1, p.2) ∈ {q : ℝ × ℝ | q.2 = 2 * q.1 - 7} ∧ (∀ q : ℝ × ℝ, q ∈ {q : ℝ × ℝ | q.2 = 2 * q.1 - 7} → dist ⟨x, y⟩ p ≤ dist ⟨x, y⟩ q)) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_closest_point_to_line_l2290_229088


namespace NUMINAMATH_GPT_gcd_proof_l2290_229013

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end NUMINAMATH_GPT_gcd_proof_l2290_229013


namespace NUMINAMATH_GPT_percentage_increase_first_to_second_l2290_229024

theorem percentage_increase_first_to_second (D1 D2 D3 : ℕ) (h1 : D2 = 12)
  (h2 : D3 = D2 + Nat.div (D2 * 25) 100) (h3 : D1 + D2 + D3 = 37) :
  Nat.div ((D2 - D1) * 100) D1 = 20 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_first_to_second_l2290_229024


namespace NUMINAMATH_GPT_zoey_holidays_l2290_229078

def visits_per_year (visits_per_month : ℕ) (months_per_year : ℕ) : ℕ :=
  visits_per_month * months_per_year

def visits_every_two_months (months_per_year : ℕ) : ℕ :=
  months_per_year / 2

def visits_every_four_months (visits_per_period : ℕ) (periods_per_year : ℕ) : ℕ :=
  visits_per_period * periods_per_year

theorem zoey_holidays (visits_per_month_first : ℕ) 
                      (months_per_year : ℕ) 
                      (visits_per_period_third : ℕ) 
                      (periods_per_year : ℕ) : 
  visits_per_year visits_per_month_first months_per_year 
  + visits_every_two_months months_per_year 
  + visits_every_four_months visits_per_period_third periods_per_year = 39 := 
  by 
  sorry

end NUMINAMATH_GPT_zoey_holidays_l2290_229078


namespace NUMINAMATH_GPT_compare_logs_l2290_229034

noncomputable def a : ℝ := Real.log 2 / Real.log 5
noncomputable def b : ℝ := Real.log 3 / Real.log 8
noncomputable def c : ℝ := 1 / 2

theorem compare_logs : a < c ∧ c < b := by
  sorry

end NUMINAMATH_GPT_compare_logs_l2290_229034


namespace NUMINAMATH_GPT_Monica_tiles_count_l2290_229055

noncomputable def total_tiles (length width : ℕ) := 
  let double_border_tiles := (2 * ((length - 4) + (width - 4)) + 8)
  let inner_area := (length - 4) * (width - 4)
  let three_foot_tiles := (inner_area + 8) / 9
  double_border_tiles + three_foot_tiles

theorem Monica_tiles_count : total_tiles 18 24 = 183 := 
by
  sorry

end NUMINAMATH_GPT_Monica_tiles_count_l2290_229055


namespace NUMINAMATH_GPT_find_xyz_l2290_229067

def divisible_by (n k : ℕ) : Prop := k % n = 0

def is_7_digit_number (a b c d e f g : ℕ) : ℕ := 
  10^6 * a + 10^5 * b + 10^4 * c + 10^3 * d + 10^2 * e + 10 * f + g

theorem find_xyz
  (x y z : ℕ)
  (h : divisible_by 792 (is_7_digit_number 1 4 x y 7 8 z))
  : (100 * x + 10 * y + z) = 644 :=
by
  sorry

end NUMINAMATH_GPT_find_xyz_l2290_229067


namespace NUMINAMATH_GPT_arithmetic_sequence_l2290_229064

-- Define the nth term of the arithmetic sequence
def a_n (n : ℕ) (d a1 : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def S_n (n : ℕ) (d a1 : ℤ) : ℤ := n * a1 + (n * (n - 1)) / 2 * d

-- Given conditions
theorem arithmetic_sequence (n : ℕ) (d a1 : ℤ) (S3 : ℤ) (h1 : a1 = 10) (h2 : S_n 3 d a1 = 24) :
  (a_n n d a1 = 12 - 2 * n) ∧ (S_n n (-2) 12 = -n^2 + 11 * n) ∧ (∀ k, S_n k (-2) 12 ≤ 30) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_l2290_229064


namespace NUMINAMATH_GPT_cone_sphere_ratio_l2290_229072

/-- A right circular cone and a sphere have bases with the same radius r. 
If the volume of the cone is one-third that of the sphere, find the ratio of 
the altitude of the cone to the radius of its base. -/
theorem cone_sphere_ratio (r h : ℝ) (h_pos : 0 < r) 
    (volume_cone : ℝ) (volume_sphere : ℝ)
    (cone_volume_formula : volume_cone = (1 / 3) * π * r^2 * h) 
    (sphere_volume_formula : volume_sphere = (4 / 3) * π * r^3) 
    (volume_relation : volume_cone = (1 / 3) * volume_sphere) : 
    h / r = 4 / 3 :=
by
    sorry

end NUMINAMATH_GPT_cone_sphere_ratio_l2290_229072


namespace NUMINAMATH_GPT_proof_problem_l2290_229018

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

theorem proof_problem (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) : f x * f (-x) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_proof_problem_l2290_229018


namespace NUMINAMATH_GPT_mrs_hilt_total_candy_l2290_229073

theorem mrs_hilt_total_candy :
  (2 * 3) + (4 * 2) + (6 * 4) = 38 :=
by
  -- here, skip the proof as instructed
  sorry

end NUMINAMATH_GPT_mrs_hilt_total_candy_l2290_229073


namespace NUMINAMATH_GPT_tan_ratio_l2290_229059

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5 / 8) (h2 : Real.sin (x - y) = 1 / 4) : 
  (Real.tan x / Real.tan y) = 2 := 
by
  sorry 

end NUMINAMATH_GPT_tan_ratio_l2290_229059


namespace NUMINAMATH_GPT_avg_of_last_11_eq_41_l2290_229010

def sum_of_first_11 : ℕ := 11 * 48
def sum_of_all_21 : ℕ := 21 * 44
def eleventh_number : ℕ := 55

theorem avg_of_last_11_eq_41 (S1 S : ℕ) :
  S1 = sum_of_first_11 →
  S = sum_of_all_21 →
  (S - S1 + eleventh_number) / 11 = 41 :=
by
  sorry

end NUMINAMATH_GPT_avg_of_last_11_eq_41_l2290_229010


namespace NUMINAMATH_GPT_rhombus_area_l2290_229074

-- Definition of a rhombus with given conditions
structure Rhombus where
  side : ℝ
  d1 : ℝ
  d2 : ℝ

noncomputable def Rhombus.area (r : Rhombus) : ℝ :=
  (r.d1 * r.d2) / 2

noncomputable example : Rhombus :=
{ side := 20,
  d1 := 16,
  d2 := 8 * Real.sqrt 21 }

theorem rhombus_area : 
  let r : Rhombus := { side := 20, d1 := 16, d2 := 8 * Real.sqrt 21 }
  Rhombus.area r = 64 * Real.sqrt 21 :=
by
  let r : Rhombus := { side := 20, d1 := 16, d2 := 8 * Real.sqrt 21 }
  sorry

end NUMINAMATH_GPT_rhombus_area_l2290_229074


namespace NUMINAMATH_GPT_length_of_AX_l2290_229012

theorem length_of_AX 
  (A B C X : Type) 
  (AB AC BC AX BX : ℕ) 
  (hx : AX + BX = AB)
  (h_angle_bisector : AC * BX = BC * AX)
  (h_AB : AB = 40)
  (h_BC : BC = 35)
  (h_AC : AC = 21) : 
  AX = 15 :=
by
  sorry

end NUMINAMATH_GPT_length_of_AX_l2290_229012


namespace NUMINAMATH_GPT_neg_exists_n_sq_gt_two_pow_n_l2290_229036

open Classical

theorem neg_exists_n_sq_gt_two_pow_n :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by
  sorry

end NUMINAMATH_GPT_neg_exists_n_sq_gt_two_pow_n_l2290_229036


namespace NUMINAMATH_GPT_percentage_decrease_l2290_229045

theorem percentage_decrease (original_price new_price : ℝ) (h1 : original_price = 1400) (h2 : new_price = 1064) :
  ((original_price - new_price) / original_price * 100) = 24 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_l2290_229045


namespace NUMINAMATH_GPT_least_perimeter_of_triangle_l2290_229046

theorem least_perimeter_of_triangle (cosA cosB cosC : ℝ)
  (h₁ : cosA = 13 / 16)
  (h₂ : cosB = 4 / 5)
  (h₃ : cosC = -3 / 5) :
  ∃ a b c : ℕ, a + b + c = 28 ∧ 
  a^2 + b^2 - c^2 = 2 * a * b * cosC ∧ 
  b^2 + c^2 - a^2 = 2 * b * c * cosA ∧ 
  c^2 + a^2 - b^2 = 2 * c * a * cosB :=
sorry

end NUMINAMATH_GPT_least_perimeter_of_triangle_l2290_229046


namespace NUMINAMATH_GPT_minimize_y_at_x_l2290_229023

-- Define the function y
def y (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 3 * (x - a) ^ 2 + (x - b) ^ 2 

-- State the theorem
theorem minimize_y_at_x (a b : ℝ) : ∃ x : ℝ, (∀ x' : ℝ, y x' a b ≥ y ((3 * a + b) / 4) a b) :=
by
  sorry

end NUMINAMATH_GPT_minimize_y_at_x_l2290_229023


namespace NUMINAMATH_GPT_isosceles_obtuse_triangle_angle_correct_l2290_229092

noncomputable def isosceles_obtuse_triangle_smallest_angle (A B C : ℝ) (h1 : A = 1.3 * 90) (h2 : B = C) (h3 : A + B + C = 180) : ℝ :=
  (180 - A) / 2

theorem isosceles_obtuse_triangle_angle_correct 
  (A B C : ℝ)
  (h1 : A = 1.3 * 90)
  (h2 : B = C)
  (h3 : A + B + C = 180) :
  isosceles_obtuse_triangle_smallest_angle A B C h1 h2 h3 = 31.5 :=
sorry

end NUMINAMATH_GPT_isosceles_obtuse_triangle_angle_correct_l2290_229092


namespace NUMINAMATH_GPT_highest_probability_face_l2290_229028

theorem highest_probability_face :
  let faces := 6
  let face_3 := 3
  let face_2 := 2
  let face_1 := 1
  (face_3 / faces > face_2 / faces) ∧ (face_2 / faces > face_1 / faces) →
  (face_3 / faces > face_1 / faces) →
  (face_3 = 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_highest_probability_face_l2290_229028


namespace NUMINAMATH_GPT_domain_of_function_l2290_229097

theorem domain_of_function (x : ℝ) (k : ℤ) :
  ∃ x, (2 * Real.sin x + 1 ≥ 0) ↔ (- (Real.pi / 6) + 2 * k * Real.pi ≤ x ∧ x ≤ (7 * Real.pi / 6) + 2 * k * Real.pi) :=
sorry

end NUMINAMATH_GPT_domain_of_function_l2290_229097


namespace NUMINAMATH_GPT_total_number_of_legs_is_40_l2290_229009

-- Define the number of octopuses Carson saw.
def number_of_octopuses := 5

-- Define the number of legs per octopus.
def legs_per_octopus := 8

-- Define the total number of octopus legs Carson saw.
def total_octopus_legs : Nat := number_of_octopuses * legs_per_octopus

-- Prove that the total number of octopus legs Carson saw is 40.
theorem total_number_of_legs_is_40 : total_octopus_legs = 40 := by
  sorry

end NUMINAMATH_GPT_total_number_of_legs_is_40_l2290_229009


namespace NUMINAMATH_GPT_convex_polygon_diagonals_l2290_229094

theorem convex_polygon_diagonals (n : ℕ) (h_n : n = 25) : 
  (n * (n - 3)) / 2 = 275 :=
by
  sorry

end NUMINAMATH_GPT_convex_polygon_diagonals_l2290_229094


namespace NUMINAMATH_GPT_small_circle_to_large_circle_ratio_l2290_229006

theorem small_circle_to_large_circle_ratio (a b : ℝ) (h : π * b^2 - π * a^2 = 3 * π * a^2) :
  a / b = 1 / 2 :=
sorry

end NUMINAMATH_GPT_small_circle_to_large_circle_ratio_l2290_229006


namespace NUMINAMATH_GPT_larger_segment_length_l2290_229022

open Real

theorem larger_segment_length (a b c : ℝ) (h : a = 50 ∧ b = 110 ∧ c = 120) :
  ∃ x : ℝ, x = 100 ∧ (∃ h : ℝ, a^2 = x^2 + h^2 ∧ b^2 = (c - x)^2 + h^2) :=
by
  sorry

end NUMINAMATH_GPT_larger_segment_length_l2290_229022


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l2290_229053

-- Definition for part 1
noncomputable def f_part1 (x : ℝ) := abs (x - 3) + 2 * x

-- Proof statement for part 1
theorem part1_solution (x : ℝ) : (f_part1 x ≥ 3) ↔ (x ≥ 0) :=
by sorry

-- Definition for part 2
noncomputable def f_part2 (x a : ℝ) := abs (x - a) + 2 * x

-- Proof statement for part 2
theorem part2_solution (a : ℝ) : 
  (∀ x, f_part2 x a ≤ 0 ↔ x ≤ -2) → (a = 2 ∨ a = -6) :=
by sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l2290_229053


namespace NUMINAMATH_GPT_part1_part2_l2290_229096

noncomputable def A : Set ℝ := {x | x^2 + 4 * x = 0 }

noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0 }

-- Part (1): Prove a = 1 given A ∪ B = B
theorem part1 (a : ℝ) (h : A ∪ B a = B a) : a = 1 :=
sorry

-- Part (2): Prove the set C composed of the values of a given A ∩ B = B
def C : Set ℝ := {a | a ≤ -1 ∨ a = 1}

theorem part2 (h : ∀ a, A ∩ B a = B a ↔ a ∈ C) : forall a, A ∩ B a = B a ↔ a ∈ C :=
sorry

end NUMINAMATH_GPT_part1_part2_l2290_229096


namespace NUMINAMATH_GPT_doves_eggs_l2290_229003

theorem doves_eggs (initial_doves total_doves : ℕ) (fraction_hatched : ℚ) (E : ℕ)
  (h_initial_doves : initial_doves = 20)
  (h_total_doves : total_doves = 65)
  (h_fraction_hatched : fraction_hatched = 3/4)
  (h_after_hatching : total_doves = initial_doves + fraction_hatched * E * initial_doves) :
  E = 3 :=
by
  -- The proof would go here.
  sorry

end NUMINAMATH_GPT_doves_eggs_l2290_229003


namespace NUMINAMATH_GPT_sum_n_10_terms_progression_l2290_229061

noncomputable def sum_arith_progression (n a d : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_n_10_terms_progression :
  ∃ (a : ℕ), (∃ (n : ℕ), sum_arith_progression n a 3 = 220) ∧
  (2 * a + (10 - 1) * 3) = 43 ∧
  sum_arith_progression 10 a 3 = 215 :=
by sorry

end NUMINAMATH_GPT_sum_n_10_terms_progression_l2290_229061


namespace NUMINAMATH_GPT_find_intersection_complement_B_find_A_minus_B_find_A_minus_A_minus_B_l2290_229063

def U := Set ℝ
def A : Set ℝ := {x | x > 4}
def B : Set ℝ := {x | -6 < x ∧ x < 6}

theorem find_intersection (x : ℝ) : x ∈ A ∧ x ∈ B ↔ 4 < x ∧ x < 6 :=
by
  sorry

theorem complement_B (x : ℝ) : x ∉ B ↔ x ≥ 6 ∨ x ≤ -6 :=
by
  sorry

def A_minus_B : Set ℝ := {x | x ∈ A ∧ x ∉ B}

theorem find_A_minus_B (x : ℝ) : x ∈ A_minus_B ↔ x ≥ 6 :=
by
  sorry

theorem find_A_minus_A_minus_B (x : ℝ) : x ∈ (A \ A_minus_B) ↔ 4 < x ∧ x < 6 :=
by
  sorry

end NUMINAMATH_GPT_find_intersection_complement_B_find_A_minus_B_find_A_minus_A_minus_B_l2290_229063


namespace NUMINAMATH_GPT_determinant_tan_matrix_l2290_229004

theorem determinant_tan_matrix (B C : ℝ) (h : B + C = 3 * π / 4) :
  Matrix.det ![
    ![Real.tan (π / 4), 1, 1],
    ![1, Real.tan B, 1],
    ![1, 1, Real.tan C]
  ] = 1 :=
by
  sorry

end NUMINAMATH_GPT_determinant_tan_matrix_l2290_229004


namespace NUMINAMATH_GPT_weight_of_B_l2290_229042

theorem weight_of_B (A B C : ℝ) 
  (h1 : A + B + C = 135) 
  (h2 : A + B = 80) 
  (h3 : B + C = 86) : 
  B = 31 :=
sorry

end NUMINAMATH_GPT_weight_of_B_l2290_229042


namespace NUMINAMATH_GPT_tank_capacity_l2290_229084

theorem tank_capacity (C : ℝ) 
  (h1 : 10 > 0) 
  (h2 : 16 > (10 : ℝ))
  (h3 : ((C/10) - 480 = (C/16))) : C = 1280 := 
by 
  sorry

end NUMINAMATH_GPT_tank_capacity_l2290_229084


namespace NUMINAMATH_GPT_monotonic_conditions_fixed_point_property_l2290_229090

noncomputable
def f (x a b c : ℝ) : ℝ := x^3 - a * x^2 - b * x + c

theorem monotonic_conditions (a b c : ℝ) :
  a = 0 ∧ c = 0 ∧ b ≤ 3 ↔ ∀ x : ℝ, (x ≥ 1 → (f x a b c) ≥ 1) → ∀ x y: ℝ, (x ≥ y ↔ f x a b c ≤ f y a b c) := sorry

theorem fixed_point_property (a b c : ℝ) :
  (∀ x : ℝ, (x ≥ 1 ∧ (f x a b c) ≥ 1) → f (f x a b c) a b c = x) ↔ (f x 0 b 0 = x) := sorry

end NUMINAMATH_GPT_monotonic_conditions_fixed_point_property_l2290_229090


namespace NUMINAMATH_GPT_relationship_y1_y2_l2290_229016

theorem relationship_y1_y2 (y1 y2 : ℤ) 
  (h1 : y1 = 2 * -3 + 1) 
  (h2 : y2 = 2 * 4 + 1) : y1 < y2 :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_relationship_y1_y2_l2290_229016


namespace NUMINAMATH_GPT_fifth_element_is_17_l2290_229037

-- Define the sequence pattern based on given conditions
def seq : ℕ → ℤ 
| 0 => 5    -- first element
| 1 => -8   -- second element
| n + 2 => seq n + 3    -- each following element is calculated by adding 3 to the two positions before

-- Additional condition: the sign of sequence based on position
def seq_sign : ℕ → ℤ
| n => if n % 2 = 0 then 1 else -1

-- The final adjusted sequence based on the above observations
def final_seq (n : ℕ) : ℤ := seq n * seq_sign n

-- Assert the expected outcome for the 5th element
theorem fifth_element_is_17 : final_seq 4 = 17 :=
by
  sorry

end NUMINAMATH_GPT_fifth_element_is_17_l2290_229037


namespace NUMINAMATH_GPT_molecular_weight_is_171_35_l2290_229081

def atomic_weight_ba : ℝ := 137.33
def atomic_weight_o : ℝ := 16.00
def atomic_weight_h : ℝ := 1.01

def molecular_weight : ℝ :=
  (1 * atomic_weight_ba) + (2 * atomic_weight_o) + (2 * atomic_weight_h)

-- The goal is to prove that the molecular weight is 171.35
theorem molecular_weight_is_171_35 : molecular_weight = 171.35 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_is_171_35_l2290_229081


namespace NUMINAMATH_GPT_correct_weight_misread_l2290_229008

theorem correct_weight_misread : 
  ∀ (x : ℝ) (n : ℝ) (avg1 : ℝ) (avg2 : ℝ) (misread : ℝ),
  n = 20 → avg1 = 58.4 → avg2 = 59 → misread = 56 → 
  (n * avg2 - n * avg1 + misread) = x → 
  x = 68 :=
by
  intros x n avg1 avg2 misread
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_correct_weight_misread_l2290_229008


namespace NUMINAMATH_GPT_find_x_orthogonal_l2290_229044

theorem find_x_orthogonal :
  ∃ x : ℝ, (2 * x + 5 * (-3) = 0) ∧ x = 15 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_orthogonal_l2290_229044


namespace NUMINAMATH_GPT_train_speed_faster_l2290_229030

-- The Lean statement of the problem
theorem train_speed_faster (Vs : ℝ) (L : ℝ) (T : ℝ) (Vf : ℝ) :
  Vs = 36 ∧ L = 340 ∧ T = 17 ∧ (Vf - Vs) * (5 / 18) = L / T → Vf = 108 :=
by 
  intros 
  sorry

end NUMINAMATH_GPT_train_speed_faster_l2290_229030


namespace NUMINAMATH_GPT_urn_gold_coins_percentage_l2290_229082

theorem urn_gold_coins_percentage (obj_perc_beads : ℝ) (coins_perc_gold : ℝ) : 
    obj_perc_beads = 0.15 → coins_perc_gold = 0.65 → 
    (1 - obj_perc_beads) * coins_perc_gold = 0.5525 := 
by
  intros h_obj_perc_beads h_coins_perc_gold
  sorry

end NUMINAMATH_GPT_urn_gold_coins_percentage_l2290_229082


namespace NUMINAMATH_GPT_cubes_side_length_l2290_229007

theorem cubes_side_length (s : ℝ) (h : 2 * (s * s + s * 2 * s + s * 2 * s) = 10) : s = 1 :=
by
  sorry

end NUMINAMATH_GPT_cubes_side_length_l2290_229007


namespace NUMINAMATH_GPT_michael_exceeds_suresh_by_36_5_l2290_229027

noncomputable def shares_total : ℝ := 730
noncomputable def punith_ratio_to_michael : ℝ := 3 / 4
noncomputable def michael_ratio_to_suresh : ℝ := 3.5 / 3

theorem michael_exceeds_suresh_by_36_5 :
  ∃ P M S : ℝ, P + M + S = shares_total
  ∧ (P / M = punith_ratio_to_michael)
  ∧ (M / S = michael_ratio_to_suresh)
  ∧ (M - S = 36.5) :=
by
  sorry

end NUMINAMATH_GPT_michael_exceeds_suresh_by_36_5_l2290_229027


namespace NUMINAMATH_GPT_emily_gardens_and_seeds_l2290_229079

variables (total_seeds planted_big_garden tom_seeds lettuce_seeds pepper_seeds tom_gardens lettuce_gardens pepper_gardens : ℕ)

def seeds_left (total_seeds planted_big_garden : ℕ) : ℕ :=
  total_seeds - planted_big_garden

def seeds_used_tomatoes (tom_seeds tom_gardens : ℕ) : ℕ :=
  tom_seeds * tom_gardens

def seeds_used_lettuce (lettuce_seeds lettuce_gardens : ℕ) : ℕ :=
  lettuce_seeds * lettuce_gardens

def seeds_used_peppers (pepper_seeds pepper_gardens : ℕ) : ℕ :=
  pepper_seeds * pepper_gardens

def remaining_seeds (total_seeds planted_big_garden tom_seeds tom_gardens lettuce_seeds lettuce_gardens : ℕ) : ℕ :=
  seeds_left total_seeds planted_big_garden - (seeds_used_tomatoes tom_seeds tom_gardens + seeds_used_lettuce lettuce_seeds lettuce_gardens)

def total_small_gardens (tom_gardens lettuce_gardens pepper_gardens : ℕ) : ℕ :=
  tom_gardens + lettuce_gardens + pepper_gardens

theorem emily_gardens_and_seeds :
  total_seeds = 42 ∧
  planted_big_garden = 36 ∧
  tom_seeds = 4 ∧
  lettuce_seeds = 3 ∧
  pepper_seeds = 2 ∧
  tom_gardens = 3 ∧
  lettuce_gardens = 2 →
  seeds_used_peppers pepper_seeds pepper_gardens = 0 ∧
  total_small_gardens tom_gardens lettuce_gardens pepper_gardens = 5 :=
by
  sorry

end NUMINAMATH_GPT_emily_gardens_and_seeds_l2290_229079


namespace NUMINAMATH_GPT_count_integer_b_for_log_b_256_l2290_229005

theorem count_integer_b_for_log_b_256 :
  (∃ b : ℕ, b > 1 ∧ ∃ n : ℕ, n > 0 ∧ b ^ n = 256) ∧ 
  (∀ b : ℕ, (b > 1 ∧ ∃ n : ℕ, n > 0 ∧ b ^ n = 256) → (b = 2 ∨ b = 4 ∨ b = 16 ∨ b = 256)) :=
by sorry

end NUMINAMATH_GPT_count_integer_b_for_log_b_256_l2290_229005


namespace NUMINAMATH_GPT_lcm_of_product_of_mutually_prime_l2290_229056

theorem lcm_of_product_of_mutually_prime (a b : ℕ) (h : Nat.gcd a b = 1) : Nat.lcm a b = a * b :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_product_of_mutually_prime_l2290_229056


namespace NUMINAMATH_GPT_alpha_beta_sum_l2290_229041

theorem alpha_beta_sum (α β : ℝ) (h : ∀ x : ℝ, x ≠ 54 → x ≠ -60 → (x - α) / (x + β) = (x^2 - 72 * x + 945) / (x^2 + 45 * x - 3240)) :
  α + β = 81 :=
sorry

end NUMINAMATH_GPT_alpha_beta_sum_l2290_229041


namespace NUMINAMATH_GPT_leap_day_2040_is_tuesday_l2290_229069

def days_in_non_leap_year := 365
def days_in_leap_year := 366
def leap_years_between_2000_and_2040 := 10

def total_days_between_2000_and_2040 := 
  30 * days_in_non_leap_year + leap_years_between_2000_and_2040 * days_in_leap_year

theorem leap_day_2040_is_tuesday :
  (total_days_between_2000_and_2040 % 7) = 0 :=
by
  sorry

end NUMINAMATH_GPT_leap_day_2040_is_tuesday_l2290_229069


namespace NUMINAMATH_GPT_cooking_oil_distribution_l2290_229058

theorem cooking_oil_distribution (total_oil : ℝ) (oil_A : ℝ) (oil_B : ℝ) (oil_C : ℝ)
    (h_total_oil : total_oil = 3 * 1000) -- Total oil is 3000 milliliters
    (h_A_B : oil_A = oil_B + 200) -- A receives 200 milliliters more than B
    (h_B_C : oil_B = oil_C + 200) -- B receives 200 milliliters more than C
    : oil_B = 1000 :=              -- We need to prove B receives 1000 milliliters
by
  sorry

end NUMINAMATH_GPT_cooking_oil_distribution_l2290_229058


namespace NUMINAMATH_GPT_find_number_l2290_229000

theorem find_number (x : ℝ) (h : x - (3 / 5) * x = 56) : x = 140 := 
by {
  -- The proof would be written here,
  -- but it is indicated to skip it using "sorry"
  sorry
}

end NUMINAMATH_GPT_find_number_l2290_229000


namespace NUMINAMATH_GPT_num_ordered_triples_unique_l2290_229049

theorem num_ordered_triples_unique : 
  (∃! (x y z : ℝ), x + y = 2 ∧ xy - z^2 = 1) := 
by 
  sorry 

end NUMINAMATH_GPT_num_ordered_triples_unique_l2290_229049


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l2290_229031

theorem sum_of_squares_of_roots (s₁ s₂ : ℝ) (h1 : s₁ + s₂ = 10) (h2 : s₁ * s₂ = 9) : 
  s₁^2 + s₂^2 = 82 := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l2290_229031


namespace NUMINAMATH_GPT_arithmetic_sequence_a1_value_l2290_229001

theorem arithmetic_sequence_a1_value (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 3 = -6) 
  (h2 : a 7 = a 5 + 4) 
  (h_seq : ∀ n, a (n+1) = a n + d) : 
  a 1 = -10 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a1_value_l2290_229001


namespace NUMINAMATH_GPT_equivalent_functions_l2290_229043

theorem equivalent_functions :
  ∀ (x t : ℝ), (x^2 - 2*x - 1 = t^2 - 2*t + 1) := 
by
  intros x t
  sorry

end NUMINAMATH_GPT_equivalent_functions_l2290_229043


namespace NUMINAMATH_GPT_no_real_roots_iff_k_lt_neg_one_l2290_229026

theorem no_real_roots_iff_k_lt_neg_one (k : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - k ≠ 0) ↔ k < -1 :=
by sorry

end NUMINAMATH_GPT_no_real_roots_iff_k_lt_neg_one_l2290_229026


namespace NUMINAMATH_GPT_four_gt_sqrt_fifteen_l2290_229075

theorem four_gt_sqrt_fifteen : 4 > Real.sqrt 15 := 
sorry

end NUMINAMATH_GPT_four_gt_sqrt_fifteen_l2290_229075


namespace NUMINAMATH_GPT_find_diameter_C_l2290_229065

noncomputable def diameter_of_circle_C (diameter_of_D : ℝ) (ratio_shaded_to_C : ℝ) : ℝ :=
  let radius_D := diameter_of_D / 2
  let radius_C := radius_D / (2 * Real.sqrt ratio_shaded_to_C)
  2 * radius_C

theorem find_diameter_C :
  let diameter_D := 20
  let ratio_shaded_area_to_C := 7
  diameter_of_circle_C diameter_D ratio_shaded_area_to_C = 5 * Real.sqrt 2 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_find_diameter_C_l2290_229065
