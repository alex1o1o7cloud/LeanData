import Mathlib

namespace NUMINAMATH_GPT_simplify_fraction_mul_l443_44346

theorem simplify_fraction_mul (a b c d : ℕ) (h1 : 405 = 27 * a) (h2 : 1215 = 27 * b) (h3 : a / d = 1) (h4 : b / d = 3) : (a / d) * (27 : ℕ) = 9 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_mul_l443_44346


namespace NUMINAMATH_GPT_carol_total_points_l443_44358

/-- Conditions -/
def first_round_points : ℤ := 17
def second_round_points : ℤ := 6
def last_round_points : ℤ := -16

/-- Proof problem statement -/
theorem carol_total_points : first_round_points + second_round_points + last_round_points = 7 := by
  sorry

end NUMINAMATH_GPT_carol_total_points_l443_44358


namespace NUMINAMATH_GPT_arithmetic_seq_40th_term_l443_44369

theorem arithmetic_seq_40th_term (a₁ d : ℕ) (n : ℕ) (h1 : a₁ = 3) (h2 : d = 4) (h3 : n = 40) : 
  a₁ + (n - 1) * d = 159 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_40th_term_l443_44369


namespace NUMINAMATH_GPT_total_goals_correct_l443_44318

-- Define the number of goals scored by each team in each period
def kickers_first_period_goals : ℕ := 2
def kickers_second_period_goals : ℕ := 2 * kickers_first_period_goals
def spiders_first_period_goals : ℕ := (1 / 2) * kickers_first_period_goals
def spiders_second_period_goals : ℕ := 2 * kickers_second_period_goals

-- Define the total goals scored by both teams
def total_goals : ℕ :=
  kickers_first_period_goals + 
  kickers_second_period_goals + 
  spiders_first_period_goals + 
  spiders_second_period_goals

-- State the theorem to be proved
theorem total_goals_correct : total_goals = 15 := by
  sorry

end NUMINAMATH_GPT_total_goals_correct_l443_44318


namespace NUMINAMATH_GPT_solve_for_x_l443_44331

noncomputable def g (x : ℝ) : ℝ := (Real.sqrt (x + 2) / 5) ^ (1 / 4)

theorem solve_for_x : 
  ∃ x : ℝ, g (3 * x) = 3 * g x ∧ x = -404 / 201 := 
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l443_44331


namespace NUMINAMATH_GPT_inequality_solution_l443_44359

theorem inequality_solution (x : ℝ) :
  (x ≠ -1 ∧ x ≠ -2 ∧ x ≠ 5) →
  ((x * x - 4 * x - 5) / (x * x + 3 * x + 2) < 0 ↔ (x ∈ Set.Ioo (-2:ℝ) (-1:ℝ) ∨ x ∈ Set.Ioo (-1:ℝ) (5:ℝ))) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l443_44359


namespace NUMINAMATH_GPT_function_machine_output_is_38_l443_44360

def function_machine (input : ℕ) : ℕ :=
  let multiplied := input * 3
  if multiplied > 40 then
    multiplied - 7
  else
    multiplied + 10

theorem function_machine_output_is_38 :
  function_machine 15 = 38 :=
by
   sorry

end NUMINAMATH_GPT_function_machine_output_is_38_l443_44360


namespace NUMINAMATH_GPT_smallest_positive_integer_l443_44389

theorem smallest_positive_integer (n : ℕ) :
  (n % 45 = 0 ∧ n % 60 = 0 ∧ n % 25 ≠ 0 ↔ n = 180) :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_l443_44389


namespace NUMINAMATH_GPT_largest_divisor_of_n5_minus_n_l443_44391

theorem largest_divisor_of_n5_minus_n (n : ℤ) : 
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n^5 - n)) ∧ d = 30 :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_n5_minus_n_l443_44391


namespace NUMINAMATH_GPT_leak_emptying_time_l443_44376

theorem leak_emptying_time (fill_rate_no_leak : ℝ) (combined_rate_with_leak : ℝ) (L : ℝ) :
  fill_rate_no_leak = 1/10 →
  combined_rate_with_leak = 1/12 →
  fill_rate_no_leak - L = combined_rate_with_leak →
  1 / L = 60 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_leak_emptying_time_l443_44376


namespace NUMINAMATH_GPT_sum_of_k_l443_44336

theorem sum_of_k : ∃ (k_vals : List ℕ), 
  (∀ k ∈ k_vals, ∃ α β : ℤ, α + β = k ∧ α * β = -20) 
  ∧ k_vals.sum = 29 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_k_l443_44336


namespace NUMINAMATH_GPT_base_9_perfect_square_b_l443_44398

theorem base_9_perfect_square_b (b : ℕ) (a : ℕ) 
  (h0 : 0 < b) (h1 : b < 9) (h2 : a < 9) : 
  ∃ n, n^2 ≡ 729 * b + 81 * a + 54 [MOD 81] :=
sorry

end NUMINAMATH_GPT_base_9_perfect_square_b_l443_44398


namespace NUMINAMATH_GPT_total_stamps_received_l443_44352

theorem total_stamps_received
  (initial_stamps : ℕ)
  (final_stamps : ℕ)
  (received_stamps : ℕ)
  (h_initial : initial_stamps = 34)
  (h_final : final_stamps = 61)
  (h_received : received_stamps = final_stamps - initial_stamps) :
  received_stamps = 27 :=
by 
  sorry

end NUMINAMATH_GPT_total_stamps_received_l443_44352


namespace NUMINAMATH_GPT_difference_in_soda_bottles_l443_44315

-- Define the given conditions
def regular_soda_bottles : ℕ := 81
def diet_soda_bottles : ℕ := 60

-- Define the difference in the number of bottles
def difference_bottles : ℕ := regular_soda_bottles - diet_soda_bottles

-- The theorem we want to prove
theorem difference_in_soda_bottles : difference_bottles = 21 := by
  sorry

end NUMINAMATH_GPT_difference_in_soda_bottles_l443_44315


namespace NUMINAMATH_GPT_part1_part2_l443_44313

theorem part1 (a m n : ℕ) (ha : a > 1) (hdiv : a^m + 1 ∣ a^n + 1) : n ∣ m :=
sorry

theorem part2 (a b m n : ℕ) (ha : a > 1) (coprime_ab : Nat.gcd a b = 1) (hdiv : a^m + b^m ∣ a^n + b^n) : n ∣ m :=
sorry

end NUMINAMATH_GPT_part1_part2_l443_44313


namespace NUMINAMATH_GPT_no_int_k_such_that_P_k_equals_8_l443_44304

theorem no_int_k_such_that_P_k_equals_8
    (P : Polynomial ℤ) 
    (a b c d k : ℤ)
    (h0: a ≠ b)
    (h1: a ≠ c)
    (h2: a ≠ d)
    (h3: b ≠ c)
    (h4: b ≠ d)
    (h5: c ≠ d)
    (h6: P.eval a = 5)
    (h7: P.eval b = 5)
    (h8: P.eval c = 5)
    (h9: P.eval d = 5)
    : P.eval k ≠ 8 := by
  sorry

end NUMINAMATH_GPT_no_int_k_such_that_P_k_equals_8_l443_44304


namespace NUMINAMATH_GPT_pounds_lost_per_month_l443_44326

variable (starting_weight : ℕ) (ending_weight : ℕ) (months_in_year : ℕ) 

theorem pounds_lost_per_month
    (h_start : starting_weight = 250)
    (h_end : ending_weight = 154)
    (h_months : months_in_year = 12) :
    (starting_weight - ending_weight) / months_in_year = 8 := 
sorry

end NUMINAMATH_GPT_pounds_lost_per_month_l443_44326


namespace NUMINAMATH_GPT_difference_of_fractions_l443_44388

theorem difference_of_fractions (a : ℝ) (b : ℝ) (h₁ : a = 7000) (h₂ : b = 1/10) :
  (a * b - a * (0.1 / 100)) = 693 :=
by 
  sorry

end NUMINAMATH_GPT_difference_of_fractions_l443_44388


namespace NUMINAMATH_GPT_sin_eleven_pi_over_three_l443_44363

theorem sin_eleven_pi_over_three : Real.sin (11 * Real.pi / 3) = -((Real.sqrt 3) / 2) :=
by
  -- Conversion factor between radians and degrees
  -- periodicity of sine function: sin theta = sin (theta + n * 360 degrees) for any integer n
  -- the sine function is odd: sin (-theta) = -sin theta
  -- sin 60 degrees = sqrt(3)/2
  sorry

end NUMINAMATH_GPT_sin_eleven_pi_over_three_l443_44363


namespace NUMINAMATH_GPT_frank_spent_per_week_l443_44353

theorem frank_spent_per_week (mowing_dollars : ℕ) (weed_eating_dollars : ℕ) (weeks : ℕ) 
    (total_dollars := mowing_dollars + weed_eating_dollars) 
    (spending_rate := total_dollars / weeks) :
    mowing_dollars = 5 → weed_eating_dollars = 58 → weeks = 9 → spending_rate = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end NUMINAMATH_GPT_frank_spent_per_week_l443_44353


namespace NUMINAMATH_GPT_prime_factors_of_product_l443_44329

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def prime_factors (n : ℕ) : List ℕ :=
  -- Assume we have a function that returns a list of prime factors of n
  sorry

def num_distinct_primes (n : ℕ) : ℕ :=
  (prime_factors n).toFinset.card

theorem prime_factors_of_product :
  num_distinct_primes (85 * 87 * 91 * 94) = 8 :=
by
  have prod_factorizations : 85 = 5 * 17 ∧ 87 = 3 * 29 ∧ 91 = 7 * 13 ∧ 94 = 2 * 47 := 
    by sorry -- each factorization step
  sorry

end NUMINAMATH_GPT_prime_factors_of_product_l443_44329


namespace NUMINAMATH_GPT_log_domain_l443_44357

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1) / Real.log 2

theorem log_domain :
  ∀ x : ℝ, (∃ y : ℝ, f y = Real.log (x - 1) / Real.log 2) ↔ x ∈ Set.Ioi 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_log_domain_l443_44357


namespace NUMINAMATH_GPT_arithmetic_sequence_term_number_l443_44361

-- Given:
def first_term : ℕ := 1
def common_difference : ℕ := 3
def target_term : ℕ := 2011

-- To prove:
theorem arithmetic_sequence_term_number :
    ∃ n : ℕ, target_term = first_term + (n - 1) * common_difference ∧ n = 671 := 
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_number_l443_44361


namespace NUMINAMATH_GPT_ancient_chinese_poem_l443_44324

theorem ancient_chinese_poem (x : ℕ) :
  (7 * x + 7 = 9 * (x - 1)) := by
  sorry

end NUMINAMATH_GPT_ancient_chinese_poem_l443_44324


namespace NUMINAMATH_GPT_A_eq_B_l443_44317

def A : Set ℤ := {
  z : ℤ | ∃ x y : ℤ, z = x^2 + 2 * y^2
}

def B : Set ℤ := {
  z : ℤ | ∃ x y : ℤ, z = x^2 - 6 * x * y + 11 * y^2
}

theorem A_eq_B : A = B :=
by {
  sorry
}

end NUMINAMATH_GPT_A_eq_B_l443_44317


namespace NUMINAMATH_GPT_find_N_l443_44334

theorem find_N (N : ℕ) :
  ((5 + 6 + 7 + 8) / 4 = (2014 + 2015 + 2016 + 2017) / N) → N = 1240 :=
by
  sorry

end NUMINAMATH_GPT_find_N_l443_44334


namespace NUMINAMATH_GPT_find_incorrect_value_l443_44377

variable (k b : ℝ)

-- Linear function definition
def linear_function (x : ℝ) : ℝ := k * x + b

-- Given points
theorem find_incorrect_value (h₁ : linear_function k b (-1) = 3)
                             (h₂ : linear_function k b 0 = 2)
                             (h₃ : linear_function k b 1 = 1)
                             (h₄ : linear_function k b 2 = 0)
                             (h₅ : linear_function k b 3 = -2) :
                             (∃ x y, linear_function k b x ≠ y) := by
  sorry

end NUMINAMATH_GPT_find_incorrect_value_l443_44377


namespace NUMINAMATH_GPT_students_like_both_l443_44323

theorem students_like_both (total_students French_fries_likers burger_likers neither_likers : ℕ)
(H1 : total_students = 25)
(H2 : French_fries_likers = 15)
(H3 : burger_likers = 10)
(H4 : neither_likers = 6)
: (French_fries_likers + burger_likers + neither_likers - total_students) = 12 :=
by sorry

end NUMINAMATH_GPT_students_like_both_l443_44323


namespace NUMINAMATH_GPT_find_number_l443_44321

theorem find_number (x : ℝ) (h : 10 * x = 2 * x - 36) : x = -4.5 :=
sorry

end NUMINAMATH_GPT_find_number_l443_44321


namespace NUMINAMATH_GPT_undefined_expression_l443_44309

theorem undefined_expression (b : ℝ) : b^2 - 9 = 0 ↔ b = 3 ∨ b = -3 := by
  sorry

end NUMINAMATH_GPT_undefined_expression_l443_44309


namespace NUMINAMATH_GPT_ratio_of_rooms_l443_44370

def rooms_in_danielle_apartment : Nat := 6
def rooms_in_heidi_apartment : Nat := 3 * rooms_in_danielle_apartment
def rooms_in_grant_apartment : Nat := 2

theorem ratio_of_rooms :
  (rooms_in_grant_apartment : ℚ) / (rooms_in_heidi_apartment : ℚ) = 1 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_rooms_l443_44370


namespace NUMINAMATH_GPT_barbara_wins_l443_44396

theorem barbara_wins (n : ℕ) (h : n = 15) (num_winning_sequences : ℕ) :
  num_winning_sequences = 8320 :=
sorry

end NUMINAMATH_GPT_barbara_wins_l443_44396


namespace NUMINAMATH_GPT_probability_circle_l443_44333

theorem probability_circle (total_figures triangles circles squares : ℕ)
  (h_total : total_figures = 10)
  (h_triangles : triangles = 4)
  (h_circles : circles = 3)
  (h_squares : squares = 3) :
  circles / total_figures = 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_circle_l443_44333


namespace NUMINAMATH_GPT_symmetry_implies_condition_l443_44366

open Function

variable {R : Type*} [Field R]
variables (p q r s : R)

theorem symmetry_implies_condition
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0) 
  (h_symmetry : ∀ x y : R, y = (p * x + q) / (r * x - s) → 
                          -x = (p * (-y) + q) / (r * (-y) - s)) :
  r + s = 0 := 
sorry

end NUMINAMATH_GPT_symmetry_implies_condition_l443_44366


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l443_44301

-- Problem 1
theorem problem_1 (x : ℝ) (h : 4.8 - 3 * x = 1.8) : x = 1 :=
by { sorry }

-- Problem 2
theorem problem_2 (x : ℝ) (h : (1 / 8) / (1 / 5) = x / 24) : x = 15 :=
by { sorry }

-- Problem 3
theorem problem_3 (x : ℝ) (h : 7.5 * x + 6.5 * x = 2.8) : x = 0.2 :=
by { sorry }

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l443_44301


namespace NUMINAMATH_GPT_halfway_between_one_eighth_and_one_third_l443_44371

theorem halfway_between_one_eighth_and_one_third : 
  (1 / 8 + 1 / 3) / 2 = 11 / 48 :=
by
  -- Skipping the proof here
  sorry

end NUMINAMATH_GPT_halfway_between_one_eighth_and_one_third_l443_44371


namespace NUMINAMATH_GPT_plane_equation_l443_44328

theorem plane_equation 
  (s t : ℝ)
  (x y z : ℝ)
  (parametric_plane : ℝ → ℝ → ℝ × ℝ × ℝ)
  (plane_eq : ℝ × ℝ × ℝ → Prop) :
  parametric_plane s t = (2 + 2 * s - t, 1 + 2 * s, 4 - 3 * s + t) →
  plane_eq (x, y, z) ↔ 2 * x - 5 * y + 2 * z - 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_plane_equation_l443_44328


namespace NUMINAMATH_GPT_common_ratio_l443_44356

theorem common_ratio (a_3 S_3 : ℝ) (q : ℝ) 
  (h1 : a_3 = 3 / 2) 
  (h2 : S_3 = 9 / 2)
  (h3 : S_3 = (1 + q + q^2) * a_3 / q^2) :
  q = 1 ∨ q = -1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_common_ratio_l443_44356


namespace NUMINAMATH_GPT_surface_area_of_interior_of_box_l443_44387

-- Definitions from conditions in a)
def length : ℕ := 25
def width : ℕ := 40
def cut_side : ℕ := 4

-- The proof statement we need to prove, using the correct answer from b)
theorem surface_area_of_interior_of_box : 
  (length - 2 * cut_side) * (width - 2 * cut_side) + 2 * (cut_side * (length + width - 2 * cut_side)) = 936 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_interior_of_box_l443_44387


namespace NUMINAMATH_GPT_mixed_number_arithmetic_l443_44303

theorem mixed_number_arithmetic :
  26 * (2 + 4 / 7 - (3 + 1 / 3)) + (3 + 1 / 5 + (2 + 3 / 7)) = -14 - 223 / 735 :=
by
  sorry

end NUMINAMATH_GPT_mixed_number_arithmetic_l443_44303


namespace NUMINAMATH_GPT_find_x_l443_44384

theorem find_x (x : ℤ) (A : Set ℤ) (B : Set ℤ) (hA : A = {1, 4, x}) (hB : B = {1, 2 * x, x ^ 2}) (hinter : A ∩ B = {4, 1}) : x = -2 :=
sorry

end NUMINAMATH_GPT_find_x_l443_44384


namespace NUMINAMATH_GPT_factorial_expression_simplification_l443_44373

theorem factorial_expression_simplification : 
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / (Nat.factorial 8) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_factorial_expression_simplification_l443_44373


namespace NUMINAMATH_GPT_baked_goods_not_eaten_l443_44397

theorem baked_goods_not_eaten : 
  let cookies_initial := 200
  let brownies_initial := 150
  let cupcakes_initial := 100
  
  let cookies_after_wife := cookies_initial - 0.30 * cookies_initial
  let brownies_after_wife := brownies_initial - 0.20 * brownies_initial
  let cupcakes_after_wife := cupcakes_initial / 2
  
  let cookies_after_daughter := cookies_after_wife - 40
  let brownies_after_daughter := brownies_after_wife - 0.15 * brownies_after_wife
  
  let cookies_after_friend := cookies_after_daughter - (cookies_after_daughter / 4)
  let brownies_after_friend := brownies_after_daughter - 0.10 * brownies_after_daughter
  let cupcakes_after_friend := cupcakes_after_wife - 10
  
  let cookies_after_other_friend := cookies_after_friend - 0.05 * cookies_after_friend
  let brownies_after_other_friend := brownies_after_friend - 0.05 * brownies_after_friend
  let cupcakes_after_other_friend := cupcakes_after_friend - 5
  
  let cookies_after_javier := cookies_after_other_friend / 2
  let brownies_after_javier := brownies_after_other_friend / 2
  let cupcakes_after_javier := cupcakes_after_other_friend / 2
  
  let total_remaining := cookies_after_javier + brownies_after_javier + cupcakes_after_javier
  total_remaining = 98 := by
{
  sorry
}

end NUMINAMATH_GPT_baked_goods_not_eaten_l443_44397


namespace NUMINAMATH_GPT_triangle_area_l443_44335

-- Define the vertices of the triangle
def point_A : (ℝ × ℝ) := (0, 0)
def point_B : (ℝ × ℝ) := (8, -3)
def point_C : (ℝ × ℝ) := (4, 7)

-- Function to compute the area of a triangle given its vertices
def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Conjecture the area of triangle ABC is 30.0 square units
theorem triangle_area : area_of_triangle point_A point_B point_C = 30.0 := by
  sorry

end NUMINAMATH_GPT_triangle_area_l443_44335


namespace NUMINAMATH_GPT_problem_solution_l443_44340

noncomputable def problem_statement : Prop :=
  8 * (Real.cos (25 * Real.pi / 180)) ^ 2 - Real.tan (40 * Real.pi / 180) - 4 = Real.sqrt 3

theorem problem_solution : problem_statement :=
by
sorry

end NUMINAMATH_GPT_problem_solution_l443_44340


namespace NUMINAMATH_GPT_paths_mat8_l443_44390

-- Define variables
def grid := [
  ["M", "A", "M", "A", "M"],
  ["A", "T", "A", "T", "A"],
  ["M", "A", "M", "A", "M"],
  ["A", "T", "A", "T", "A"],
  ["M", "A", "M", "A", "M"]
]

def is_adjacent (x1 y1 x2 y2 : Nat): Bool :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y1 = y2 - 1)) ∨ (y1 = y2 ∧ (x1 = x2 + 1 ∨ x1 = x2 - 1))

def count_paths (grid: List (List String)): Nat :=
  -- implementation to count number of paths
  4 * 4 * 2

theorem paths_mat8 (grid: List (List String)): count_paths grid = 32 := by
  sorry

end NUMINAMATH_GPT_paths_mat8_l443_44390


namespace NUMINAMATH_GPT_range_of_a_l443_44310

theorem range_of_a :
  ∀ (a : ℝ), (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → (x-a) / (2 - (x + 1 - a)) > 0)
  ↔ -2 ≤ a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l443_44310


namespace NUMINAMATH_GPT_train_crossing_time_correct_l443_44379

noncomputable def train_crossing_time (speed_kmph : ℕ) (length_m : ℕ) (train_dir_opposite : Bool) : ℕ :=
  if train_dir_opposite then
    let speed_mps := speed_kmph * 1000 / 3600
    let relative_speed := speed_mps + speed_mps
    let total_distance := length_m + length_m
    total_distance / relative_speed
  else 0

theorem train_crossing_time_correct :
  train_crossing_time 54 120 true = 8 :=
by
  sorry

end NUMINAMATH_GPT_train_crossing_time_correct_l443_44379


namespace NUMINAMATH_GPT_orchard_trees_l443_44395

theorem orchard_trees (x p : ℕ) (h : x + p = 480) (h2 : p = 3 * x) : x = 120 ∧ p = 360 :=
by
  sorry

end NUMINAMATH_GPT_orchard_trees_l443_44395


namespace NUMINAMATH_GPT_max_min_value_l443_44392

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x - 2)

theorem max_min_value (M m : ℝ) (hM : M = f 3) (hm : m = f 4) : (m * m) / M = 8 / 3 := by
  sorry

end NUMINAMATH_GPT_max_min_value_l443_44392


namespace NUMINAMATH_GPT_tape_mounting_cost_correct_l443_44314

-- Define the given conditions as Lean definitions
def os_overhead_cost : ℝ := 1.07
def cost_per_millisecond : ℝ := 0.023
def total_cost : ℝ := 40.92
def runtime_seconds : ℝ := 1.5

-- Define the required target cost for mounting a data tape
def cost_of_data_tape : ℝ := 5.35

-- Prove that the cost of mounting a data tape is correct given the conditions
theorem tape_mounting_cost_correct :
  let computer_time_cost := cost_per_millisecond * (runtime_seconds * 1000)
  let total_cost_computed := os_overhead_cost + computer_time_cost
  cost_of_data_tape = total_cost - total_cost_computed := by
{
  sorry
}

end NUMINAMATH_GPT_tape_mounting_cost_correct_l443_44314


namespace NUMINAMATH_GPT_required_moles_H2SO4_l443_44365

-- Definitions for the problem
def moles_NaCl := 2
def moles_H2SO4_needed := 2
def moles_HCl_produced := 2
def moles_NaHSO4_produced := 2

-- Condition representing stoichiometry of the reaction
axiom reaction_stoichiometry : ∀ (moles_NaCl moles_H2SO4 moles_HCl moles_NaHSO4 : ℕ), 
  moles_NaCl = moles_HCl ∧ moles_HCl = moles_NaHSO4 → moles_NaCl = moles_H2SO4

-- Proof statement we want to establish
theorem required_moles_H2SO4 : 
  ∃ (moles_H2SO4 : ℕ), moles_H2SO4 = 2 ∧ ∀ (moles_NaCl : ℕ), moles_NaCl = 2 → moles_H2SO4_needed = 2 := by
  sorry

end NUMINAMATH_GPT_required_moles_H2SO4_l443_44365


namespace NUMINAMATH_GPT_sin_alpha_plus_2beta_l443_44339

theorem sin_alpha_plus_2beta
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (hcosalpha_plus_beta : Real.cos (α + β) = -5 / 13)
  (h sinbeta : Real.sin β = 3 / 5) :
  Real.sin (α + 2 * β) = 33 / 65 :=
  sorry

end NUMINAMATH_GPT_sin_alpha_plus_2beta_l443_44339


namespace NUMINAMATH_GPT_average_consecutive_from_c_l443_44330

variable (a : ℕ) (c : ℕ)

-- Condition: c is the average of seven consecutive integers starting from a
axiom h1 : c = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) / 7

-- Target statement: Prove the average of seven consecutive integers starting from c is a + 6
theorem average_consecutive_from_c : 
  (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7 = a + 6 :=
by
  sorry

end NUMINAMATH_GPT_average_consecutive_from_c_l443_44330


namespace NUMINAMATH_GPT_henry_earnings_correct_l443_44399

-- Define constants for the amounts earned per task
def earn_per_lawn : Nat := 5
def earn_per_leaves : Nat := 10
def earn_per_driveway : Nat := 15

-- Define constants for the number of tasks he actually managed to do
def lawns_mowed : Nat := 5
def leaves_raked : Nat := 3
def driveways_shoveled : Nat := 2

-- Define the expected total earnings calculation
def expected_earnings : Nat :=
  (lawns_mowed * earn_per_lawn) +
  (leaves_raked * earn_per_leaves) +
  (driveways_shoveled * earn_per_driveway)

-- State the theorem that the total earnings are 85 dollars.
theorem henry_earnings_correct : expected_earnings = 85 :=
by
  sorry

end NUMINAMATH_GPT_henry_earnings_correct_l443_44399


namespace NUMINAMATH_GPT_pictures_at_the_museum_l443_44394

theorem pictures_at_the_museum (M : ℕ) (zoo_pics : ℕ) (deleted_pics : ℕ) (remaining_pics : ℕ)
    (h1 : zoo_pics = 15) (h2 : deleted_pics = 31) (h3 : remaining_pics = 2) (h4 : zoo_pics + M = deleted_pics + remaining_pics) :
    M = 18 := 
sorry

end NUMINAMATH_GPT_pictures_at_the_museum_l443_44394


namespace NUMINAMATH_GPT_equilateral_triangle_AB_length_l443_44355

noncomputable def Q := 2
noncomputable def R := 3
noncomputable def S := 4

theorem equilateral_triangle_AB_length :
  ∀ (AB BC CA : ℝ), 
  AB = BC ∧ BC = CA ∧ (∃ P : ℝ × ℝ, (Q = 2) ∧ (R = 3) ∧ (S = 4)) →
  AB = 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_equilateral_triangle_AB_length_l443_44355


namespace NUMINAMATH_GPT_fraction_phone_numbers_9_ending_even_l443_44305

def isValidPhoneNumber (n : Nat) : Bool :=
  n / 10^6 != 0 && n / 10^6 != 1 && n / 10^6 != 2

def isValidEndEven (n : Nat) : Bool :=
  let lastDigit := n % 10
  lastDigit == 0 || lastDigit == 2 || lastDigit == 4 || lastDigit == 6 || lastDigit == 8

def countValidPhoneNumbers : Nat :=
  7 * 10^6

def countValidStarting9EndingEven : Nat :=
  5 * 10^5

theorem fraction_phone_numbers_9_ending_even :
  (countValidStarting9EndingEven : ℚ) / (countValidPhoneNumbers : ℚ) = 1 / 14 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_phone_numbers_9_ending_even_l443_44305


namespace NUMINAMATH_GPT_map_point_to_result_l443_44332

def f (x y : ℝ) : ℝ × ℝ := (x + y, x - y)

theorem map_point_to_result :
  f 2 0 = (2, 2) :=
by
  unfold f
  simp

end NUMINAMATH_GPT_map_point_to_result_l443_44332


namespace NUMINAMATH_GPT_parity_of_f_min_value_of_f_min_value_of_f_l443_44385

open Real

def f (a x : ℝ) := x^2 + abs (x - a) + 1

theorem parity_of_f (a : ℝ) :
  (∀ x : ℝ, f 0 x = f 0 (-x)) ∧ (∀ x : ℝ, f a x ≠ f a (-x) ∧ f a x ≠ -f a x) ↔ a = 0 :=
by sorry

theorem min_value_of_f (a : ℝ) (h : a ≤ -1/2) : 
  ∀ x : ℝ, x ≥ a → f a x ≥ f a (-1/2) :=
by sorry

theorem min_value_of_f' (a : ℝ) (h : -1/2 < a) : 
  ∀ x : ℝ, x ≥ a → f a x ≥ f a a :=
by sorry

end NUMINAMATH_GPT_parity_of_f_min_value_of_f_min_value_of_f_l443_44385


namespace NUMINAMATH_GPT_cathy_wins_probability_l443_44372

theorem cathy_wins_probability : 
  -- Definitions of the problem conditions
  let p_win := (1 : ℚ) / 6
  let p_not_win := (5 : ℚ) / 6
  -- The probability that Cathy wins
  (p_not_win ^ 2 * p_win) / (1 - p_not_win ^ 3) = 25 / 91 :=
by
  sorry

end NUMINAMATH_GPT_cathy_wins_probability_l443_44372


namespace NUMINAMATH_GPT_inverse_of_217_mod_397_l443_44354

theorem inverse_of_217_mod_397 :
  ∃ a : ℤ, 0 ≤ a ∧ a < 397 ∧ 217 * a % 397 = 1 :=
sorry

end NUMINAMATH_GPT_inverse_of_217_mod_397_l443_44354


namespace NUMINAMATH_GPT_solve_f_eq_x_l443_44393

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry

axiom f_inv_domain : ∀ (x : ℝ), 0 ≤ x ∧ x < 1 → 1 ≤ f_inv x ∧ f_inv x < 2
axiom f_inv_range : ∀ (x : ℝ), 2 < x ∧ x ≤ 4 → 0 ≤ f_inv x ∧ f_inv x < 1
-- Assumption that f is invertible on [0, 3]
axiom f_inv_exists : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → ∃ y : ℝ, f y = x

theorem solve_f_eq_x : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x = x → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_f_eq_x_l443_44393


namespace NUMINAMATH_GPT_remainder_mul_mod_l443_44383

theorem remainder_mul_mod (a b n : ℕ) (h₁ : a ≡ 3 [MOD n]) (h₂ : b ≡ 150 [MOD n]) (n_eq : n = 400) : 
  (a * b) % n = 50 :=
by 
  sorry

end NUMINAMATH_GPT_remainder_mul_mod_l443_44383


namespace NUMINAMATH_GPT_market_value_of_stock_l443_44374

def face_value : ℝ := 100
def dividend_per_share : ℝ := 0.10 * face_value
def yield : ℝ := 0.08

theorem market_value_of_stock : (dividend_per_share / yield) = 125 := by
  -- Proof not required
  sorry

end NUMINAMATH_GPT_market_value_of_stock_l443_44374


namespace NUMINAMATH_GPT_harmonic_mean_2_3_6_l443_44382

def harmonic_mean (a b c : ℕ) : ℚ := 3 / ((1 / a) + (1 / b) + (1 / c))

theorem harmonic_mean_2_3_6 : harmonic_mean 2 3 6 = 3 := 
by
  sorry

end NUMINAMATH_GPT_harmonic_mean_2_3_6_l443_44382


namespace NUMINAMATH_GPT_probability_of_specific_choice_l443_44368

-- Define the sets of subjects
inductive Subject
| Chinese
| Mathematics
| ForeignLanguage
| Physics
| History
| PoliticalScience
| Geography
| Chemistry
| Biology

-- Define the conditions of the examination mode "3+1+2"
def threeSubjects := [Subject.Chinese, Subject.Mathematics, Subject.ForeignLanguage]
def oneSubject := [Subject.Physics, Subject.History]
def twoSubjects := [Subject.PoliticalScience, Subject.Geography, Subject.Chemistry, Subject.Biology]

-- Calculate the total number of ways to choose one subject from Physics or History and two subjects from PoliticalScience, Geography, Chemistry, and Biology
def totalWays : Nat := 2 * Nat.choose 4 2  -- 2 choices for "1" part, and C(4, 2) ways for "2" part

-- Calculate the probability that a candidate will choose Political Science, History, and Geography
def favorableOutcome := 1  -- Only one specific combination counts

theorem probability_of_specific_choice :
  let total_ways := totalWays
  let specific_combination := favorableOutcome
  (specific_combination : ℚ) / total_ways = 1 / 12 :=
by
  let total_ways := totalWays
  let specific_combination := favorableOutcome
  show (specific_combination : ℚ) / total_ways = 1 / 12
  sorry

end NUMINAMATH_GPT_probability_of_specific_choice_l443_44368


namespace NUMINAMATH_GPT_average_income_of_other_40_customers_l443_44364

theorem average_income_of_other_40_customers
    (avg_income_50 : ℝ)
    (num_50 : ℕ)
    (avg_income_10 : ℝ)
    (num_10 : ℕ)
    (total_num : ℕ)
    (remaining_num : ℕ)
    (total_income_50 : ℝ)
    (total_income_10 : ℝ)
    (total_income_40 : ℝ)
    (avg_income_40 : ℝ) 
    (hyp_avg_income_50 : avg_income_50 = 45000)
    (hyp_num_50 : num_50 = 50)
    (hyp_avg_income_10 : avg_income_10 = 55000)
    (hyp_num_10 : num_10 = 10)
    (hyp_total_num : total_num = 50)
    (hyp_remaining_num : remaining_num = 40)
    (hyp_total_income_50 : total_income_50 = 2250000)
    (hyp_total_income_10 : total_income_10 = 550000)
    (hyp_total_income_40 : total_income_40 = 1700000)
    (hyp_avg_income_40 : avg_income_40 = total_income_40 / remaining_num) :
  avg_income_40 = 42500 :=
  by
    sorry

end NUMINAMATH_GPT_average_income_of_other_40_customers_l443_44364


namespace NUMINAMATH_GPT_profit_ratio_l443_44308

theorem profit_ratio (p_investment q_investment : ℝ) (h₁ : p_investment = 50000) (h₂ : q_investment = 66666.67) :
  (1 / q_investment) = (3 / 4 * 1 / p_investment) :=
by
  sorry

end NUMINAMATH_GPT_profit_ratio_l443_44308


namespace NUMINAMATH_GPT_kibble_recommendation_difference_l443_44351

theorem kibble_recommendation_difference :
  (0.2 * 1000 : ℝ) < (0.3 * 1000) ∧ ((0.3 * 1000) - (0.2 * 1000)) = 100 :=
by
  sorry

end NUMINAMATH_GPT_kibble_recommendation_difference_l443_44351


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l443_44362

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_angle : b / a = Real.sqrt 3 / 3) :
    let e := Real.sqrt (1 + (b / a)^2)
    e = 2 * Real.sqrt 3 / 3 := 
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l443_44362


namespace NUMINAMATH_GPT_sum_of_repeating_decimals_l443_44306

-- Definitions based on the conditions
def x := 0.6666666666666666 -- Lean may not directly support \(0.\overline{6}\) notation
def y := 0.7777777777777777 -- Lean may not directly support \(0.\overline{7}\) notation

-- Translate those to the correct fractional forms
def x_as_fraction := (2 : ℚ) / 3
def y_as_fraction := (7 : ℚ) / 9

-- The main statement to prove
theorem sum_of_repeating_decimals : x_as_fraction + y_as_fraction = 13 / 9 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_sum_of_repeating_decimals_l443_44306


namespace NUMINAMATH_GPT_f_decreasing_on_negative_interval_and_min_value_l443_44367

noncomputable def f : ℝ → ℝ := sorry

-- Define the conditions
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

def minimum_value (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, f x ≥ m ∧ ∃ x0, f x0 = m

-- Given the conditions
variables (condition1 : even_function f)
          (condition2 : increasing_on_interval f 3 7)
          (condition3 : minimum_value f 2)

-- Prove that f is decreasing on [-7,-3] and minimum value is 2
theorem f_decreasing_on_negative_interval_and_min_value :
  ∀ x y, -7 ≤ x → x ≤ y → y ≤ -3 → f y ≤ f x ∧ minimum_value f 2 :=
sorry

end NUMINAMATH_GPT_f_decreasing_on_negative_interval_and_min_value_l443_44367


namespace NUMINAMATH_GPT_function_no_extrema_k_equals_one_l443_44341

theorem function_no_extrema_k_equals_one (k : ℝ) (h : ∀ x : ℝ, ¬ ∃ m, (k - 1) * x^2 - 4 * x + 5 - k = m) : k = 1 :=
sorry

end NUMINAMATH_GPT_function_no_extrema_k_equals_one_l443_44341


namespace NUMINAMATH_GPT_alternating_binomial_sum_l443_44349

open BigOperators Finset

theorem alternating_binomial_sum :
  ∑ k in range 34, (-1 : ℤ)^k * (Nat.choose 99 (3 * k)) = -1 := by
  sorry

end NUMINAMATH_GPT_alternating_binomial_sum_l443_44349


namespace NUMINAMATH_GPT_sally_baseball_cards_l443_44300

theorem sally_baseball_cards (initial_cards torn_cards purchased_cards : ℕ) 
    (h_initial : initial_cards = 39)
    (h_torn : torn_cards = 9)
    (h_purchased : purchased_cards = 24) :
    initial_cards - torn_cards - purchased_cards = 6 := by
  sorry

end NUMINAMATH_GPT_sally_baseball_cards_l443_44300


namespace NUMINAMATH_GPT_sum_seven_consecutive_integers_l443_44344

theorem sum_seven_consecutive_integers (m : ℕ) :
  m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5) + (m + 6) = 7 * m + 21 :=
by
  -- Sorry to skip the actual proof steps.
  sorry

end NUMINAMATH_GPT_sum_seven_consecutive_integers_l443_44344


namespace NUMINAMATH_GPT_polynomial_square_l443_44343

theorem polynomial_square (x : ℝ) : x^4 + 2*x^3 - 2*x^2 - 4*x - 5 = y^2 → x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_GPT_polynomial_square_l443_44343


namespace NUMINAMATH_GPT_stratified_sampling_expected_females_l443_44380

noncomputable def sample_size := 14
noncomputable def total_athletes := 44 + 33
noncomputable def female_athletes := 33
noncomputable def stratified_sample := (female_athletes * sample_size) / total_athletes

theorem stratified_sampling_expected_females :
  stratified_sample = 6 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_expected_females_l443_44380


namespace NUMINAMATH_GPT_initial_children_on_bus_l443_44316

-- Define the conditions
variables (x : ℕ)

-- Define the problem statement
theorem initial_children_on_bus (h : x + 7 = 25) : x = 18 :=
sorry

end NUMINAMATH_GPT_initial_children_on_bus_l443_44316


namespace NUMINAMATH_GPT_parallel_line_eq_l443_44307

theorem parallel_line_eq (x y : ℝ) (c : ℝ) :
  (∀ x y, x - 2 * y - 2 = 0 → x - 2 * y + c = 0) ∧ (x = 1 ∧ y = 0) → c = -1 :=
by
  sorry

end NUMINAMATH_GPT_parallel_line_eq_l443_44307


namespace NUMINAMATH_GPT_solution_of_phi_l443_44342

theorem solution_of_phi 
    (φ : ℝ) 
    (H : ∃ k : ℤ, 2 * (π / 6) + φ = k * π) :
    φ = - (π / 3) := 
sorry

end NUMINAMATH_GPT_solution_of_phi_l443_44342


namespace NUMINAMATH_GPT_greatest_divisor_of_arithmetic_sequence_sum_l443_44350

theorem greatest_divisor_of_arithmetic_sequence_sum (x c : ℕ) (hx : x > 0) (hc : c > 0) :
  ∃ k, (∀ (S : ℕ), S = 6 * (2 * x + 11 * c) → k ∣ S) ∧ k = 6 :=
by
  sorry

end NUMINAMATH_GPT_greatest_divisor_of_arithmetic_sequence_sum_l443_44350


namespace NUMINAMATH_GPT_smallest_number_among_options_l443_44312

noncomputable def binary_to_decimal (n : ℕ) : ℕ :=
  match n with
  | 111111 => 63
  | _ => 0

noncomputable def base_six_to_decimal (n : ℕ) : ℕ :=
  match n with
  | 210 => 2 * 6^2 + 1 * 6
  | _ => 0

noncomputable def base_nine_to_decimal (n : ℕ) : ℕ :=
  match n with
  | 85 => 8 * 9 + 5
  | _ => 0

theorem smallest_number_among_options :
  min 75 (min (binary_to_decimal 111111) (min (base_six_to_decimal 210) (base_nine_to_decimal 85))) = binary_to_decimal 111111 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_number_among_options_l443_44312


namespace NUMINAMATH_GPT_factor_roots_l443_44347

theorem factor_roots (t : ℝ) : (x - t) ∣ (8 * x^2 + 18 * x - 5) ↔ (t = 1 / 4 ∨ t = -5) :=
by
  sorry

end NUMINAMATH_GPT_factor_roots_l443_44347


namespace NUMINAMATH_GPT_twentieth_fisherman_caught_l443_44325

-- Definitions based on conditions
def fishermen_count : ℕ := 20
def total_fish_caught : ℕ := 10000
def fish_per_nineteen_fishermen : ℕ := 400
def nineteen_count : ℕ := 19

-- Calculation based on the problem conditions
def total_fish_by_nineteen : ℕ := nineteen_count * fish_per_nineteen_fishermen

-- Prove the number of fish caught by the twentieth fisherman
theorem twentieth_fisherman_caught : 
  total_fish_caught - total_fish_by_nineteen = 2400 := 
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_twentieth_fisherman_caught_l443_44325


namespace NUMINAMATH_GPT_minimum_number_of_kings_maximum_number_of_non_attacking_kings_l443_44345

-- Definitions for the chessboard and king placement problem

-- Problem (a): Minimum number of kings covering the board
def minimum_kings_covering_board (board_size : Nat) : Nat :=
  sorry

theorem minimum_number_of_kings (h : 6 = board_size) :
  minimum_kings_covering_board 6 = 4 := 
  sorry

-- Problem (b): Maximum number of non-attacking kings
def maximum_non_attacking_kings (board_size : Nat) : Nat :=
  sorry

theorem maximum_number_of_non_attacking_kings (h : 6 = board_size) :
  maximum_non_attacking_kings 6 = 9 :=
  sorry

end NUMINAMATH_GPT_minimum_number_of_kings_maximum_number_of_non_attacking_kings_l443_44345


namespace NUMINAMATH_GPT_find_functions_l443_44320

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def f' (a b x : ℝ) : ℝ := 2 * a * x + b

theorem find_functions
  (a b c : ℝ)
  (h_a : a ≠ 0)
  (h_1 : ∀ x : ℝ, |x| ≤ 1 → |f a b c x| ≤ 1)
  (h_2 : ∃ x₀ : ℝ, |x₀| ≤ 1 ∧ ∀ x : ℝ, |x| ≤ 1 → |f' a b x₀| ≥ |f' a b x| )
  (K : ℝ)
  (h_3 : ∃ x₀ : ℝ, |x₀| ≤ 1 ∧ |f' a b x₀| = K) :
  (f a b c = fun x ↦ 2 * x^2 - 1) ∨ (f a b c = fun x ↦ -2 * x^2 + 1) ∧ K = 4 := 
sorry

end NUMINAMATH_GPT_find_functions_l443_44320


namespace NUMINAMATH_GPT_net_error_24x_l443_44311

theorem net_error_24x (x : ℕ) : 
  let penny_value := 1
  let nickel_value := 5
  let dime_value := 10
  let quarter_value := 25
  let error_pennies := (nickel_value - penny_value) * x
  let error_nickels := (dime_value - nickel_value) * x
  let error_dimes := (quarter_value - dime_value) * x
  let total_error := error_pennies + error_nickels + error_dimes
  total_error = 24 * x := 
by 
  sorry

end NUMINAMATH_GPT_net_error_24x_l443_44311


namespace NUMINAMATH_GPT_cubic_coefficient_relationship_l443_44338

theorem cubic_coefficient_relationship (a b c p q r : ℝ)
    (h1 : ∀ s1 s2 s3: ℝ, s1 + s2 + s3 = -a ∧ s1 * s2 + s2 * s3 + s3 * s1 = b ∧ s1 * s2 * s3 = -c)
    (h2 : ∀ s1 s2 s3: ℝ, s1^2 + s2^2 + s3^2 = -p ∧ s1^2 * s2^2 + s2^2 * s3^2 + s3^2 * s1^2 = q ∧ s1^2 * s2^2 * s3^2 = r) :
    p = a^2 - 2 * b ∧ q = b^2 + 2 * a * c ∧ r = c^2 :=
by
  sorry

end NUMINAMATH_GPT_cubic_coefficient_relationship_l443_44338


namespace NUMINAMATH_GPT_find_center_radius_sum_l443_44319

theorem find_center_radius_sum :
    let x := x
    let y := y
    let a := 2
    let b := 3
    let r := 2 * Real.sqrt 6
    (x^2 - 4 * x + y^2 - 6 * y = 11) →
    (a + b + r = 5 + 2 * Real.sqrt 6) :=
by
  intros x y a b r
  sorry

end NUMINAMATH_GPT_find_center_radius_sum_l443_44319


namespace NUMINAMATH_GPT_staircase_toothpicks_l443_44337

theorem staircase_toothpicks :
  ∀ (T : ℕ → ℕ), 
  (T 4 = 28) →
  (∀ n : ℕ, T (n + 1) = T n + (12 + 3 * (n - 3))) →
  T 6 - T 4 = 33 :=
by
  intros T T4_step H_increase
  -- proof goes here
  sorry

end NUMINAMATH_GPT_staircase_toothpicks_l443_44337


namespace NUMINAMATH_GPT_complement_P_relative_to_U_l443_44375

variable (U : Set ℝ) (P : Set ℝ)

theorem complement_P_relative_to_U (hU : U = Set.univ) (hP : P = {x : ℝ | x < 1}) : 
  U \ P = {x : ℝ | x ≥ 1} := by
  sorry

end NUMINAMATH_GPT_complement_P_relative_to_U_l443_44375


namespace NUMINAMATH_GPT_remaining_perimeter_l443_44327

-- Definitions based on conditions
noncomputable def GH : ℝ := 2
noncomputable def HI : ℝ := 2
noncomputable def GI : ℝ := Real.sqrt (GH^2 + HI^2)
noncomputable def side_JKL : ℝ := 5
noncomputable def JI : ℝ := side_JKL - GH
noncomputable def IK : ℝ := side_JKL - HI
noncomputable def JK : ℝ := side_JKL

-- Problem statement in Lean 4
theorem remaining_perimeter :
  JI + IK + JK = 11 :=
by
  sorry

end NUMINAMATH_GPT_remaining_perimeter_l443_44327


namespace NUMINAMATH_GPT_least_integer_gt_sqrt_450_l443_44386

theorem least_integer_gt_sqrt_450 : 
  ∀ n : ℕ, (∃ m : ℕ, m * m ≤ 450 ∧ 450 < (m + 1) * (m + 1)) → n = 22 :=
by
  sorry

end NUMINAMATH_GPT_least_integer_gt_sqrt_450_l443_44386


namespace NUMINAMATH_GPT_sum_digits_in_possibilities_l443_44348

noncomputable def sum_of_digits (a b c d : ℕ) : ℕ :=
  a + b + c + d

theorem sum_digits_in_possibilities :
  ∃ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 
  (sum_of_digits a b c d = 10 ∨ sum_of_digits a b c d = 18 ∨ sum_of_digits a b c d = 19) := sorry

end NUMINAMATH_GPT_sum_digits_in_possibilities_l443_44348


namespace NUMINAMATH_GPT_mary_prevents_pat_l443_44381

noncomputable def smallest_initial_integer (N: ℕ) : Prop :=
  N > 2017 ∧ 
  ∀ x, ∃ n: ℕ, 
  (x = N + n * 2018 → x % 2018 ≠ 0 ∧
   (2017 * x + 2) % 2018 ≠ 0 ∧
   (2017 * x + 2021) % 2018 ≠ 0)

theorem mary_prevents_pat (N : ℕ) : smallest_initial_integer N → N = 2022 :=
sorry

end NUMINAMATH_GPT_mary_prevents_pat_l443_44381


namespace NUMINAMATH_GPT_skylar_current_age_l443_44302

theorem skylar_current_age (started_age : ℕ) (annual_donation : ℕ) (total_donation : ℕ) (h1 : started_age = 17) (h2 : annual_donation = 8000) (h3 : total_donation = 440000) : 
  (started_age + total_donation / annual_donation = 72) :=
by
  sorry

end NUMINAMATH_GPT_skylar_current_age_l443_44302


namespace NUMINAMATH_GPT_find_tangent_parallel_to_x_axis_l443_44378

theorem find_tangent_parallel_to_x_axis :
  ∃ (x y : ℝ), y = x^2 - 3 * x ∧ (2 * x - 3 = 0) ∧ (x = 3 / 2) ∧ (y = -9 / 4) := 
by
  sorry

end NUMINAMATH_GPT_find_tangent_parallel_to_x_axis_l443_44378


namespace NUMINAMATH_GPT_carol_additional_cupcakes_l443_44322

-- Define the initial number of cupcakes Carol made
def initial_cupcakes : ℕ := 30

-- Define the number of cupcakes Carol sold
def sold_cupcakes : ℕ := 9

-- Define the total number of cupcakes Carol wanted to have
def total_cupcakes : ℕ := 49

-- Calculate the number of cupcakes Carol had left after selling
def remaining_cupcakes : ℕ := initial_cupcakes - sold_cupcakes

-- The number of additional cupcakes Carol made can be defined and proved as follows:
theorem carol_additional_cupcakes : initial_cupcakes - sold_cupcakes + 28 = total_cupcakes :=
by
  -- left side: initial_cupcakes (30) - sold_cupcakes (9) + additional_cupcakes (28) = total_cupcakes (49)
  sorry

end NUMINAMATH_GPT_carol_additional_cupcakes_l443_44322
