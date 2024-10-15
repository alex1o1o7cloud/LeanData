import Mathlib

namespace NUMINAMATH_GPT_packet_a_weight_l1336_133695

theorem packet_a_weight (A B C D E : ℕ) :
  A + B + C = 252 →
  A + B + C + D = 320 →
  E = D + 3 →
  B + C + D + E = 316 →
  A = 75 := by
  sorry

end NUMINAMATH_GPT_packet_a_weight_l1336_133695


namespace NUMINAMATH_GPT_circle_symmetry_l1336_133658

theorem circle_symmetry {a : ℝ} (h : a ≠ 0) :
  ∀ {x y : ℝ}, (x^2 + y^2 + 2*a*x - 2*a*y = 0) → (x + y = 0) :=
sorry

end NUMINAMATH_GPT_circle_symmetry_l1336_133658


namespace NUMINAMATH_GPT_total_cost_of_items_l1336_133631

variable (M R F : ℝ)
variable (h1 : 10 * M = 24 * R)
variable (h2 : F = 2 * R)
variable (h3 : F = 21)

theorem total_cost_of_items : 4 * M + 3 * R + 5 * F = 237.3 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_items_l1336_133631


namespace NUMINAMATH_GPT_johns_total_earnings_l1336_133600

noncomputable def total_earnings_per_week (baskets_monday : ℕ) (baskets_thursday : ℕ) (small_crabs_per_basket : ℕ) (large_crabs_per_basket : ℕ) (price_small_crab : ℕ) (price_large_crab : ℕ) : ℕ :=
  let small_crabs := baskets_monday * small_crabs_per_basket
  let large_crabs := baskets_thursday * large_crabs_per_basket
  (small_crabs * price_small_crab) + (large_crabs * price_large_crab)

theorem johns_total_earnings :
  total_earnings_per_week 3 4 4 5 3 5 = 136 :=
by
  sorry

end NUMINAMATH_GPT_johns_total_earnings_l1336_133600


namespace NUMINAMATH_GPT_tan_sum_identity_l1336_133690

theorem tan_sum_identity (α β : ℝ)
  (h1 : Real.tan (α - π / 6) = 3 / 7)
  (h2 : Real.tan (π / 6 + β) = 2 / 5) :
  Real.tan (α + β) = 1 :=
sorry

end NUMINAMATH_GPT_tan_sum_identity_l1336_133690


namespace NUMINAMATH_GPT_original_cost_price_l1336_133691

theorem original_cost_price (selling_price_friend : ℝ) (gain_percent : ℝ) (loss_percent : ℝ) 
  (final_selling_price : ℝ) : 
  final_selling_price = 54000 → gain_percent = 0.2 → loss_percent = 0.1 → 
  selling_price_friend = (1 - loss_percent) * x → final_selling_price = (1 + gain_percent) * selling_price_friend → 
  x = 50000 :=
by 
  sorry

end NUMINAMATH_GPT_original_cost_price_l1336_133691


namespace NUMINAMATH_GPT_remainder_sum_l1336_133621

theorem remainder_sum (x y z : ℕ) (h1 : x % 15 = 11) (h2 : y % 15 = 13) (h3 : z % 15 = 9) :
  ((2 * (x % 15) + (y % 15) + (z % 15)) % 15) = 14 :=
by
  sorry

end NUMINAMATH_GPT_remainder_sum_l1336_133621


namespace NUMINAMATH_GPT_age_sum_l1336_133685

theorem age_sum (my_age : ℕ) (mother_age : ℕ) (h1 : mother_age = 3 * my_age) (h2 : my_age = 10) :
  my_age + mother_age = 40 :=
by 
  -- proof omitted
  sorry

end NUMINAMATH_GPT_age_sum_l1336_133685


namespace NUMINAMATH_GPT_find_value_of_p_l1336_133667

variable (x y : ℝ)

/-- Given that the hyperbola has the equation x^2 / 4 - y^2 / 12 = 1
    and the eccentricity e = 2, and that the parabola x = 2 * p * y^2 has its focus at (e, 0), 
    prove that the value of the real number p is 1/8. -/
theorem find_value_of_p :
  (∃ (p : ℝ), 
    (∀ (x y : ℝ), x^2 / 4 - y^2 / 12 = 1) ∧ 
    (∀ (x y : ℝ), x = 2 * p * y^2) ∧
    (2 = 2)) →
    ∃ (p : ℝ), p = 1/8 :=
by 
  sorry

end NUMINAMATH_GPT_find_value_of_p_l1336_133667


namespace NUMINAMATH_GPT_smallest_integer_neither_prime_nor_square_with_prime_factors_ge_60_l1336_133601

def is_not_prime (n : ℕ) := ¬ Prime n
def is_not_square (n : ℕ) := ∀ m : ℕ, m * m ≠ n
def no_prime_factors_less_than (n k : ℕ) := ∀ p : ℕ, Prime p → p < k → ¬ p ∣ n
def smallest_integer_prop (n : ℕ) := is_not_prime n ∧ is_not_square n ∧ no_prime_factors_less_than n 60

theorem smallest_integer_neither_prime_nor_square_with_prime_factors_ge_60 : ∃ n : ℕ, smallest_integer_prop n ∧ n = 4087 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_neither_prime_nor_square_with_prime_factors_ge_60_l1336_133601


namespace NUMINAMATH_GPT_no_n_such_that_n_times_s_is_20222022_l1336_133654

-- Define the sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main theorem
theorem no_n_such_that_n_times_s_is_20222022 :
  ∀ n : ℕ, n * sum_of_digits n ≠ 20222022 :=
by
  sorry

end NUMINAMATH_GPT_no_n_such_that_n_times_s_is_20222022_l1336_133654


namespace NUMINAMATH_GPT_last_four_digits_5_2011_l1336_133696

theorem last_four_digits_5_2011 :
  (5^2011 % 10000) = 8125 := by
  sorry

end NUMINAMATH_GPT_last_four_digits_5_2011_l1336_133696


namespace NUMINAMATH_GPT_correct_fraction_statement_l1336_133677

theorem correct_fraction_statement (x : ℝ) :
  (∀ a b : ℝ, (-a) / (-b) = a / b) ∧
  (¬ (∀ a : ℝ, a / 0 = 0)) ∧
  (∀ a b : ℝ, b ≠ 0 → (a * b) / (c * b) = a / c) → 
  ((∃ (a b : ℝ), a = 0 → a / b = 0) ∧ 
   (∀ (a b : ℝ), (a * k) / (b * k) = a / b) ∧ 
   (∀ (a b : ℝ), (-a) / (-b) = a / b) ∧ 
   (x < 1 → (|2 - x| + x) / 2 ≠ 0) 
  -> (∀ (a b : ℝ), (-a) / (-b) = a / b)) :=
by sorry

end NUMINAMATH_GPT_correct_fraction_statement_l1336_133677


namespace NUMINAMATH_GPT_xavier_yvonne_not_zelda_prob_l1336_133637

def Px : ℚ := 1 / 4
def Py : ℚ := 2 / 3
def Pz : ℚ := 5 / 8

theorem xavier_yvonne_not_zelda_prob : 
  (Px * Py * (1 - Pz) = 1 / 16) :=
by 
  sorry

end NUMINAMATH_GPT_xavier_yvonne_not_zelda_prob_l1336_133637


namespace NUMINAMATH_GPT_min_abs_sum_half_l1336_133616

theorem min_abs_sum_half :
  ∀ (f g : ℝ → ℝ),
  (∀ x, f x = Real.sin (x + Real.pi / 3)) →
  (∀ x, g x = Real.sin (2 * x + Real.pi / 3)) →
  (∀ x1 x2 : ℝ, g x1 * g x2 = -1 ∧ x1 ≠ x2 → abs ((x1 + x2) / 2) = Real.pi / 6) := by
-- Definitions and conditions are set, now we can state the theorem.
  sorry

end NUMINAMATH_GPT_min_abs_sum_half_l1336_133616


namespace NUMINAMATH_GPT_foil_covered_prism_width_l1336_133612

theorem foil_covered_prism_width 
    (l w h : ℕ) 
    (h_w_eq_2l : w = 2 * l)
    (h_w_eq_2h : w = 2 * h)
    (h_volume : l * w * h = 128) 
    (h_foiled_width : q = w + 2) :
  q = 10 := 
sorry

end NUMINAMATH_GPT_foil_covered_prism_width_l1336_133612


namespace NUMINAMATH_GPT_second_caterer_cheaper_l1336_133636

theorem second_caterer_cheaper (x : ℕ) :
  (∀ n : ℕ, n < x → 150 + 18 * n ≤ 250 + 15 * n) ∧ (150 + 18 * x > 250 + 15 * x) ↔ x = 34 :=
by sorry

end NUMINAMATH_GPT_second_caterer_cheaper_l1336_133636


namespace NUMINAMATH_GPT_largest_5_digit_congruent_15_mod_24_l1336_133624

theorem largest_5_digit_congruent_15_mod_24 : ∃ x, 10000 ≤ x ∧ x < 100000 ∧ x % 24 = 15 ∧ x = 99999 := by
  sorry

end NUMINAMATH_GPT_largest_5_digit_congruent_15_mod_24_l1336_133624


namespace NUMINAMATH_GPT_jake_weight_loss_l1336_133641

variable {J K L : Nat}

theorem jake_weight_loss
  (h1 : J + K = 290)
  (h2 : J = 196)
  (h3 : J - L = 2 * K) : L = 8 :=
by
  sorry

end NUMINAMATH_GPT_jake_weight_loss_l1336_133641


namespace NUMINAMATH_GPT_limit_sum_infinite_geometric_series_l1336_133650

noncomputable def infinite_geometric_series_limit (a_1 q : ℝ) :=
  if |q| < 1 then (a_1 / (1 - q)) else 0

theorem limit_sum_infinite_geometric_series :
  infinite_geometric_series_limit 1 (1 / 3) = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_limit_sum_infinite_geometric_series_l1336_133650


namespace NUMINAMATH_GPT_workers_l1336_133642

theorem workers (N C : ℕ) (h1 : N * C = 300000) (h2 : N * (C + 50) = 315000) : N = 300 :=
by
  sorry

end NUMINAMATH_GPT_workers_l1336_133642


namespace NUMINAMATH_GPT_h_at_2_l1336_133697

noncomputable def h (x : ℝ) : ℝ := 
(x + 2) * (x - 1) * (x + 4) * (x - 3) - x^2

theorem h_at_2 : 
  h (-2) = -4 ∧ h (1) = -1 ∧ h (-4) = -16 ∧ h (3) = -9 → h (2) = -28 := 
by
  intro H
  sorry

end NUMINAMATH_GPT_h_at_2_l1336_133697


namespace NUMINAMATH_GPT_probability_of_being_closer_to_origin_l1336_133645

noncomputable def probability_closer_to_origin 
  (rect : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2})
  (origin : ℝ × ℝ := (0, 0))
  (point : ℝ × ℝ := (4, 2))
  : ℚ :=
1/3

theorem probability_of_being_closer_to_origin :
  probability_closer_to_origin = 1/3 :=
by sorry

end NUMINAMATH_GPT_probability_of_being_closer_to_origin_l1336_133645


namespace NUMINAMATH_GPT_scholarship_awards_l1336_133668

theorem scholarship_awards (x : ℕ) (h : 10000 * x + 2000 * (28 - x) = 80000) : x = 3 ∧ (28 - x) = 25 :=
by {
  sorry
}

end NUMINAMATH_GPT_scholarship_awards_l1336_133668


namespace NUMINAMATH_GPT_original_weight_of_apple_box_l1336_133675

theorem original_weight_of_apple_box:
  ∀ (x : ℕ), (3 * x - 12 = x) → x = 6 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_original_weight_of_apple_box_l1336_133675


namespace NUMINAMATH_GPT_simplify_expression_l1336_133689

theorem simplify_expression :
  (Real.sqrt 2 * 2 ^ (1 / 2 : ℝ) + 18 / 3 * 3 - 8 ^ (3 / 2 : ℝ)) = (20 - 16 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1336_133689


namespace NUMINAMATH_GPT_total_cost_l1336_133686

def cost_burger := 5
def cost_sandwich := 4
def cost_smoothie := 4
def count_smoothies := 2

theorem total_cost :
  cost_burger + cost_sandwich + count_smoothies * cost_smoothie = 17 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_l1336_133686


namespace NUMINAMATH_GPT_sum_le_six_l1336_133622

theorem sum_le_six (a b : ℤ) (h1 : a ≠ -1) (h2 : b ≠ -1) 
    (h3 : ∃ (r s : ℤ), r * s = a + b ∧ r + s = ab) : a + b ≤ 6 :=
sorry

end NUMINAMATH_GPT_sum_le_six_l1336_133622


namespace NUMINAMATH_GPT_sum_of_squares_eq_23456_l1336_133682

theorem sum_of_squares_eq_23456 (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 1728 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 23456 :=
by sorry

end NUMINAMATH_GPT_sum_of_squares_eq_23456_l1336_133682


namespace NUMINAMATH_GPT_cameron_list_count_l1336_133698

theorem cameron_list_count :
  let lower := 100
  let upper := 1000
  let step := 20
  let n_min := lower / step
  let n_max := upper / step
  lower % step = 0 ∧ upper % step = 0 →
  upper ≥ lower →
  n_max - n_min + 1 = 46 :=
by
  sorry

end NUMINAMATH_GPT_cameron_list_count_l1336_133698


namespace NUMINAMATH_GPT_find_y_l1336_133633

-- Hypotheses
variable (x y : ℤ)

-- Given conditions
def condition1 : Prop := x = 4
def condition2 : Prop := x + y = 0

-- The goal is to prove y = -4 given the conditions
theorem find_y (h1 : condition1 x) (h2 : condition2 x y) : y = -4 := by
  sorry

end NUMINAMATH_GPT_find_y_l1336_133633


namespace NUMINAMATH_GPT_eval_64_pow_5_over_6_l1336_133672

theorem eval_64_pow_5_over_6 (h : 64 = 2^6) : 64^(5/6) = 32 := 
by 
  sorry

end NUMINAMATH_GPT_eval_64_pow_5_over_6_l1336_133672


namespace NUMINAMATH_GPT_max_A_min_A_l1336_133620

-- Define the problem and its conditions and question

def A_max (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) : ℕ :=
  let b := B % 10
  10^8 * b + (B - b) / 10

def A_min (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) : ℕ :=
  let b := B % 10
  10^8 * b + (B - b) / 10

theorem max_A (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) :
  A_max B h1 h2 h3 = 999999998 := sorry

theorem min_A (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) :
  A_min B h1 h2 h3 = 122222224 := sorry

end NUMINAMATH_GPT_max_A_min_A_l1336_133620


namespace NUMINAMATH_GPT_robin_initial_gum_is_18_l1336_133625

-- Defining the conditions as given in the problem
def given_gum : ℝ := 44
def total_gum : ℝ := 62

-- Statement to prove that the initial number of pieces of gum Robin had is 18
theorem robin_initial_gum_is_18 : total_gum - given_gum = 18 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_robin_initial_gum_is_18_l1336_133625


namespace NUMINAMATH_GPT_cylinder_height_relation_l1336_133626

theorem cylinder_height_relation (r1 r2 h1 h2 V1 V2 : ℝ) 
  (h_volumes_equal : V1 = V2)
  (h_r2_gt_r1 : r2 = 1.1 * r1)
  (h_volume_first : V1 = π * r1^2 * h1)
  (h_volume_second : V2 = π * r2^2 * h2) : 
  h1 = 1.21 * h2 :=
by 
  sorry

end NUMINAMATH_GPT_cylinder_height_relation_l1336_133626


namespace NUMINAMATH_GPT_problem_statement_l1336_133673

-- Definitions of propositions p and q
def p : Prop := ∃ x : ℝ, Real.tan x = 1
def q : Prop := ∀ x : ℝ, x^2 > 0

-- The proof problem
theorem problem_statement : ¬ (¬ p ∧ ¬ q) :=
by 
  -- sorry here indicates that actual proof is omitted
  sorry

end NUMINAMATH_GPT_problem_statement_l1336_133673


namespace NUMINAMATH_GPT_arithmetic_sequence_term_l1336_133657

theorem arithmetic_sequence_term (a : ℕ → ℤ) (d : ℤ) (n : ℕ) :
  a 5 = 33 ∧ a 45 = 153 ∧ (∀ n, a n = a 1 + (n - 1) * d) ∧ a n = 201 → n = 61 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_l1336_133657


namespace NUMINAMATH_GPT_largest_divisible_by_digits_sum_l1336_133678

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem largest_divisible_by_digits_sum : ∃ n, n < 900 ∧ n % digits_sum n = 0 ∧ ∀ m, m < 900 ∧ m % digits_sum m = 0 → m ≤ 888 :=
by
  sorry

end NUMINAMATH_GPT_largest_divisible_by_digits_sum_l1336_133678


namespace NUMINAMATH_GPT_chess_team_boys_count_l1336_133649

theorem chess_team_boys_count (J S B : ℕ) 
  (h1 : J + S + B = 32) 
  (h2 : (1 / 3 : ℚ) * J + (1 / 2 : ℚ) * S + B = 18) : 
  B = 4 :=
by
  sorry

end NUMINAMATH_GPT_chess_team_boys_count_l1336_133649


namespace NUMINAMATH_GPT_linear_relationship_correct_profit_160_max_profit_l1336_133651

-- Define the conditions for the problem
def data_points : List (ℝ × ℝ) := [(3.5, 280), (5.5, 120)]

-- The linear function relationship between y and x
def linear_relationship (x : ℝ) : ℝ := -80 * x + 560

-- The equation for profit, given selling price and sales quantity
def profit (x : ℝ) : ℝ := (x - 3) * (linear_relationship x) - 80

-- Prove the relationship y = -80x + 560 from given data points
theorem linear_relationship_correct : 
  ∀ (x y : ℝ), (x, y) ∈ data_points → y = linear_relationship x :=
sorry

-- Prove the selling price x = 4 results in a profit of $160 per day
theorem profit_160 (x : ℝ) (h : profit x = 160) : x = 4 :=
sorry

-- Prove the maximum profit and corresponding selling price
theorem max_profit : 
  ∃ x : ℝ, ∃ w : ℝ, 3.5 ≤ x ∧ x ≤ 5.5 ∧ profit x = w ∧ ∀ y, 3.5 ≤ y ∧ y ≤ 5.5 → profit y ≤ w ∧ w = 240 ∧ x = 5 :=
sorry

end NUMINAMATH_GPT_linear_relationship_correct_profit_160_max_profit_l1336_133651


namespace NUMINAMATH_GPT_max_possible_N_l1336_133628

theorem max_possible_N (cities roads N : ℕ) (h1 : cities = 1000) (h2 : roads = 2017) (h3 : N = roads - (cities - 1 + 7 - 1)) :
  N = 1009 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_possible_N_l1336_133628


namespace NUMINAMATH_GPT_arithmetic_example_l1336_133646

theorem arithmetic_example : 2546 + 240 / 60 - 346 = 2204 := by
  sorry

end NUMINAMATH_GPT_arithmetic_example_l1336_133646


namespace NUMINAMATH_GPT_sum_first_23_natural_numbers_l1336_133610

theorem sum_first_23_natural_numbers :
  (23 * (23 + 1)) / 2 = 276 := 
by
  sorry

end NUMINAMATH_GPT_sum_first_23_natural_numbers_l1336_133610


namespace NUMINAMATH_GPT_ratio_total_length_to_perimeter_l1336_133643

noncomputable def length_initial : ℝ := 25
noncomputable def width_initial : ℝ := 15
noncomputable def extension : ℝ := 10
noncomputable def length_total : ℝ := length_initial + extension
noncomputable def perimeter_new : ℝ := 2 * (length_total + width_initial)
noncomputable def ratio : ℝ := length_total / perimeter_new

theorem ratio_total_length_to_perimeter : ratio = 35 / 100 := by
  sorry

end NUMINAMATH_GPT_ratio_total_length_to_perimeter_l1336_133643


namespace NUMINAMATH_GPT_eric_less_than_ben_l1336_133647

variables (E B J : ℕ)

theorem eric_less_than_ben
  (hJ : J = 26)
  (hB : B = J - 9)
  (total_money : E + B + J = 50) :
  B - E = 10 :=
sorry

end NUMINAMATH_GPT_eric_less_than_ben_l1336_133647


namespace NUMINAMATH_GPT_gcd_72_120_168_l1336_133692

theorem gcd_72_120_168 : Nat.gcd (Nat.gcd 72 120) 168 = 24 :=
by
  sorry

end NUMINAMATH_GPT_gcd_72_120_168_l1336_133692


namespace NUMINAMATH_GPT_proportion_solution_l1336_133676

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 4.5 / (7 / 3)) : x = 0.3888888889 :=
by
  sorry

end NUMINAMATH_GPT_proportion_solution_l1336_133676


namespace NUMINAMATH_GPT_smallest_possible_value_of_N_l1336_133681

theorem smallest_possible_value_of_N :
  ∀ (a b c d e f : ℕ), a + b + c + d + e + f = 3015 → (0 < a) → (0 < b) → (0 < c) → (0 < d) → (0 < e) → (0 < f) →
  (∃ N : ℕ, N = max (max (max (max (a + b) (b + c)) (c + d)) (d + e)) (e + f) ∧ N = 604) := 
by
  sorry

end NUMINAMATH_GPT_smallest_possible_value_of_N_l1336_133681


namespace NUMINAMATH_GPT_length_of_train_l1336_133608

noncomputable def speed_kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

noncomputable def total_distance (speed_m_s : ℝ) (time_s : ℝ) : ℝ :=
  speed_m_s * time_s

noncomputable def train_length (total_distance : ℝ) (bridge_length : ℝ) : ℝ :=
  total_distance - bridge_length

theorem length_of_train
  (speed_kmh : ℝ)
  (time_s : ℝ)
  (bridge_length : ℝ)
  (speed_in_kmh : speed_kmh = 45)
  (time_in_seconds : time_s = 30)
  (length_of_bridge : bridge_length = 220.03) :
  train_length (total_distance (speed_kmh_to_ms speed_kmh) time_s) bridge_length = 154.97 :=
by
  sorry

end NUMINAMATH_GPT_length_of_train_l1336_133608


namespace NUMINAMATH_GPT_initial_tickets_count_l1336_133615

def spent_tickets : ℕ := 5
def additional_tickets : ℕ := 10
def current_tickets : ℕ := 16

theorem initial_tickets_count (initial_tickets : ℕ) :
  initial_tickets - spent_tickets + additional_tickets = current_tickets ↔ initial_tickets = 11 :=
by
  sorry

end NUMINAMATH_GPT_initial_tickets_count_l1336_133615


namespace NUMINAMATH_GPT_solution_set_of_cx2_minus_bx_plus_a_lt_0_is_correct_l1336_133661

theorem solution_set_of_cx2_minus_bx_plus_a_lt_0_is_correct (a b c : ℝ) :
  (∀ x : ℝ, ax^2 + bx + c ≤ 0 ↔ x ≤ -1 ∨ x ≥ 3) →
  b = -2*a →
  c = -3*a →
  a < 0 →
  (∀ x : ℝ, cx^2 - bx + a < 0 ↔ -1/3 < x ∧ x < 1) := 
by 
  intro h_root_set h_b_eq h_c_eq h_a_lt_0 
  sorry

end NUMINAMATH_GPT_solution_set_of_cx2_minus_bx_plus_a_lt_0_is_correct_l1336_133661


namespace NUMINAMATH_GPT_green_ball_probability_l1336_133688

def prob_green_ball : ℚ :=
  let prob_container := (1 : ℚ) / 3
  let prob_green_I := (4 : ℚ) / 12
  let prob_green_II := (5 : ℚ) / 8
  let prob_green_III := (4 : ℚ) / 8
  prob_container * prob_green_I + prob_container * prob_green_II + prob_container * prob_green_III

theorem green_ball_probability :
  prob_green_ball = 35 / 72 :=
by
  -- Proof steps are omitted as "sorry" is used to skip the proof.
  sorry

end NUMINAMATH_GPT_green_ball_probability_l1336_133688


namespace NUMINAMATH_GPT_find_sin_minus_cos_l1336_133669

variable {a : ℝ}
variable {α : ℝ}

def point_of_angle (a : ℝ) (h : a < 0) := (3 * a, -4 * a)

theorem find_sin_minus_cos (a : ℝ) (h : a < 0) (ha : point_of_angle a h = (3 * a, -4 * a)) (sinα : ℝ) (cosα : ℝ) :
  sinα = 4 / 5 → cosα = -3 / 5 → sinα - cosα = 7 / 5 :=
by sorry

end NUMINAMATH_GPT_find_sin_minus_cos_l1336_133669


namespace NUMINAMATH_GPT_sammy_mistakes_l1336_133611

def bryan_score : ℕ := 20
def jen_score : ℕ := bryan_score + 10
def sammy_score : ℕ := jen_score - 2
def total_points : ℕ := 35
def mistakes : ℕ := total_points - sammy_score

theorem sammy_mistakes : mistakes = 7 := by
  sorry

end NUMINAMATH_GPT_sammy_mistakes_l1336_133611


namespace NUMINAMATH_GPT_smallest_constant_obtuse_triangle_l1336_133619

theorem smallest_constant_obtuse_triangle (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  (a^2 > b^2 + c^2) → (b^2 + c^2) / (a^2) ≥ 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_constant_obtuse_triangle_l1336_133619


namespace NUMINAMATH_GPT_limes_remaining_l1336_133603

-- Definitions based on conditions
def initial_limes : ℕ := 9
def limes_given_to_Sara : ℕ := 4

-- Theorem to prove
theorem limes_remaining : initial_limes - limes_given_to_Sara = 5 :=
by
  -- Sorry keyword to skip the actual proof
  sorry

end NUMINAMATH_GPT_limes_remaining_l1336_133603


namespace NUMINAMATH_GPT_ratio_of_ages_l1336_133660

theorem ratio_of_ages (joe_age_now james_age_now : ℕ) (h1 : joe_age_now = james_age_now + 10)
  (h2 : 2 * (joe_age_now + 8) = 3 * (james_age_now + 8)) : 
  (james_age_now + 8) / (joe_age_now + 8) = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l1336_133660


namespace NUMINAMATH_GPT_number_exceeds_fraction_l1336_133627

theorem number_exceeds_fraction (x : ℝ) (h : x = (3/8) * x + 15) : x = 24 :=
sorry

end NUMINAMATH_GPT_number_exceeds_fraction_l1336_133627


namespace NUMINAMATH_GPT_stuart_initial_marbles_l1336_133679

theorem stuart_initial_marbles (B S : ℝ) (h1 : B = 60) (h2 : 0.40 * B = 24) (h3 : S + 24 = 80) : S = 56 :=
by
  sorry

end NUMINAMATH_GPT_stuart_initial_marbles_l1336_133679


namespace NUMINAMATH_GPT_smallest_a_exists_l1336_133694

theorem smallest_a_exists : ∃ a b c : ℕ, 
                          (∀ α β : ℝ, 
                          (α > 0 ∧ α ≤ 1 / 1000) ∧ 
                          (β > 0 ∧ β ≤ 1 / 1000) ∧ 
                          (α + β = -b / a) ∧ 
                          (α * β = c / a) ∧ 
                          (b * b - 4 * a * c > 0)) ∧ 
                          (a = 1001000) := sorry

end NUMINAMATH_GPT_smallest_a_exists_l1336_133694


namespace NUMINAMATH_GPT_find_base_l1336_133653

theorem find_base (b : ℝ) (h : 2.134 * b^3 < 21000) : b ≤ 21 :=
by
  have h1 : b < (21000 / 2.134) ^ (1 / 3) := sorry
  have h2 : (21000 / 2.134) ^ (1 / 3) < 21.5 := sorry
  have h3 : b ≤ 21 := sorry
  exact h3

end NUMINAMATH_GPT_find_base_l1336_133653


namespace NUMINAMATH_GPT_find_d_value_l1336_133684

theorem find_d_value (a b : ℚ) (d : ℚ) (h1 : a = 2) (h2 : b = 11) 
  (h3 : ∀ x, 2 * x^2 + 11 * x + d = 0 ↔ x = (-11 + Real.sqrt 15) / 4 ∨ x = (-11 - Real.sqrt 15) / 4) : 
  d = 53 / 4 :=
sorry

end NUMINAMATH_GPT_find_d_value_l1336_133684


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l1336_133693

variable {R : Type} [LinearOrderedField R]

theorem quadratic_has_two_distinct_real_roots (c d : R) :
  ∀ x : R, (x + c) * (x + d) - (2 * x + c + d) = 0 → 
  (x + c)^2 + 4 > 0 :=
by
  intros x h
  -- Proof (skipped)
  sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l1336_133693


namespace NUMINAMATH_GPT_parallelogram_base_l1336_133670

theorem parallelogram_base
  (Area Height Base : ℕ)
  (h_area : Area = 120)
  (h_height : Height = 10)
  (h_area_eq : Area = Base * Height) :
  Base = 12 :=
by
  /- 
    We assume the conditions:
    1. Area = 120
    2. Height = 10
    3. Area = Base * Height 
    Then, we need to prove that Base = 12.
  -/
  sorry

end NUMINAMATH_GPT_parallelogram_base_l1336_133670


namespace NUMINAMATH_GPT_Ms_Thompsons_statement_contrapositive_of_Ms_Thompsons_statement_l1336_133638

-- Define P and Q as propositions where P indicates submission of all required essays and Q indicates failing the course.
variable (P Q : Prop)

-- Ms. Thompson's statement translated to logical form.
theorem Ms_Thompsons_statement : ¬P → Q := sorry

-- The goal is to prove that if a student did not fail the course, then they submitted all the required essays.
theorem contrapositive_of_Ms_Thompsons_statement (h : ¬Q) : P := 
by {
  -- Proof will go here
  sorry 
}

end NUMINAMATH_GPT_Ms_Thompsons_statement_contrapositive_of_Ms_Thompsons_statement_l1336_133638


namespace NUMINAMATH_GPT_project_presentation_period_length_l1336_133605

theorem project_presentation_period_length
  (students : ℕ)
  (presentation_time_per_student : ℕ)
  (number_of_periods : ℕ)
  (total_students : students = 32)
  (time_per_student : presentation_time_per_student = 5)
  (periods_needed : number_of_periods = 4) :
  (32 * 5) / 4 = 40 := 
by {
  sorry
}

end NUMINAMATH_GPT_project_presentation_period_length_l1336_133605


namespace NUMINAMATH_GPT_minimum_triangle_area_l1336_133671

theorem minimum_triangle_area :
  ∀ (m n : ℝ), (m > 0) ∧ (n > 0) ∧ (1 / m + 2 / n = 1) → (1 / 2 * m * n) = 4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_triangle_area_l1336_133671


namespace NUMINAMATH_GPT_hats_per_yard_of_velvet_l1336_133656

theorem hats_per_yard_of_velvet
  (H : ℕ)
  (velvet_for_cloak : ℕ := 3)
  (total_velvet : ℕ := 21)
  (number_of_cloaks : ℕ := 6)
  (number_of_hats : ℕ := 12)
  (yards_for_6_cloaks : ℕ := number_of_cloaks * velvet_for_cloak)
  (remaining_yards_for_hats : ℕ := total_velvet - yards_for_6_cloaks)
  (hats_per_remaining_yard : ℕ := number_of_hats / remaining_yards_for_hats)
  : H = hats_per_remaining_yard :=
  by
  sorry

end NUMINAMATH_GPT_hats_per_yard_of_velvet_l1336_133656


namespace NUMINAMATH_GPT_average_first_three_numbers_l1336_133606

theorem average_first_three_numbers (A B C D : ℝ) 
  (hA : A = 33) 
  (hD : D = 18)
  (hBCD : (B + C + D) / 3 = 15) : 
  (A + B + C) / 3 = 20 := 
by 
  sorry

end NUMINAMATH_GPT_average_first_three_numbers_l1336_133606


namespace NUMINAMATH_GPT_arccos_one_eq_zero_l1336_133659

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end NUMINAMATH_GPT_arccos_one_eq_zero_l1336_133659


namespace NUMINAMATH_GPT_barn_painting_total_area_l1336_133674

theorem barn_painting_total_area :
  let width := 12
  let length := 15
  let height := 5
  let divider_width := 12
  let divider_height := 5

  let external_wall_area := 2 * (width * height + length * height)
  let dividing_wall_area := 2 * (divider_width * divider_height)
  let ceiling_area := width * length
  let total_area := 2 * external_wall_area + dividing_wall_area + ceiling_area

  total_area = 840 := by
    sorry

end NUMINAMATH_GPT_barn_painting_total_area_l1336_133674


namespace NUMINAMATH_GPT_total_profit_l1336_133680

theorem total_profit (C_profit : ℝ) (x : ℝ) (h1 : 4 * x = 48000) : 12 * x = 144000 :=
by
  sorry

end NUMINAMATH_GPT_total_profit_l1336_133680


namespace NUMINAMATH_GPT_expression_evaluation_l1336_133699

theorem expression_evaluation (a : ℕ) (h : a = 1580) : 
  2 * a - ((2 * a - 3) / (a + 1) - (a + 1) / (2 - 2 * a) - (a^2 + 3) / 2) * ((a^3 + 1) / (a^2 - a)) + 2 / a = 2 := 
sorry

end NUMINAMATH_GPT_expression_evaluation_l1336_133699


namespace NUMINAMATH_GPT_sum_arith_seq_elems_l1336_133664

noncomputable def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem sum_arith_seq_elems (a d : ℝ) 
  (h : arithmetic_seq a d 2 + arithmetic_seq a d 5 + arithmetic_seq a d 8 + arithmetic_seq a d 11 = 48) :
  arithmetic_seq a d 6 + arithmetic_seq a d 7 = 24 := 
by 
  sorry

end NUMINAMATH_GPT_sum_arith_seq_elems_l1336_133664


namespace NUMINAMATH_GPT_find_value_l1336_133635

theorem find_value (m n : ℤ) (h : 2 * m + n - 2 = 0) : 2 * m + n + 1 = 3 :=
by { sorry }

end NUMINAMATH_GPT_find_value_l1336_133635


namespace NUMINAMATH_GPT_smallest_divisible_by_2022_l1336_133655

theorem smallest_divisible_by_2022 (n : ℕ) (N : ℕ) :
  (N = 20230110) ∧ (∃ k : ℕ, N = 2023 * 10^n + k) ∧ N % 2022 = 0 → 
  ∀ M: ℕ, (∃ m : ℕ, M = 2023 * 10^n + m) ∧ M % 2022 = 0 → N ≤ M :=
sorry

end NUMINAMATH_GPT_smallest_divisible_by_2022_l1336_133655


namespace NUMINAMATH_GPT_solve_equation_l1336_133648

theorem solve_equation (x y : ℝ) (h : 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2):
  y = -x - 2 ∨ y = -2 * x + 1 :=
sorry


end NUMINAMATH_GPT_solve_equation_l1336_133648


namespace NUMINAMATH_GPT_quadratic_eq_roots_are_coeffs_l1336_133632

theorem quadratic_eq_roots_are_coeffs :
  ∃ (a b : ℝ), (a = r_1) → (b = r_2) →
  (r_1 + r_2 = -a) → (r_1 * r_2 = b) →
  r_1 = 1 ∧ r_2 = -2 ∧ (x^2 + x - 2 = 0):=
by
  sorry

end NUMINAMATH_GPT_quadratic_eq_roots_are_coeffs_l1336_133632


namespace NUMINAMATH_GPT_negative_integer_reciprocal_of_d_l1336_133644

def a : ℚ := 3
def b : ℚ := |1 / 3|
def c : ℚ := -2
def d : ℚ := -1 / 2

theorem negative_integer_reciprocal_of_d (h : d ≠ 0) : ∃ k : ℤ, (d⁻¹ : ℚ) = ↑k ∧ k < 0 :=
by
  sorry

end NUMINAMATH_GPT_negative_integer_reciprocal_of_d_l1336_133644


namespace NUMINAMATH_GPT_son_age_l1336_133604

theorem son_age (M S : ℕ) (h1 : M = S + 24) (h2 : M + 2 = 2 * (S + 2)) : S = 22 :=
by
  sorry

end NUMINAMATH_GPT_son_age_l1336_133604


namespace NUMINAMATH_GPT_initial_bottle_caps_l1336_133665

variable (initial_caps added_caps total_caps : ℕ)

theorem initial_bottle_caps 
  (h1 : added_caps = 7) 
  (h2 : total_caps = 14) 
  (h3 : total_caps = initial_caps + added_caps) : 
  initial_caps = 7 := 
by 
  sorry

end NUMINAMATH_GPT_initial_bottle_caps_l1336_133665


namespace NUMINAMATH_GPT_no_solution_in_positive_integers_l1336_133617

theorem no_solution_in_positive_integers
    (x y : ℕ)
    (h : x > 0 ∧ y > 0) :
    x^2006 - 4 * y^2006 - 2006 ≠ 4 * y^2007 + 2007 * y :=
by
  sorry

end NUMINAMATH_GPT_no_solution_in_positive_integers_l1336_133617


namespace NUMINAMATH_GPT_common_root_values_l1336_133618

def has_common_root (p x : ℝ) : Prop :=
  (x^2 - (p+1)*x + (p+1) = 0) ∧ (2*x^2 + (p-2)*x - p - 7 = 0)

theorem common_root_values :
  (has_common_root 3 2) ∧ (has_common_root (-3/2) (-1)) :=
by {
  sorry
}

end NUMINAMATH_GPT_common_root_values_l1336_133618


namespace NUMINAMATH_GPT_Tim_has_52_photos_l1336_133666

theorem Tim_has_52_photos (T : ℕ) (Paul : ℕ) (Total : ℕ) (Tom : ℕ) : 
  (Paul = T + 10) → (Total = Tom + T + Paul) → (Tom = 38) → (Total = 152) → T = 52 :=
by
  intros hPaul hTotal hTom hTotalVal
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_Tim_has_52_photos_l1336_133666


namespace NUMINAMATH_GPT_max_value_f_l1336_133687

noncomputable def f (a x : ℝ) : ℝ := a^2 * Real.sin (2 * x) + (a - 2) * Real.cos (2 * x)

theorem max_value_f (a : ℝ) (h : a < 0)
  (symm : ∀ x, f a (x - π / 4) = f a (-x - π / 4)) :
  ∃ x, f a x = 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_max_value_f_l1336_133687


namespace NUMINAMATH_GPT_prime_p_geq_7_div_240_l1336_133629

theorem prime_p_geq_7_div_240 (p : ℕ) (hp : Nat.Prime p) (hge7 : p ≥ 7) : 240 ∣ p^4 - 1 := 
sorry

end NUMINAMATH_GPT_prime_p_geq_7_div_240_l1336_133629


namespace NUMINAMATH_GPT_range_of_x_l1336_133614

noncomputable def f (x : ℝ) : ℝ := x * (2^x - 1 / 2^x)

theorem range_of_x (x : ℝ) (h : f (x - 1) > f x) : x < 1 / 2 :=
by sorry

end NUMINAMATH_GPT_range_of_x_l1336_133614


namespace NUMINAMATH_GPT_determine_suit_cost_l1336_133652

def cost_of_suit (J B V : ℕ) : Prop :=
  (J + B + V = 150)

theorem determine_suit_cost
  (J B V : ℕ)
  (h1 : J = B + V)
  (h2 : J + 2 * B = 175)
  (h3 : B + 2 * V = 100) :
  cost_of_suit J B V :=
by
  sorry

end NUMINAMATH_GPT_determine_suit_cost_l1336_133652


namespace NUMINAMATH_GPT_oleg_can_find_adjacent_cells_divisible_by_4_l1336_133607

theorem oleg_can_find_adjacent_cells_divisible_by_4 :
  ∀ (grid : Fin 22 → Fin 22 → ℕ),
  (∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 22 * 22) →
  ∃ i j k l, ((i = k ∧ (j = l + 1 ∨ j = l - 1)) ∨ ((i = k + 1 ∨ i = k - 1) ∧ j = l)) ∧ ((grid i j + grid k l) % 4 = 0) :=
by
  sorry

end NUMINAMATH_GPT_oleg_can_find_adjacent_cells_divisible_by_4_l1336_133607


namespace NUMINAMATH_GPT_paving_stones_needed_l1336_133640

-- Definition for the dimensions of the paving stone and the courtyard
def paving_stone_length : ℝ := 2.5
def paving_stone_width : ℝ := 2
def courtyard_length : ℝ := 30
def courtyard_width : ℝ := 16.5

-- Compute areas
def paving_stone_area : ℝ := paving_stone_length * paving_stone_width
def courtyard_area : ℝ := courtyard_length * courtyard_width

-- The theorem to prove that the number of paving stones needed is 99
theorem paving_stones_needed :
  (courtyard_area / paving_stone_area) = 99 :=
by
  sorry

end NUMINAMATH_GPT_paving_stones_needed_l1336_133640


namespace NUMINAMATH_GPT_minimum_distance_l1336_133683

noncomputable def distance (M Q : ℝ × ℝ) : ℝ :=
  ( (M.1 - Q.1) ^ 2 + (M.2 - Q.2) ^ 2 ) ^ (1 / 2)

theorem minimum_distance (M : ℝ × ℝ) :
  ∃ Q : ℝ × ℝ, ( (Q.1 - 1) ^ 2 + Q.2 ^ 2 = 1 ) ∧ distance M Q = 1 :=
sorry

end NUMINAMATH_GPT_minimum_distance_l1336_133683


namespace NUMINAMATH_GPT_tangent_line_equation_l1336_133602

-- Define the function
def f (x : ℝ) : ℝ := x^2

-- Define the point of tangency
def x0 : ℝ := 2

-- Define the value of function at the point of tangency
def y0 : ℝ := f x0

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x

-- State the theorem
theorem tangent_line_equation : ∃ (m b : ℝ), m = f' x0 ∧ b = y0 - m * x0 ∧ ∀ x, (y = m * x + b) ↔ (x = 2 → y = f x - f' x0 * (x - 2)) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_equation_l1336_133602


namespace NUMINAMATH_GPT_cos_13pi_over_4_eq_neg_one_div_sqrt_two_l1336_133662

noncomputable def cos_13pi_over_4 : Real :=
  Real.cos (13 * Real.pi / 4)

theorem cos_13pi_over_4_eq_neg_one_div_sqrt_two : 
  cos_13pi_over_4 = -1 / Real.sqrt 2 := by 
  sorry

end NUMINAMATH_GPT_cos_13pi_over_4_eq_neg_one_div_sqrt_two_l1336_133662


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_for_positive_quadratic_l1336_133630

variables {a b c : ℝ}

theorem sufficient_not_necessary_condition_for_positive_quadratic 
  (ha : a > 0)
  (hb : b^2 - 4 * a * c < 0) :
  (∀ x : ℝ, a * x ^ 2 + b * x + c > 0) 
  ∧ ¬ (∀ x : ℝ, ∃ a b c : ℝ, a > 0 ∧ b^2 - 4 * a * c ≥ 0 ∧ (a * x ^ 2 + b * x + c > 0)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_for_positive_quadratic_l1336_133630


namespace NUMINAMATH_GPT_minimum_balls_l1336_133613

/-- Given that tennis balls are stored in big boxes containing 25 balls each 
    and small boxes containing 20 balls each, and the least number of balls 
    that can be left unboxed is 5, prove that the least number of 
    freshly manufactured balls is 105.
-/
theorem minimum_balls (B S : ℕ) : 
  ∃ (n : ℕ), 25 * B + 20 * S = n ∧ n % 25 = 5 ∧ n % 20 = 5 ∧ n = 105 := 
sorry

end NUMINAMATH_GPT_minimum_balls_l1336_133613


namespace NUMINAMATH_GPT_find_sum_a_b_l1336_133663

theorem find_sum_a_b (a b : ℝ) 
  (h : (a^2 + 4 * a + 6) * (2 * b^2 - 4 * b + 7) ≤ 10) : a + 2 * b = 0 := 
sorry

end NUMINAMATH_GPT_find_sum_a_b_l1336_133663


namespace NUMINAMATH_GPT_knights_in_exchange_l1336_133609

noncomputable def count_knights (total_islanders : ℕ) (odd_statements : ℕ) (even_statements : ℕ) : ℕ :=
if total_islanders % 2 = 0 ∧ odd_statements = total_islanders ∧ even_statements = total_islanders then
    total_islanders / 2
else
    0

theorem knights_in_exchange : count_knights 30 30 30 = 15 :=
by
    -- proof part will go here but is not required.
    sorry

end NUMINAMATH_GPT_knights_in_exchange_l1336_133609


namespace NUMINAMATH_GPT_max_product_min_quotient_l1336_133634

theorem max_product_min_quotient :
  let nums := [-5, -3, -1, 2, 4]
  let a := max (max (-5 * -3) (-5 * -1)) (max (-3 * -1) (max (2 * 4) (max (2 * -1) (4 * -1))))
  let b := min (min (4 / -1) (2 / -3)) (min (2 / -5) (min (4 / -3) (-5 / -3)))
  a = 15 ∧ b = -4 → a / b = -15 / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_product_min_quotient_l1336_133634


namespace NUMINAMATH_GPT_smallest_number_with_2020_divisors_l1336_133639

theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, 
  (∀ n : ℕ, (∃ (p : ℕ) (α : ℕ), n = p^α) → 
  ∃ (p1 p2 p3 p4 : ℕ) (α1 α2 α3 α4 : ℕ), 
  n = p1^α1 * p2^α2 * p3^α3 * p4^α4 ∧ 
  (α1 + 1) * (α2 + 1) * (α3 + 1) * (α4 + 1) = 2020) → 
  n = 2^100 * 3^4 * 5 * 7 :=
sorry

end NUMINAMATH_GPT_smallest_number_with_2020_divisors_l1336_133639


namespace NUMINAMATH_GPT_vertical_distance_from_top_to_bottom_l1336_133623

-- Conditions
def ring_thickness : ℕ := 2
def largest_ring_diameter : ℕ := 18
def smallest_ring_diameter : ℕ := 4

-- Additional definitions based on the problem context
def count_rings : ℕ := (largest_ring_diameter - smallest_ring_diameter) / ring_thickness + 1
def inner_diameters_sum : ℕ := count_rings * (largest_ring_diameter - ring_thickness + smallest_ring_diameter) / 2
def vertical_distance : ℕ := inner_diameters_sum + 2 * ring_thickness

-- The problem statement to prove
theorem vertical_distance_from_top_to_bottom :
  vertical_distance = 76 := by
  sorry

end NUMINAMATH_GPT_vertical_distance_from_top_to_bottom_l1336_133623
