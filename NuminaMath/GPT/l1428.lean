import Mathlib

namespace cos_pi_over_4_minus_alpha_l1428_142862

theorem cos_pi_over_4_minus_alpha (α : ℝ) (h : Real.sin (Real.pi / 4 + α) = 2 / 3) :
  Real.cos (Real.pi / 4 - α) = 2 / 3 := 
by
  sorry

end cos_pi_over_4_minus_alpha_l1428_142862


namespace fred_games_last_year_proof_l1428_142849

def fred_games_last_year (this_year: ℕ) (diff: ℕ) : ℕ := this_year + diff

theorem fred_games_last_year_proof : 
  ∀ (this_year: ℕ) (diff: ℕ),
  this_year = 25 → 
  diff = 11 →
  fred_games_last_year this_year diff = 36 := 
by 
  intros this_year diff h_this_year h_diff
  rw [h_this_year, h_diff]
  sorry

end fred_games_last_year_proof_l1428_142849


namespace quadrilateral_area_l1428_142818

theorem quadrilateral_area (a b x : ℝ)
  (h1: ∀ (y z : ℝ), y^2 + z^2 = a^2 ∧ (x + y)^2 + (x + z)^2 = b^2)
  (hx_perp: ∀ (p q : ℝ), x * q = 0 ∧ x * p = 0) :
  S = (1 / 4) * |b^2 - a^2| :=
by
  sorry

end quadrilateral_area_l1428_142818


namespace at_least_one_not_less_than_two_l1428_142876

theorem at_least_one_not_less_than_two (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x + 1/y) ≥ 2 ∨ (y + 1/z) ≥ 2 ∨ (z + 1/x) ≥ 2 :=
by
  sorry

end at_least_one_not_less_than_two_l1428_142876


namespace sum_first_four_terms_eq_12_l1428_142822

noncomputable def a : ℕ → ℤ := sorry -- An arithmetic sequence aₙ

-- Given conditions
axiom h1 : a 2 = 4
axiom h2 : a 1 + a 5 = 4 * a 3 - 4

theorem sum_first_four_terms_eq_12 : (a 1 + a 2 + a 3 + a 4) = 12 := 
by {
  sorry
}

end sum_first_four_terms_eq_12_l1428_142822


namespace quoted_value_of_stock_l1428_142880

theorem quoted_value_of_stock (D Y Q : ℝ) (h1 : D = 8) (h2 : Y = 10) (h3 : Y = (D / Q) * 100) : Q = 80 :=
by 
  -- Insert proof here
  sorry

end quoted_value_of_stock_l1428_142880


namespace dots_not_visible_on_3_dice_l1428_142813

theorem dots_not_visible_on_3_dice :
  let total_dots := 18 * 21 / 6
  let visible_dots := 1 + 2 + 2 + 3 + 5 + 4 + 5 + 6
  let hidden_dots := total_dots - visible_dots
  hidden_dots = 35 := 
by 
  let total_dots := 18 * 21 / 6
  let visible_dots := 1 + 2 + 2 + 3 + 5 + 4 + 5 + 6
  let hidden_dots := total_dots - visible_dots
  show total_dots - visible_dots = 35
  sorry

end dots_not_visible_on_3_dice_l1428_142813


namespace p_iff_q_l1428_142825

def f (x a : ℝ) := x * (x - a) * (x - 2)

def p (a : ℝ) := 0 < a ∧ a < 2

def q (a : ℝ) := 
  let f' x := 3 * x^2 - 2 * (a + 2) * x + 2 * a
  f' a < 0

theorem p_iff_q (a : ℝ) : (p a) ↔ (q a) := by
  sorry

end p_iff_q_l1428_142825


namespace sum_ratio_l1428_142836

variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}
variable {d : ℕ}

axiom arithmetic_sequence : ∀ n, a_n n = a_n 1 + (n - 1) * d
axiom sum_of_first_n_terms : ∀ n, S_n n = n * (a_n 1 + a_n n) / 2
axiom condition_a4 : a_n 4 = 2 * (a_n 2 + a_n 3)
axiom non_zero_difference : d ≠ 0

theorem sum_ratio : S_n 7 / S_n 4 = 7 / 4 := 
by
  sorry

end sum_ratio_l1428_142836


namespace chess_group_players_l1428_142827

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 91) : n = 14 :=
sorry

end chess_group_players_l1428_142827


namespace tens_digit_of_72_pow_25_l1428_142886

theorem tens_digit_of_72_pow_25 : (72^25 % 100) / 10 = 3 := 
by
  sorry

end tens_digit_of_72_pow_25_l1428_142886


namespace permutations_containing_substring_l1428_142842

open Nat

/-- Prove that the number of permutations of the string "000011112222" that contain the substring "2020" is equal to 3575. -/
theorem permutations_containing_substring :
  let total_permutations := factorial 8 / (factorial 2 * factorial 4 * factorial 2)
  let num_positions := 9
  let non_overlap_count := total_permutations * num_positions
  let overlap_subtract := 7 * (factorial 6 / (factorial 1 * factorial 4 * factorial 1))
  let add_back := 5 * (factorial 4 / factorial 4)
  non_overlap_count - overlap_subtract + add_back = 3575 := 
by
  let total_permutations := factorial 8 / (factorial 2 * factorial 4 * factorial 2)
  let num_positions := 9
  let non_overlap_count := total_permutations * num_positions
  let overlap_subtract := 7 * (factorial 6 / (factorial 1 * factorial 4 * factorial 1))
  let add_back := 5 * (factorial 4 / factorial 4)
  have h: non_overlap_count - overlap_subtract + add_back = 3575 := by sorry
  exact h

end permutations_containing_substring_l1428_142842


namespace part1_part2_l1428_142808

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l1428_142808


namespace average_additional_minutes_per_day_l1428_142844

def daily_differences : List ℤ := [20, 5, -5, 0, 15, -10, 10]

theorem average_additional_minutes_per_day :
  (List.sum daily_differences / daily_differences.length) = 5 := by
  sorry

end average_additional_minutes_per_day_l1428_142844


namespace value_of_k_l1428_142850

theorem value_of_k (x y k : ℝ) (h1 : 4 * x - 3 * y = k) (h2 : 2 * x + 3 * y = 5) (h3 : x = y) : k = 1 :=
sorry

end value_of_k_l1428_142850


namespace empty_vessel_percentage_l1428_142896

theorem empty_vessel_percentage
  (P : ℝ) -- weight of the paint that completely fills the vessel
  (E : ℝ) -- weight of the empty vessel
  (h1 : 0.5 * (E + P) = E + 0.42857142857142855 * P)
  (h2 : 0.07142857142857145 * P = 0.5 * E):
  (E / (E + P) * 100) = 12.5 :=
by
  sorry

end empty_vessel_percentage_l1428_142896


namespace sin_A_sin_C_eq_3_over_4_triangle_is_equilateral_l1428_142833

variable {α : Type*}

-- Part 1
theorem sin_A_sin_C_eq_3_over_4
  (A B C : Real)
  (a b c : Real)
  (h1 : b ^ 2 = a * c)
  (h2 : (Real.cos (A - C)) + (Real.cos B) = 3 / 2) :
  Real.sin A * Real.sin C = 3 / 4 :=
sorry

-- Part 2
theorem triangle_is_equilateral
  (A B C : Real)
  (a b c : Real)
  (h1 : b ^ 2 = a * c)
  (h2 : (Real.cos (A - C)) + (Real.cos B) = 3 / 2) :
  A = B ∧ B = C :=
sorry

end sin_A_sin_C_eq_3_over_4_triangle_is_equilateral_l1428_142833


namespace distance_between_cities_A_B_l1428_142866

-- Define the problem parameters
def train_1_speed : ℝ := 60 -- km/hr
def train_2_speed : ℝ := 75 -- km/hr
def start_time_train_1 : ℝ := 8 -- 8 a.m.
def start_time_train_2 : ℝ := 9 -- 9 a.m.
def meeting_time : ℝ := 12 -- 12 p.m.

-- Define the times each train travels
def hours_train_1_travelled := meeting_time - start_time_train_1
def hours_train_2_travelled := meeting_time - start_time_train_2

-- Calculate the distances covered by each train
def distance_train_1_cover := train_1_speed * hours_train_1_travelled
def distance_train_2_cover := train_2_speed * hours_train_2_travelled

-- Define the total distance between cities A and B
def distance_AB := distance_train_1_cover + distance_train_2_cover

-- The theorem to be proved
theorem distance_between_cities_A_B : distance_AB = 465 := 
  by
    -- placeholder for the proof
    sorry

end distance_between_cities_A_B_l1428_142866


namespace tom_total_amount_l1428_142820

-- Definitions of the initial conditions
def initial_amount : ℕ := 74
def amount_earned : ℕ := 86

-- Main statement to prove
theorem tom_total_amount : initial_amount + amount_earned = 160 := 
by
  -- sorry added to skip the proof
  sorry

end tom_total_amount_l1428_142820


namespace solve_inequality_l1428_142897

theorem solve_inequality (x : ℝ) : 
  3 * (2 * x - 1) - 2 * (x + 1) ≤ 1 → x ≤ 3 / 2 :=
by
  sorry

end solve_inequality_l1428_142897


namespace option_c_not_equivalent_l1428_142815

theorem option_c_not_equivalent :
  ¬ (785 * 10^(-9) = 7.845 * 10^(-6)) :=
by
  sorry

end option_c_not_equivalent_l1428_142815


namespace k_polygonal_intersects_fermat_l1428_142888

theorem k_polygonal_intersects_fermat (k : ℕ) (n m : ℕ) (h1: k > 2) 
  (h2 : ∃ n m, (k - 2) * n * (n - 1) / 2 + n = 2 ^ (2 ^ m) + 1) : 
  k = 3 ∨ k = 5 :=
  sorry

end k_polygonal_intersects_fermat_l1428_142888


namespace smallest_t_for_circle_l1428_142899

theorem smallest_t_for_circle (t : ℝ) :
  (∀ r θ, 0 ≤ θ ∧ θ ≤ t → r = Real.sin θ) → t ≥ π :=
by sorry

end smallest_t_for_circle_l1428_142899


namespace elysse_bags_per_trip_l1428_142860

-- Definitions from the problem conditions
def total_bags : ℕ := 30
def total_trips : ℕ := 5
def bags_per_trip : ℕ := total_bags / total_trips

def carries_same_amount (elysse_bags brother_bags : ℕ) : Prop := elysse_bags = brother_bags

-- Statement to prove
theorem elysse_bags_per_trip :
  ∀ (elysse_bags brother_bags : ℕ), 
  bags_per_trip = elysse_bags + brother_bags → 
  carries_same_amount elysse_bags brother_bags → 
  elysse_bags = 3 := 
by 
  intros elysse_bags brother_bags h1 h2
  sorry

end elysse_bags_per_trip_l1428_142860


namespace greatest_prime_factor_294_l1428_142810

theorem greatest_prime_factor_294 : ∃ p, Nat.Prime p ∧ p ∣ 294 ∧ ∀ q, Nat.Prime q ∧ q ∣ 294 → q ≤ p := 
by
  let prime_factors := [2, 3, 7]
  have h1 : 294 = 2 * 3 * 7 * 7 := by
    -- Proof of factorization should be inserted here
    sorry

  have h2 : ∀ p, p ∣ 294 → p = 2 ∨ p = 3 ∨ p = 7 := by
    -- Proof of prime factor correctness should be inserted here
    sorry

  use 7
  -- Prove 7 is the greatest prime factor here
  sorry

end greatest_prime_factor_294_l1428_142810


namespace ball_count_difference_l1428_142890

open Nat

theorem ball_count_difference :
  (total_balls = 145) →
  (soccer_balls = 20) →
  (basketballs > soccer_balls) →
  (tennis_balls = 2 * soccer_balls) →
  (baseballs = soccer_balls + 10) →
  (volleyballs = 30) →
  let accounted_balls := soccer_balls + tennis_balls + baseballs + volleyballs
  let basketballs := total_balls - accounted_balls
  (basketballs - soccer_balls = 5) :=
by
  intros
  let tennis_balls := 2 * soccer_balls
  let baseballs := soccer_balls + 10
  let accounted_balls := soccer_balls + tennis_balls + baseballs + volleyballs
  let basketballs := total_balls - accounted_balls
  exact sorry

end ball_count_difference_l1428_142890


namespace floor_T_value_l1428_142861

noncomputable def floor_T : ℝ := 
  let p := (0 : ℝ)
  let q := (0 : ℝ)
  let r := (0 : ℝ)
  let s := (0 : ℝ)
  p + q + r + s

theorem floor_T_value (p q r s : ℝ) (hpq: p^2 + q^2 = 2500) (hrs: r^2 + s^2 = 2500) (hpr: p * r = 1200) (hqs: q * s = 1200) (hpos: p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0) :
  ∃ T : ℝ, T = p + q + r + s ∧ ⌊T⌋ = 140 := 
  by
  sorry

end floor_T_value_l1428_142861


namespace solve_equation_l1428_142838

theorem solve_equation (x : ℝ) : x^2 = 5 * x → x = 0 ∨ x = 5 := 
by
  sorry

end solve_equation_l1428_142838


namespace ratio_of_areas_eq_nine_sixteenth_l1428_142803

-- Definitions based on conditions
def side_length_C : ℝ := 45
def side_length_D : ℝ := 60
def area (s : ℝ) : ℝ := s * s

-- Theorem stating the desired proof problem
theorem ratio_of_areas_eq_nine_sixteenth :
  (area side_length_C) / (area side_length_D) = 9 / 16 :=
by
  sorry

end ratio_of_areas_eq_nine_sixteenth_l1428_142803


namespace length_LM_in_triangle_l1428_142830

theorem length_LM_in_triangle 
  (A B C K L M : Type*) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace K] [MetricSpace L] [MetricSpace M]
  (angle_A: Real) (angle_B: Real) (angle_C: Real)
  (AK: Real) (BL: Real) (MC: Real) (KL: Real) (KM: Real)
  (H1: angle_A = 90) (H2: angle_B = 30) (H3: angle_C = 60) 
  (H4: AK = 4) (H5: BL = 31) (H6: MC = 3) 
  (H7: KL = KM) : 
  (LM = 20) :=
sorry

end length_LM_in_triangle_l1428_142830


namespace unique_solution_l1428_142878

theorem unique_solution (p : ℕ) (a b n : ℕ) : 
  p.Prime → 2^a + p^b = n^(p-1) → (p, a, b, n) = (3, 0, 1, 2) ∨ (p = 2) :=
by {
  sorry
}

end unique_solution_l1428_142878


namespace simplify_expression_l1428_142892

theorem simplify_expression (x y : ℝ) : 7 * x + 8 * y - 3 * x + 4 * y + 10 = 4 * x + 12 * y + 10 :=
by
  sorry

end simplify_expression_l1428_142892


namespace complementary_angles_not_obtuse_l1428_142867

-- Define the concept of complementary angles.
def is_complementary (a b : ℝ) : Prop :=
  a + b = 90

-- Define that neither angle should be obtuse.
def not_obtuse (a b : ℝ) : Prop :=
  a < 90 ∧ b < 90

-- Proof problem statement
theorem complementary_angles_not_obtuse (a b : ℝ) (ha : a < 90) (hb : b < 90) (h_comp : is_complementary a b) : 
  not_obtuse a b :=
by
  sorry

end complementary_angles_not_obtuse_l1428_142867


namespace distinct_meals_l1428_142874

def num_entrees : ℕ := 4
def num_drinks : ℕ := 2
def num_desserts : ℕ := 2

theorem distinct_meals : num_entrees * num_drinks * num_desserts = 16 := by
  sorry

end distinct_meals_l1428_142874


namespace max_value_sqrt_add_l1428_142881

noncomputable def sqrt_add (a b : ℝ) : ℝ := Real.sqrt (a + 1) + Real.sqrt (b + 3)

theorem max_value_sqrt_add (a b : ℝ) (h : 0 < a) (h' : 0 < b) (hab : a + b = 5) :
  sqrt_add a b ≤ 3 * Real.sqrt 2 :=
by
  sorry

end max_value_sqrt_add_l1428_142881


namespace exists_natural_numbers_gt_10pow10_divisible_by_self_plus_2012_l1428_142816

theorem exists_natural_numbers_gt_10pow10_divisible_by_self_plus_2012 :
  ∃ (a b c : ℕ), 
    a > 10^10 ∧ b > 10^10 ∧ c > 10^10 ∧ 
    a ∣ (a * b * c + 2012) ∧ b ∣ (a * b * c + 2012) ∧ c ∣ (a * b * c + 2012) :=
by
  sorry

end exists_natural_numbers_gt_10pow10_divisible_by_self_plus_2012_l1428_142816


namespace ratio_of_segments_l1428_142887

variable (F S T : ℕ)

theorem ratio_of_segments : T = 10 → F = 2 * (S + T) → F + S + T = 90 → (T / S = 1 / 2) :=
by
  intros hT hF hSum
  sorry

end ratio_of_segments_l1428_142887


namespace problem_statement_l1428_142826

theorem problem_statement (a b c : ℝ) (h_nonneg_a : a ≥ 0) (h_nonneg_b : b ≥ 0) (h_nonneg_c : c ≥ 0) (h_condition : a * b + b * c + c * a = 1 / 3) :
  1 / (a^2 - b * c + 1) + 1 / (b^2 - c * a + 1) + 1 / (c^2 - a * b + 1) ≤ 3 :=
by
  sorry

end problem_statement_l1428_142826


namespace zeta_1_8_add_zeta_2_8_add_zeta_3_8_l1428_142832

noncomputable def compute_s8 (s : ℕ → ℂ) : ℂ :=
  s 8

theorem zeta_1_8_add_zeta_2_8_add_zeta_3_8 {ζ : ℕ → ℂ} 
  (h1 : ζ 1 + ζ 2 + ζ 3 = 2)
  (h2 : ζ 1^2 + ζ 2^2 + ζ 3^2 = 6)
  (h3 : ζ 1^3 + ζ 2^3 + ζ 3^3 = 18)
  (rec : ∀ n, ζ (n + 3) = 2 * ζ (n + 2) + ζ (n + 1) - (4 / 3) * ζ n)
  (s0 : ζ 0 = 3)
  (s1 : ζ 1 = 2)
  (s2 : ζ 2 = 6)
  (s3 : ζ 3 = 18)
  : ζ 8 = compute_s8 ζ := 
sorry

end zeta_1_8_add_zeta_2_8_add_zeta_3_8_l1428_142832


namespace sum_of_solutions_l1428_142863

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 2)^2 = 81) (h2 : (x2 - 2)^2 = 81) :
  x1 + x2 = 4 := by
  sorry

end sum_of_solutions_l1428_142863


namespace ninth_term_l1428_142823

variable (a d : ℤ)
variable (h1 : a + 2 * d = 20)
variable (h2 : a + 5 * d = 26)

theorem ninth_term (a d : ℤ) (h1 : a + 2 * d = 20) (h2 : a + 5 * d = 26) : a + 8 * d = 32 :=
sorry

end ninth_term_l1428_142823


namespace peg_arrangement_l1428_142840

theorem peg_arrangement :
  let Y := 5
  let R := 4
  let G := 3
  let B := 2
  let O := 1
  (Y! * R! * G! * B! * O!) = 34560 :=
by
  sorry

end peg_arrangement_l1428_142840


namespace abs_inequality_solution_set_l1428_142807

theorem abs_inequality_solution_set (x : ℝ) :
  |x| + |x - 1| < 2 ↔ - (1 / 2) < x ∧ x < (3 / 2) :=
by
  sorry

end abs_inequality_solution_set_l1428_142807


namespace am_hm_inequality_l1428_142870

theorem am_hm_inequality (a1 a2 a3 : ℝ) (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) (h_sum : a1 + a2 + a3 = 1) : 
  (1 / a1) + (1 / a2) + (1 / a3) ≥ 9 :=
by
  sorry

end am_hm_inequality_l1428_142870


namespace candidate_function_is_odd_and_increasing_l1428_142856

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_increasing_function (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

def candidate_function (x : ℝ) : ℝ := x * |x|

theorem candidate_function_is_odd_and_increasing :
  is_odd_function candidate_function ∧ is_increasing_function candidate_function :=
by
  sorry

end candidate_function_is_odd_and_increasing_l1428_142856


namespace sum_of_squares_of_rates_l1428_142854

theorem sum_of_squares_of_rates :
  ∃ (b j s : ℕ), 3 * b + j + 5 * s = 89 ∧ 4 * b + 3 * j + 2 * s = 106 ∧ b^2 + j^2 + s^2 = 821 := 
by
  sorry

end sum_of_squares_of_rates_l1428_142854


namespace solve_inequality_l1428_142814

theorem solve_inequality (x : ℝ) :
  (x^2 - 4 * x - 12) / (x - 3) < 0 ↔ (-2 < x ∧ x < 3) ∨ (3 < x ∧ x < 6) := by
  sorry

end solve_inequality_l1428_142814


namespace inequality_proof_equality_condition_l1428_142824

theorem inequality_proof (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ( (3 * a * b * c / (a * b + a * c + b * c)) ^ (a^2 + b^2 + c^2) ) ≥ (a ^ (b * c) * b ^ (a * c) * c ^ (a * b)) := 
sorry

theorem equality_condition (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ( (3 * a * b * c / (a * b + a * c + b * c)) ^ (a^2 + b^2 + c^2) ) = (a ^ (b * c) * b ^ (a * c) * c ^ (a * b)) ↔ a = b ∧ b = c := 
sorry

end inequality_proof_equality_condition_l1428_142824


namespace compute_d1e1_d2e2_d3e3_l1428_142817

-- Given polynomials and conditions
variables {R : Type*} [CommRing R]

noncomputable def P (x : R) : R :=
  x^7 - x^6 + x^4 - x^3 + x^2 - x + 1

noncomputable def Q (x : R) (d1 d2 d3 e1 e2 e3 : R) : R :=
  (x^2 + d1 * x + e1) * (x^2 + d2 * x + e2) * (x^2 + d3 * x + e3)

-- Given conditions
theorem compute_d1e1_d2e2_d3e3 
  (d1 d2 d3 e1 e2 e3 : R)
  (h : ∀ x : R, P x = Q x d1 d2 d3 e1 e2 e3) : 
  d1 * e1 + d2 * e2 + d3 * e3 = -1 :=
by
  sorry

end compute_d1e1_d2e2_d3e3_l1428_142817


namespace balloon_permutations_l1428_142883

theorem balloon_permutations : 
  let n : ℕ := 7
  let k1 : ℕ := 2
  let k2 : ℕ := 2
  ∃ distinct_arrangements : ℕ, 
  distinct_arrangements = n.factorial / (k1.factorial * k2.factorial) 
  ∧ distinct_arrangements = 1260 :=
by
  sorry

end balloon_permutations_l1428_142883


namespace sum_of_digits_l1428_142847

theorem sum_of_digits (x : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (h : 10 * x + 6 * x = 16) : x + 6 * x = 7 :=
by
  -- The proof is omitted
  sorry

end sum_of_digits_l1428_142847


namespace james_bought_400_fish_l1428_142875

theorem james_bought_400_fish
  (F : ℝ)
  (h1 : 0.80 * F = 320)
  (h2 : F / 0.80 = 400) :
  F = 400 :=
by
  sorry

end james_bought_400_fish_l1428_142875


namespace fraction_sum_l1428_142853

theorem fraction_sum : (3 / 4 : ℚ) + (6 / 9 : ℚ) = 17 / 12 := 
by 
  -- Sorry placeholder to indicate proof is not provided.
  sorry

end fraction_sum_l1428_142853


namespace inverse_variation_z_x_square_l1428_142846

theorem inverse_variation_z_x_square (x z : ℝ) (K : ℝ) 
  (h₀ : z * x^2 = K) 
  (h₁ : x = 3 ∧ z = 2)
  (h₂ : z = 8) :
  x = 3 / 2 := 
by 
  sorry

end inverse_variation_z_x_square_l1428_142846


namespace exists_x_gt_zero_negation_l1428_142839

theorem exists_x_gt_zero_negation :
  (∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
  sorry  -- Proof goes here

end exists_x_gt_zero_negation_l1428_142839


namespace common_ratio_of_geometric_progression_l1428_142805

theorem common_ratio_of_geometric_progression (a1 q : ℝ) (S3 : ℝ) (a2 : ℝ)
  (h1 : S3 = a1 * (1 + q + q^2))
  (h2 : a2 = a1 * q)
  (h3 : a2 + S3 = 0) :
  q = -1 := 
  sorry

end common_ratio_of_geometric_progression_l1428_142805


namespace kindergarten_library_models_l1428_142828

theorem kindergarten_library_models
  (paid : ℕ)
  (reduced_price : ℕ)
  (models_total_gt_5 : ℕ)
  (bought : ℕ) 
  (condition : paid = 570 ∧ reduced_price = 95 ∧ models_total_gt_5 > 5 ∧ bought = 3 * (2 : ℕ)) :
  exists x : ℕ, bought / 3 = x ∧ x = 2 :=
by
  sorry

end kindergarten_library_models_l1428_142828


namespace actual_distance_traveled_l1428_142801

theorem actual_distance_traveled :
  ∀ (t : ℝ) (d1 d2 : ℝ),
  d1 = 15 * t →
  d2 = 30 * t →
  d2 = d1 + 45 →
  d1 = 45 := by
  intro t d1 d2 h1 h2 h3
  sorry

end actual_distance_traveled_l1428_142801


namespace problem_f_f2_equals_16_l1428_142845

noncomputable def f (x : ℝ) : ℝ :=
if h : x < 3 then x^2 else 2^x

theorem problem_f_f2_equals_16 : f (f 2) = 16 :=
by
  sorry

end problem_f_f2_equals_16_l1428_142845


namespace total_amount_shared_l1428_142895

theorem total_amount_shared (X_share Y_share Z_share total_amount : ℝ) 
                            (h1 : Y_share = 0.45 * X_share) 
                            (h2 : Z_share = 0.50 * X_share) 
                            (h3 : Y_share = 45) : 
                            total_amount = X_share + Y_share + Z_share := 
by 
  -- Sorry to skip the proof
  sorry

end total_amount_shared_l1428_142895


namespace no_base6_digit_d_divisible_by_7_l1428_142864

theorem no_base6_digit_d_divisible_by_7 : 
∀ d : ℕ, (d < 6) → ¬ (654 + 42 * d) % 7 = 0 :=
by
  intro d h
  -- Proof is omitted as requested
  sorry

end no_base6_digit_d_divisible_by_7_l1428_142864


namespace sean_less_points_than_combined_l1428_142869

def tobee_points : ℕ := 4
def jay_points : ℕ := tobee_points + 6
def combined_points_tobee_jay : ℕ := tobee_points + jay_points
def total_team_points : ℕ := 26
def sean_points : ℕ := total_team_points - combined_points_tobee_jay

theorem sean_less_points_than_combined : (combined_points_tobee_jay - sean_points) = 2 := by
  sorry

end sean_less_points_than_combined_l1428_142869


namespace company_hired_22_additional_males_l1428_142831

theorem company_hired_22_additional_males
  (E M : ℕ) 
  (initial_percentage_female : ℝ)
  (final_total_employees : ℕ)
  (final_percentage_female : ℝ)
  (initial_female_count : initial_percentage_female * E = 0.6 * E)
  (final_employee_count : E + M = 264) 
  (final_female_count : initial_percentage_female * E = final_percentage_female * (E + M)) :
  M = 22 := 
by
  sorry

end company_hired_22_additional_males_l1428_142831


namespace square_of_1027_l1428_142898

theorem square_of_1027 :
  1027 * 1027 = 1054729 :=
by
  sorry

end square_of_1027_l1428_142898


namespace min_height_regular_quadrilateral_pyramid_l1428_142858

theorem min_height_regular_quadrilateral_pyramid (r : ℝ) (a : ℝ) (h : 2 * r < a / 2) : 
  ∃ x : ℝ, (0 < x) ∧ (∃ V : ℝ, ∀ x' : ℝ, V = (a^2 * x) / 3 ∧ (∀ x' ≠ x, V < (a^2 * x') / 3)) ∧ x = (r * (5 + Real.sqrt 17)) / 2 :=
sorry

end min_height_regular_quadrilateral_pyramid_l1428_142858


namespace Keith_picked_6_apples_l1428_142879

def m : ℝ := 7.0
def n : ℝ := 3.0
def t : ℝ := 10.0

noncomputable def r_m := m - n
noncomputable def k := t - r_m

-- Theorem Statement confirming Keith picked 6.0 apples
theorem Keith_picked_6_apples : k = 6.0 := by
  sorry

end Keith_picked_6_apples_l1428_142879


namespace find_f_of_3_l1428_142851

noncomputable def f : ℝ → ℝ :=
  sorry

theorem find_f_of_3 (h : ∀ x : ℝ, x ≠ 0 → f x - 3 * f (1 / x) = 3 ^ x) :
  f 3 = (-27 + 3 * (3 ^ (1 / 3))) / 8 :=
sorry

end find_f_of_3_l1428_142851


namespace probability_exactly_two_even_dice_l1428_142843

theorem probability_exactly_two_even_dice :
  let p_even := 1 / 2
  let p_not_even := 1 / 2
  let number_of_ways := 3
  let probability_each_way := (p_even * p_even * p_not_even)
  3 * probability_each_way = 3 / 8 :=
by
  sorry

end probability_exactly_two_even_dice_l1428_142843


namespace root_ratio_equiv_l1428_142811

theorem root_ratio_equiv :
  (81 ^ (1 / 3)) / (81 ^ (1 / 4)) = 81 ^ (1 / 12) :=
by
  sorry

end root_ratio_equiv_l1428_142811


namespace latest_start_time_is_correct_l1428_142835

noncomputable def doughComingToRoomTemp : ℕ := 1  -- 1 hour
noncomputable def shapingDough : ℕ := 15         -- 15 minutes
noncomputable def proofingDough : ℕ := 2         -- 2 hours
noncomputable def bakingBread : ℕ := 30          -- 30 minutes
noncomputable def coolingBread : ℕ := 15         -- 15 minutes
noncomputable def bakeryOpeningTime : ℕ := 6     -- 6:00 am

-- Total preparation time in minutes
noncomputable def totalPreparationTimeInMinutes : ℕ :=
  (doughComingToRoomTemp * 60) + shapingDough + (proofingDough * 60) + bakingBread + coolingBread

-- Total preparation time in hours
noncomputable def totalPreparationTimeInHours : ℕ :=
  totalPreparationTimeInMinutes / 60

-- Latest time the baker can start working
noncomputable def latestTimeBakerCanStart : ℕ :=
  if (bakeryOpeningTime - totalPreparationTimeInHours) < 0 then 24 + (bakeryOpeningTime - totalPreparationTimeInHours)
  else bakeryOpeningTime - totalPreparationTimeInHours

theorem latest_start_time_is_correct : latestTimeBakerCanStart = 2 := by
  sorry

end latest_start_time_is_correct_l1428_142835


namespace maple_logs_correct_l1428_142806

/-- Each pine tree makes 80 logs. -/
def pine_logs := 80

/-- Each walnut tree makes 100 logs. -/
def walnut_logs := 100

/-- Jerry cuts up 8 pine trees. -/
def pine_trees := 8

/-- Jerry cuts up 3 maple trees. -/
def maple_trees := 3

/-- Jerry cuts up 4 walnut trees. -/
def walnut_trees := 4

/-- The total number of logs is 1220. -/
def total_logs := 1220

/-- The number of logs each maple tree makes. -/
def maple_logs := 60

theorem maple_logs_correct :
  (pine_trees * pine_logs) + (maple_trees * maple_logs) + (walnut_trees * walnut_logs) = total_logs :=
by
  -- (8 * 80) + (3 * 60) + (4 * 100) = 1220
  sorry

end maple_logs_correct_l1428_142806


namespace car_with_highest_avg_speed_l1428_142800

-- Conditions
def distance_A : ℕ := 715
def time_A : ℕ := 11
def distance_B : ℕ := 820
def time_B : ℕ := 12
def distance_C : ℕ := 950
def time_C : ℕ := 14

-- Average Speeds
def avg_speed_A : ℚ := distance_A / time_A
def avg_speed_B : ℚ := distance_B / time_B
def avg_speed_C : ℚ := distance_C / time_C

-- Theorem
theorem car_with_highest_avg_speed : avg_speed_B > avg_speed_A ∧ avg_speed_B > avg_speed_C :=
by
  sorry

end car_with_highest_avg_speed_l1428_142800


namespace cylinder_radius_l1428_142841

theorem cylinder_radius
  (r₁ r₂ : ℝ)
  (rounds₁ rounds₂ : ℕ)
  (H₁ : r₁ = 14)
  (H₂ : rounds₁ = 70)
  (H₃ : rounds₂ = 49)
  (L₁ : rounds₁ * 2 * Real.pi * r₁ = rounds₂ * 2 * Real.pi * r₂) :
  r₂ = 20 := 
sorry

end cylinder_radius_l1428_142841


namespace combined_work_rate_l1428_142859

-- Define the context and the key variables
variable (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)

-- State the theorem corresponding to the proof problem
theorem combined_work_rate (h_a : a ≠ 0) (h_b : b ≠ 0) : 
  1/a + 1/b = (a * b) / (a + b) * (1/a * 1/b) :=
sorry

end combined_work_rate_l1428_142859


namespace ratio_of_perimeters_l1428_142837

-- Define lengths of the rectangular patch
def length_rect : ℝ := 400
def width_rect : ℝ := 300

-- Define the length of the side of the square patch
def side_square : ℝ := 700

-- Define the perimeters of both patches
def P_square : ℝ := 4 * side_square
def P_rectangle : ℝ := 2 * (length_rect + width_rect)

-- Theorem stating the ratio of the perimeters
theorem ratio_of_perimeters : P_square / P_rectangle = 2 :=
by sorry

end ratio_of_perimeters_l1428_142837


namespace remainder_div_1234_567_89_1011_mod_12_l1428_142891

theorem remainder_div_1234_567_89_1011_mod_12 :
  (1234^567 + 89^1011) % 12 = 9 := 
sorry

end remainder_div_1234_567_89_1011_mod_12_l1428_142891


namespace average_temperature_problem_l1428_142834

variable {T W Th F : ℝ}

theorem average_temperature_problem (h1 : (W + Th + 44) / 3 = 34) (h2 : T = 38) : 
  (T + W + Th) / 3 = 32 := by
  sorry

end average_temperature_problem_l1428_142834


namespace chord_length_y_eq_x_plus_one_meets_circle_l1428_142865

noncomputable def chord_length (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem chord_length_y_eq_x_plus_one_meets_circle 
  (A B : ℝ × ℝ) 
  (hA : A.2 = A.1 + 1) 
  (hB : B.2 = B.1 + 1) 
  (hA_on_circle : A.1^2 + A.2^2 + 2 * A.2 - 3 = 0)
  (hB_on_circle : B.1^2 + B.2^2 + 2 * B.2 - 3 = 0) :
  chord_length A B = 2 * Real.sqrt 2 := 
sorry

end chord_length_y_eq_x_plus_one_meets_circle_l1428_142865


namespace find_a_b_monotonicity_l1428_142871

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (x^2 + a * x + b) / x

theorem find_a_b (a b : ℝ) (h_odd : ∀ x ≠ 0, f (-x) a b = -f x a b) (h_eq : f 1 a b = f 4 a b) :
  a = 0 ∧ b = 4 := by sorry

theorem monotonicity (f : ℝ → ℝ) (h : ∀ x ≠ 0, f x = x + 4 / x) :
  (∀ x1 x2, 0 < x1 ∧ x1 ≤ 2 ∧ x1 < x2 ∧ x2 ≤ 2 → f x1 > f x2) ∧
  (∀ x1 x2, 2 < x1 ∧ x1 < x2 → f x1 < f x2) := by sorry

end find_a_b_monotonicity_l1428_142871


namespace correct_statement_l1428_142868

open Set

variable (U : Set ℕ) (M : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5}) (hM : U \ M = {1, 3})

theorem correct_statement : 2 ∈ M :=
by
  sorry

end correct_statement_l1428_142868


namespace ratio_a_f_l1428_142852

theorem ratio_a_f (a b c d e f : ℕ)
  (h1 : a / b = 1 / 4)
  (h2 : b / c = 13 / 9)
  (h3 : c / d = 5 / 13)
  (h4 : d / e = 2 / 3)
  (h5 : e / f = 7 / 5) :
  a / f = 7 / 6 := by
  sorry

end ratio_a_f_l1428_142852


namespace solve_for_x_l1428_142885

theorem solve_for_x 
  (a b c d x y z w : ℝ) 
  (H1 : x + y + z + w = 360)
  (H2 : a = x + y / 2) 
  (H3 : b = y + z / 2) 
  (H4 : c = z + w / 2) 
  (H5 : d = w + x / 2) : 
  x = (16 / 15) * (a - b / 2 + c / 4 - d / 8) :=
sorry


end solve_for_x_l1428_142885


namespace K_time_9_hours_l1428_142882

theorem K_time_9_hours
  (x : ℝ) -- x is the speed of K
  (hx : 45 / x = 9) -- K's time for 45 miles is 9 hours
  (y : ℝ) -- y is the speed of M
  (h₁ : x = y + 0.5) -- K travels 0.5 mph faster than M
  (h₂ : 45 / y - 45 / x = 3 / 4) -- K takes 3/4 hour less than M
  : 45 / x = 9 :=
by
  sorry

end K_time_9_hours_l1428_142882


namespace find_triplets_l1428_142894

theorem find_triplets (m n k : ℕ) (pos_m : 0 < m) (pos_n : 0 < n) (pos_k : 0 < k) : 
  (k^m ∣ m^n - 1) ∧ (k^n ∣ n^m - 1) ↔ (k = 1) ∨ (m = 1 ∧ n = 1) :=
by
  sorry

end find_triplets_l1428_142894


namespace greatest_difference_areas_l1428_142855

theorem greatest_difference_areas (l w l' w' : ℕ) (h₁ : 2*l + 2*w = 120) (h₂ : 2*l' + 2*w' = 120) : 
  l * w ≤ 900 ∧ (l = 30 → w = 30) ∧ l' * w' ≤ 900 ∧ (l' = 30 → w' = 30)  → 
  ∃ (A₁ A₂ : ℕ), (A₁ = l * w ∧ A₂ = l' * w') ∧ (841 = l * w - l' * w') := 
sorry

end greatest_difference_areas_l1428_142855


namespace sum_of_x_y_l1428_142819

theorem sum_of_x_y (x y : ℚ) (h1 : 1/x + 1/y = 5) (h2 : 1/x - 1/y = -9) : x + y = -5/14 := 
by
  sorry

end sum_of_x_y_l1428_142819


namespace total_weight_of_shells_l1428_142877

noncomputable def initial_weight : ℝ := 5.25
noncomputable def weight_large_shell_g : ℝ := 700
noncomputable def grams_per_pound : ℝ := 453.592
noncomputable def additional_weight : ℝ := 4.5

/-
We need to prove:
5.25 pounds (initial weight) + (700 grams * (1 pound / 453.592 grams)) (weight of large shell in pounds) + 4.5 pounds (additional weight) = 11.293235835 pounds
-/
theorem total_weight_of_shells :
  initial_weight + (weight_large_shell_g / grams_per_pound) + additional_weight = 11.293235835 := by
    -- Proof will be inserted here
    sorry

end total_weight_of_shells_l1428_142877


namespace price_increase_for_1620_profit_maximizing_profit_l1428_142857

-- To state the problem, we need to define some variables and the associated conditions.

def cost_price : ℝ := 13
def initial_selling_price : ℝ := 20
def initial_monthly_sales : ℝ := 200
def decrease_in_sales_per_yuan : ℝ := 10
def profit_condition (x : ℝ) : ℝ := (initial_selling_price + x - cost_price) * (initial_monthly_sales - decrease_in_sales_per_yuan * x)
def profit_function (x : ℝ) : ℝ := -(10 * x ^ 2) + (130 * x) + 140

-- Part (1): Prove the price increase x such that the profit is 1620 yuan
theorem price_increase_for_1620_profit :
  ∃ (x : ℝ), profit_condition x = 1620 ∧ (x = 2 ∨ x = 11) :=
sorry

-- Part (2): Prove that the selling price that maximizes profit is 26.5 yuan and max profit is 1822.5 yuan
theorem maximizing_profit :
  ∃ (x : ℝ), (x = 13 / 2) ∧ profit_function (13 / 2) = 3645 / 2 :=
sorry

end price_increase_for_1620_profit_maximizing_profit_l1428_142857


namespace trig_identity_cos2theta_tan_minus_pi_over_4_l1428_142884

variable (θ : ℝ)

-- Given condition
def tan_theta_is_2 : Prop := Real.tan θ = 2

-- Proof problem 1: Prove that cos(2θ) = -3/5
def cos2theta (θ : ℝ) (h : tan_theta_is_2 θ) : Prop :=
  Real.cos (2 * θ) = -3 / 5

-- Proof problem 2: Prove that tan(θ - π/4) = 1/3
def tan_theta_minus_pi_over_4 (θ : ℝ) (h : tan_theta_is_2 θ) : Prop :=
  Real.tan (θ - Real.pi / 4) = 1 / 3

-- Main theorem statement
theorem trig_identity_cos2theta_tan_minus_pi_over_4 
  (θ : ℝ) (h : tan_theta_is_2 θ) :
  cos2theta θ h ∧ tan_theta_minus_pi_over_4 θ h :=
sorry

end trig_identity_cos2theta_tan_minus_pi_over_4_l1428_142884


namespace total_number_of_boys_in_class_is_40_l1428_142829

theorem total_number_of_boys_in_class_is_40 
  (n : ℕ) (h : 27 - 7 = n / 2):
  n = 40 :=
by
  sorry

end total_number_of_boys_in_class_is_40_l1428_142829


namespace transaction_loss_l1428_142821

theorem transaction_loss :
  let house_sale_price := 10000
  let store_sale_price := 15000
  let house_loss_percentage := 0.25
  let store_gain_percentage := 0.25
  let h := house_sale_price / (1 - house_loss_percentage)
  let s := store_sale_price / (1 + store_gain_percentage)
  let total_cost_price := h + s
  let total_selling_price := house_sale_price + store_sale_price
  let difference := total_selling_price - total_cost_price
  difference = -1000 / 3 :=
by
  sorry

end transaction_loss_l1428_142821


namespace fraction_equality_l1428_142872

theorem fraction_equality
  (a b : ℝ)
  (x : ℝ)
  (h1 : x = (a^2) / (b^2))
  (h2 : a ≠ b)
  (h3 : b ≠ 0) :
  (a^2 + b^2) / (a^2 - b^2) = (x + 1) / (x - 1) :=
by
  sorry

end fraction_equality_l1428_142872


namespace solve_inequalities_l1428_142873

theorem solve_inequalities :
  (∀ x : ℝ, x^2 + 3 * x - 10 ≥ 0 ↔ (x ≤ -5 ∨ x ≥ 2)) ∧
  (∀ x : ℝ, x^2 - 3 * x - 2 ≤ 0 ↔ (3 - Real.sqrt 17) / 2 ≤ x ∧ x ≤ (3 + Real.sqrt 17) / 2) :=
by
  sorry

end solve_inequalities_l1428_142873


namespace arithmetic_mean_of_geometric_sequence_l1428_142809

theorem arithmetic_mean_of_geometric_sequence (a r : ℕ) (h_a : a = 4) (h_r : r = 3) :
    ((a) + (a * r) + (a * r^2)) / 3 = (52 / 3) :=
by
  sorry

end arithmetic_mean_of_geometric_sequence_l1428_142809


namespace two_circles_common_tangents_l1428_142889

theorem two_circles_common_tangents (r : ℝ) (h_r : 0 < r) :
  ¬ ∃ (n : ℕ), n = 2 ∧
  (∀ (config : ℕ), 
    (config = 0 → n = 4) ∨
    (config = 1 → n = 0) ∨
    (config = 2 → n = 3) ∨
    (config = 3 → n = 1)) :=
by
  sorry

end two_circles_common_tangents_l1428_142889


namespace miguel_paint_area_l1428_142802

def wall_height := 10
def wall_length := 15
def window_side := 3

theorem miguel_paint_area :
  (wall_height * wall_length) - (window_side * window_side) = 141 := 
by
  sorry

end miguel_paint_area_l1428_142802


namespace remainder_division_l1428_142812

-- Define the polynomial f(x) = x^51 + 51
def f (x : ℤ) : ℤ := x^51 + 51

-- State the theorem to be proven
theorem remainder_division : f (-1) = 50 :=
by
  -- proof goes here
  sorry

end remainder_division_l1428_142812


namespace homothety_maps_C_to_E_l1428_142893

-- Defining Points and Circles
variable {Point Circle : Type}
variable [Inhabited Point] -- assuming Point type is inhabited

-- Definitions for points H, K_A, I_A, K_B, I_B, K_C, I_C
variables (H K_A I_A K_B I_B K_C I_C : Point)

-- Define midpoints
def is_midpoint (A B M : Point) : Prop := sorry -- In a real proof, you would define midpoint in terms of coordinates

-- Define homothety function
def homothety (center : Point) (ratio : ℝ) (P : Point) : Point := sorry -- In a real proof, you would define the homothety transformation

-- Defining Circles
variables (C E : Circle)

-- Define circumcircle of a triangle
def is_circumcircle (a b c : Point) (circle : Circle) : Prop := sorry

-- Statements from conditions
axiom midpointA : is_midpoint H K_A I_A
axiom midpointB : is_midpoint H K_B I_B
axiom midpointC : is_midpoint H K_C I_C

axiom circumcircle_C : is_circumcircle K_A K_B K_C C
axiom circumcircle_E : is_circumcircle I_A I_B I_C E

-- Lean theorem stating the proof problem
theorem homothety_maps_C_to_E :
  ∀ (H K_A I_A K_B I_B K_C I_C : Point) (C E : Circle),
  (is_midpoint H K_A I_A) →
  (is_midpoint H K_B I_B) →
  (is_midpoint H K_C I_C) →
  (is_circumcircle K_A K_B K_C C) →
  (is_circumcircle I_A I_B I_C E) →
  (homothety H 0.5 K_A = I_A ) →
  (homothety H 0.5 K_B = I_B ) →
  (homothety H 0.5 K_C = I_C ) →
  C = E :=
by intro; sorry

end homothety_maps_C_to_E_l1428_142893


namespace smallest_value_at_x_5_l1428_142848

-- Define the variable x
def x : ℕ := 5

-- Define each expression
def exprA := 8 / x
def exprB := 8 / (x + 2)
def exprC := 8 / (x - 2)
def exprD := x / 8
def exprE := (x + 2) / 8

-- The goal is to prove that exprD yields the smallest value
theorem smallest_value_at_x_5 : exprD = min (min (min exprA exprB) (min exprC exprE)) :=
sorry

end smallest_value_at_x_5_l1428_142848


namespace maximize_value_l1428_142804

noncomputable def maximum_value (x y : ℝ) : ℝ :=
  3 * x - 2 * y

theorem maximize_value (x y : ℝ) (h : x^2 + y^2 + x * y = 1) : maximum_value x y ≤ 5 :=
sorry

end maximize_value_l1428_142804
