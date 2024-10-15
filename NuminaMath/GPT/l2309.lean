import Mathlib

namespace NUMINAMATH_GPT_find_a_minus_b_l2309_230939

variable {a b : ℤ}

theorem find_a_minus_b (h1 : a^2 = 9) (h2 : |b| = 4) (h3 : a > b) : a - b = 7 :=
  sorry

end NUMINAMATH_GPT_find_a_minus_b_l2309_230939


namespace NUMINAMATH_GPT_pos_difference_between_highest_and_second_smallest_enrollment_l2309_230950

def varsity_enrollment : ℕ := 1520
def northwest_enrollment : ℕ := 1430
def central_enrollment : ℕ := 1900
def greenbriar_enrollment : ℕ := 1850

theorem pos_difference_between_highest_and_second_smallest_enrollment :
  (central_enrollment - varsity_enrollment) = 380 := 
by 
  sorry

end NUMINAMATH_GPT_pos_difference_between_highest_and_second_smallest_enrollment_l2309_230950


namespace NUMINAMATH_GPT_range_of_a_l2309_230917

variable (a : ℝ)

def proposition_p : Prop :=
  ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (x^2 - a ≥ 0)

def proposition_q : Prop :=
  ∃ x : ℝ, x^2 + 2 * a * x + (2 - a) = 0

theorem range_of_a (hp : proposition_p a) (hq : proposition_q a) : a ≤ -2 ∨ a = 1 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l2309_230917


namespace NUMINAMATH_GPT_sin_13pi_over_4_l2309_230943

theorem sin_13pi_over_4 : Real.sin (13 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_13pi_over_4_l2309_230943


namespace NUMINAMATH_GPT_tournament_byes_and_games_l2309_230915

/-- In a single-elimination tournament with 300 players initially registered,
- if the number of players in each subsequent round must be a power of 2,
- then 44 players must receive a bye in the first round, and 255 total games
- must be played to determine the champion. -/
theorem tournament_byes_and_games :
  let initial_players := 300
  let pow2_players := 256
  44 = initial_players - pow2_players ∧
  255 = pow2_players - 1 :=
by
  let initial_players := 300
  let pow2_players := 256
  have h_byes : 44 = initial_players - pow2_players := by sorry
  have h_games : 255 = pow2_players - 1 := by sorry
  exact ⟨h_byes, h_games⟩

end NUMINAMATH_GPT_tournament_byes_and_games_l2309_230915


namespace NUMINAMATH_GPT_range_of_a_l2309_230964

def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 2

noncomputable def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x = 0 → x < 0) → (∃ x : ℝ, f a x = 0) → a < -Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l2309_230964


namespace NUMINAMATH_GPT_payment_equation_1_payment_equation_2_cost_effective_40_combined_cost_effective_40_l2309_230922

namespace ShoppingMall

def tea_set_price : ℕ := 200
def tea_bowl_price : ℕ := 20
def discount_option_1 (x : ℕ) : ℕ := 20 * x + 5400
def discount_option_2 (x : ℕ) : ℕ := 19 * x + 5700
def combined_option_40 : ℕ := 6000 + 190

theorem payment_equation_1 (x : ℕ) (hx : x > 30) : 
  discount_option_1 x = 20 * x + 5400 :=
by sorry

theorem payment_equation_2 (x : ℕ) (hx : x > 30) : 
  discount_option_2 x = 19 * x + 5700 :=
by sorry

theorem cost_effective_40 : discount_option_1 40 < discount_option_2 40 :=
by sorry

theorem combined_cost_effective_40 : combined_option_40 < discount_option_1 40 ∧ combined_option_40 < discount_option_2 40 :=
by sorry

end ShoppingMall

end NUMINAMATH_GPT_payment_equation_1_payment_equation_2_cost_effective_40_combined_cost_effective_40_l2309_230922


namespace NUMINAMATH_GPT_sequence_arithmetic_difference_neg1_l2309_230991

variable (a : ℕ → ℝ)

theorem sequence_arithmetic_difference_neg1 (h : ∀ n, a (n + 1) + 1 = a n) : ∀ n, a (n + 1) - a n = -1 :=
by
  intro n
  specialize h n
  linarith

-- Assuming natural numbers starting from 1 (ℕ^*), which is not directly available in Lean.
-- So we use assumptions accordingly.

end NUMINAMATH_GPT_sequence_arithmetic_difference_neg1_l2309_230991


namespace NUMINAMATH_GPT_find_digits_l2309_230929

variable (M N : ℕ)
def x := 10 * N + M
def y := 10 * M + N

theorem find_digits (h₁ : x > y) (h₂ : x + y = 11 * (x - y)) : M = 4 ∧ N = 5 :=
sorry

end NUMINAMATH_GPT_find_digits_l2309_230929


namespace NUMINAMATH_GPT_population_stable_at_K_l2309_230974

-- Definitions based on conditions
def follows_S_curve (population : ℕ → ℝ) : Prop := sorry
def relatively_stable_at_K (population : ℕ → ℝ) (K : ℝ) : Prop := sorry
def ecological_factors_limit (population : ℕ → ℝ) : Prop := sorry

-- The main statement to be proved
theorem population_stable_at_K (population : ℕ → ℝ) (K : ℝ) :
  follows_S_curve population ∧ relatively_stable_at_K population K ∧ ecological_factors_limit population →
  relatively_stable_at_K population K :=
by sorry

end NUMINAMATH_GPT_population_stable_at_K_l2309_230974


namespace NUMINAMATH_GPT_expand_product_l2309_230990

theorem expand_product (y : ℝ) : (y + 3) * (y + 7) = y^2 + 10 * y + 21 := by
  sorry

end NUMINAMATH_GPT_expand_product_l2309_230990


namespace NUMINAMATH_GPT_age_difference_is_36_l2309_230959

open Nat

theorem age_difference_is_36 (a b : ℕ) (h1 : a < 10) (h2 : b < 10) 
    (h_eq : (10 * a + b) + 8 = 3 * ((10 * b + a) + 8)) :
    (10 * a + b) - (10 * b + a) = 36 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_is_36_l2309_230959


namespace NUMINAMATH_GPT_smallest_x_abs_eq_29_l2309_230934

theorem smallest_x_abs_eq_29 : ∃ x: ℝ, |4*x - 5| = 29 ∧ (∀ y: ℝ, |4*y - 5| = 29 → -6 ≤ y) :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_abs_eq_29_l2309_230934


namespace NUMINAMATH_GPT_square_side_length_l2309_230979

/-- 
If a square is drawn by joining the midpoints of the sides of a given square and repeating this process continues indefinitely,
and the sum of the areas of all the squares is 32 cm²,
then the length of the side of the first square is 4 cm. 
-/
theorem square_side_length (s : ℝ) (h : ∑' n : ℕ, (s^2) * (1 / 2)^n = 32) : s = 4 := 
by 
  sorry

end NUMINAMATH_GPT_square_side_length_l2309_230979


namespace NUMINAMATH_GPT_common_ratio_of_geometric_series_l2309_230952

noncomputable def geometric_series_common_ratio (a S : ℝ) : ℝ := 1 - (a / S)

theorem common_ratio_of_geometric_series :
  geometric_series_common_ratio 520 3250 = 273 / 325 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_series_l2309_230952


namespace NUMINAMATH_GPT_kostya_table_prime_l2309_230956

theorem kostya_table_prime {n : ℕ} (hn : n > 3)
  (h : ∀ r s : ℕ, r ≥ 3 → s ≥ 3 → rs - (r + s) ≠ n) : Prime (n + 1) := 
sorry

end NUMINAMATH_GPT_kostya_table_prime_l2309_230956


namespace NUMINAMATH_GPT_number_of_divisors_2310_l2309_230954

theorem number_of_divisors_2310 : Nat.sqrt 2310 = 32 :=
by
  sorry

end NUMINAMATH_GPT_number_of_divisors_2310_l2309_230954


namespace NUMINAMATH_GPT_find_integer_values_of_m_l2309_230963

theorem find_integer_values_of_m (m : ℤ) (x : ℚ) 
  (h₁ : 5 * x - 2 * m = 3 * x - 6 * m + 1)
  (h₂ : -3 < x ∧ x ≤ 2) : m = 0 ∨ m = 1 := 
by 
  sorry

end NUMINAMATH_GPT_find_integer_values_of_m_l2309_230963


namespace NUMINAMATH_GPT_tamtam_blue_shells_l2309_230965

theorem tamtam_blue_shells 
  (total_shells : ℕ)
  (purple_shells : ℕ)
  (pink_shells : ℕ)
  (yellow_shells : ℕ)
  (orange_shells : ℕ)
  (H_total : total_shells = 65)
  (H_purple : purple_shells = 13)
  (H_pink : pink_shells = 8)
  (H_yellow : yellow_shells = 18)
  (H_orange : orange_shells = 14) :
  ∃ blue_shells : ℕ, blue_shells = 12 :=
by
  sorry

end NUMINAMATH_GPT_tamtam_blue_shells_l2309_230965


namespace NUMINAMATH_GPT_alyssa_initial_puppies_l2309_230980

theorem alyssa_initial_puppies (gave_away has_left : ℝ) (h1 : gave_away = 8.5) (h2 : has_left = 12.5) :
    (gave_away + has_left = 21) :=
by
    sorry

end NUMINAMATH_GPT_alyssa_initial_puppies_l2309_230980


namespace NUMINAMATH_GPT_remainder_2011_2015_mod_17_l2309_230925

theorem remainder_2011_2015_mod_17 :
  ((2011 * 2012 * 2013 * 2014 * 2015) % 17) = 7 :=
by
  have h1 : 2011 % 17 = 5 := by sorry
  have h2 : 2012 % 17 = 6 := by sorry
  have h3 : 2013 % 17 = 7 := by sorry
  have h4 : 2014 % 17 = 8 := by sorry
  have h5 : 2015 % 17 = 9 := by sorry
  sorry

end NUMINAMATH_GPT_remainder_2011_2015_mod_17_l2309_230925


namespace NUMINAMATH_GPT_exist_abc_l2309_230913

theorem exist_abc (n k : ℕ) (h1 : 20 < n) (h2 : 1 < k) (h3 : n % k^2 = 0) :
  ∃ a b c : ℕ, n = a * b + b * c + c * a :=
sorry

end NUMINAMATH_GPT_exist_abc_l2309_230913


namespace NUMINAMATH_GPT_inequality_solution_l2309_230968

theorem inequality_solution (x : ℝ) : (3 < x ∧ x < 5) → (x - 5) / ((x - 3)^2) < 0 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_inequality_solution_l2309_230968


namespace NUMINAMATH_GPT_problem_f1_l2309_230947

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f1 (h : ∀ x y : ℝ, f x + f (2 * x + y) + 7 * x * y = f (3 * x - y) + 3 * x^2 + 2) : f 10 = -48 :=
sorry

end NUMINAMATH_GPT_problem_f1_l2309_230947


namespace NUMINAMATH_GPT_part1_part2_part3_l2309_230997

variable {x y : ℚ}

def star (x y : ℚ) : ℚ := x * y + 1

theorem part1 : star 2 4 = 9 := by
  sorry

theorem part2 : star (star 1 4) (-2) = -9 := by
  sorry

theorem part3 (a b c : ℚ) : star a (b + c) + 1 = star a b + star a c := by
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l2309_230997


namespace NUMINAMATH_GPT_total_matches_in_group_l2309_230970

theorem total_matches_in_group (n : ℕ) (hn : n = 6) : 2 * (n * (n - 1) / 2) = 30 :=
by
  sorry

end NUMINAMATH_GPT_total_matches_in_group_l2309_230970


namespace NUMINAMATH_GPT_simplify_expression_l2309_230918

theorem simplify_expression : 
  3 * Real.sqrt 48 - 9 * Real.sqrt (1 / 3) - Real.sqrt 3 * (2 - Real.sqrt 27) = 7 * Real.sqrt 3 + 9 :=
by
  -- The proof is omitted as per the instructions
  sorry

end NUMINAMATH_GPT_simplify_expression_l2309_230918


namespace NUMINAMATH_GPT_rectangle_area_l2309_230942

theorem rectangle_area (b l: ℕ) (h1: l = 3 * b) (h2: 2 * (l + b) = 120) : l * b = 675 := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_area_l2309_230942


namespace NUMINAMATH_GPT_max_distinct_prime_factors_of_a_l2309_230957

noncomputable def distinct_prime_factors (n : ℕ) : ℕ := sorry -- placeholder for the number of distinct prime factors

theorem max_distinct_prime_factors_of_a (a b : ℕ)
  (ha_pos : a > 0) (hb_pos : b > 0)
  (gcd_ab_primes : distinct_prime_factors (gcd a b) = 5)
  (lcm_ab_primes : distinct_prime_factors (lcm a b) = 18)
  (a_less_than_b : distinct_prime_factors a < distinct_prime_factors b) :
  distinct_prime_factors a = 11 :=
sorry

end NUMINAMATH_GPT_max_distinct_prime_factors_of_a_l2309_230957


namespace NUMINAMATH_GPT_largest_int_less_150_gcd_18_eq_6_l2309_230909

theorem largest_int_less_150_gcd_18_eq_6 : ∃ (n : ℕ), n < 150 ∧ gcd n 18 = 6 ∧ ∀ (m : ℕ), m < 150 ∧ gcd m 18 = 6 → m ≤ n ∧ n = 138 := 
by
  sorry

end NUMINAMATH_GPT_largest_int_less_150_gcd_18_eq_6_l2309_230909


namespace NUMINAMATH_GPT_point_of_tangent_parallel_x_axis_l2309_230962

theorem point_of_tangent_parallel_x_axis :
  ∃ M : ℝ × ℝ, (M.1 = -1 ∧ M.2 = -3) ∧
    (∃ y : ℝ, y = M.1^2 + 2 * M.1 - 2 ∧
    (∃ y' : ℝ, y' = 2 * M.1 + 2 ∧ y' = 0)) :=
sorry

end NUMINAMATH_GPT_point_of_tangent_parallel_x_axis_l2309_230962


namespace NUMINAMATH_GPT_avg_rate_of_change_l2309_230971

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 5

theorem avg_rate_of_change :
  (f 0.2 - f 0.1) / (0.2 - 0.1) = 0.9 := by
  sorry

end NUMINAMATH_GPT_avg_rate_of_change_l2309_230971


namespace NUMINAMATH_GPT_base_k_perfect_square_l2309_230972

theorem base_k_perfect_square (k : ℤ) (h : k ≥ 6) : 
  (1 * k^8 + 2 * k^7 + 3 * k^6 + 4 * k^5 + 5 * k^4 + 4 * k^3 + 3 * k^2 + 2 * k + 1) = (k^4 + k^3 + k^2 + k + 1)^2 := 
by
  sorry

end NUMINAMATH_GPT_base_k_perfect_square_l2309_230972


namespace NUMINAMATH_GPT_a_and_b_solution_l2309_230986

noncomputable def solve_for_a_b (a b : ℕ) : Prop :=
  a > 0 ∧ (∀ b : ℤ, b > 0) ∧ (2 * a^b + 16 + 3 * a^b - 8) / 2 = 84 → a = 2 ∧ b = 5

theorem a_and_b_solution (a b : ℕ) (h : solve_for_a_b a b) : a = 2 ∧ b = 5 :=
sorry

end NUMINAMATH_GPT_a_and_b_solution_l2309_230986


namespace NUMINAMATH_GPT_ratio_of_compositions_l2309_230989

def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 2 * x - 3

theorem ratio_of_compositions :
  f (g (f 2)) / g (f (g 2)) = 41 / 7 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_ratio_of_compositions_l2309_230989


namespace NUMINAMATH_GPT_inequality_solution_l2309_230927

noncomputable def inequality (x : ℝ) : Prop :=
  ((x - 1) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0

theorem inequality_solution :
  {x : ℝ | inequality x} = {x : ℝ | x < 1} ∪ {x : ℝ | 2 < x ∧ x < 4} ∪ {x : ℝ | 5 < x ∧ x < 6} ∪ {x : ℝ | 7 < x} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2309_230927


namespace NUMINAMATH_GPT_geometric_sequence_terms_l2309_230973

theorem geometric_sequence_terms
  (a : ℚ) (l : ℚ) (r : ℚ) (n : ℕ)
  (h_a : a = 9 / 8)
  (h_l : l = 1 / 3)
  (h_r : r = 2 / 3)
  (h_geo : l = a * r^(n - 1)) :
  n = 4 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_terms_l2309_230973


namespace NUMINAMATH_GPT_num_good_triples_at_least_l2309_230960

noncomputable def num_good_triples (S : Finset (ℕ × ℕ)) (n m : ℕ) : ℕ :=
  4 * m * (m - n^2 / 4) / (3 * n)

theorem num_good_triples_at_least
  (S : Finset (ℕ × ℕ))
  (n m : ℕ)
  (h_S : ∀ (x : ℕ × ℕ), x ∈ S → 1 ≤ x.1 ∧ x.1 < x.2 ∧ x.2 ≤ n)
  (h_m : S.card = m)
  : ∃ t ≤ num_good_triples S n m, True := 
sorry

end NUMINAMATH_GPT_num_good_triples_at_least_l2309_230960


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2309_230977

theorem simplify_and_evaluate (x : ℝ) (h : x = -2) : (x + 5)^2 - (x - 2) * (x + 2) = 9 :=
by
  rw [h]
  -- Continue with standard proof techniques here
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2309_230977


namespace NUMINAMATH_GPT_mean_steps_per_day_l2309_230923

theorem mean_steps_per_day (total_steps : ℕ) (days_in_april : ℕ) (h_total : total_steps = 243000) (h_days : days_in_april = 30) :
  (total_steps / days_in_april) = 8100 :=
by
  sorry

end NUMINAMATH_GPT_mean_steps_per_day_l2309_230923


namespace NUMINAMATH_GPT_sequence_value_l2309_230981

theorem sequence_value (x : ℕ) : 
  (5 - 2 = 1 * 3) ∧ 
  (11 - 5 = 2 * 3) ∧ 
  (20 - 11 = 3 * 3) ∧ 
  (x - 20 = 4 * 3) ∧ 
  (47 - x = 5 * 3) → 
  x = 32 :=
by 
  intros h 
  sorry

end NUMINAMATH_GPT_sequence_value_l2309_230981


namespace NUMINAMATH_GPT_seq_geom_seq_of_geom_and_arith_l2309_230921

theorem seq_geom_seq_of_geom_and_arith (a : ℕ → ℕ) (b : ℕ → ℕ) 
  (h1 : ∃ a₁ : ℕ, ∀ n : ℕ, a n = a₁ * 2^(n-1))
  (h2 : ∃ b₁ d : ℕ, d = 3 ∧ ∀ n : ℕ, b (n + 1) = b₁ + n * d ∧ b₁ > 0) :
  ∃ r : ℕ, r = 8 ∧ ∃ a₁ : ℕ, ∀ n : ℕ, a (b (n + 1)) = a₁ * r^n :=
by
  sorry

end NUMINAMATH_GPT_seq_geom_seq_of_geom_and_arith_l2309_230921


namespace NUMINAMATH_GPT_gasoline_price_increase_l2309_230982

theorem gasoline_price_increase :
  ∀ (p_low p_high : ℝ), p_low = 14 → p_high = 23 → 
  ((p_high - p_low) / p_low) * 100 = 64.29 :=
by
  intro p_low p_high h_low h_high
  rw [h_low, h_high]
  sorry

end NUMINAMATH_GPT_gasoline_price_increase_l2309_230982


namespace NUMINAMATH_GPT_ceil_sqrt_product_l2309_230903

noncomputable def ceil_sqrt_3 : ℕ := ⌈Real.sqrt 3⌉₊
noncomputable def ceil_sqrt_12 : ℕ := ⌈Real.sqrt 12⌉₊
noncomputable def ceil_sqrt_120 : ℕ := ⌈Real.sqrt 120⌉₊

theorem ceil_sqrt_product :
  ceil_sqrt_3 * ceil_sqrt_12 * ceil_sqrt_120 = 88 :=
by
  sorry

end NUMINAMATH_GPT_ceil_sqrt_product_l2309_230903


namespace NUMINAMATH_GPT_wyatt_envelopes_fewer_l2309_230924

-- Define assets for envelopes
variables (blue_envelopes yellow_envelopes : ℕ)

-- Conditions from the problem
def wyatt_conditions :=
  blue_envelopes = 10 ∧ yellow_envelopes < blue_envelopes ∧ blue_envelopes + yellow_envelopes = 16

-- Theorem: How many fewer yellow envelopes Wyatt has compared to blue envelopes?
theorem wyatt_envelopes_fewer (hb : blue_envelopes = 10) (ht : blue_envelopes + yellow_envelopes = 16) : 
  blue_envelopes - yellow_envelopes = 4 := 
by sorry

end NUMINAMATH_GPT_wyatt_envelopes_fewer_l2309_230924


namespace NUMINAMATH_GPT_number_of_hydrogen_atoms_l2309_230940

theorem number_of_hydrogen_atoms (C_atoms : ℕ) (O_atoms : ℕ) (molecular_weight : ℕ) 
    (C_weight : ℕ) (O_weight : ℕ) (H_weight : ℕ) : C_atoms = 3 → O_atoms = 1 → 
    molecular_weight = 58 → C_weight = 12 → O_weight = 16 → H_weight = 1 → 
    (molecular_weight - (C_atoms * C_weight + O_atoms * O_weight)) / H_weight = 6 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_number_of_hydrogen_atoms_l2309_230940


namespace NUMINAMATH_GPT_smallest_consecutive_integer_l2309_230928

theorem smallest_consecutive_integer (n : ℤ) (h : 7 * n + 21 = 112) : n = 13 :=
sorry

end NUMINAMATH_GPT_smallest_consecutive_integer_l2309_230928


namespace NUMINAMATH_GPT_xiao_dong_not_both_understand_english_and_french_l2309_230919

variables (P Q : Prop)

theorem xiao_dong_not_both_understand_english_and_french (h : ¬ (P ∧ Q)) : P → ¬ Q :=
sorry

end NUMINAMATH_GPT_xiao_dong_not_both_understand_english_and_french_l2309_230919


namespace NUMINAMATH_GPT_pyramid_volume_l2309_230996

theorem pyramid_volume (a : ℝ) (h : a = 2)
  (b : ℝ) (hb : b = 18) :
  ∃ V, V = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_volume_l2309_230996


namespace NUMINAMATH_GPT_price_of_other_frisbees_l2309_230987

-- Lean 4 Statement
theorem price_of_other_frisbees (P : ℝ) (x : ℕ) (h1 : x ≥ 40) (h2 : P * x + 4 * (60 - x) = 200) :
  P = 3 := 
  sorry

end NUMINAMATH_GPT_price_of_other_frisbees_l2309_230987


namespace NUMINAMATH_GPT_x_sq_plus_3x_eq_1_l2309_230951

theorem x_sq_plus_3x_eq_1 (x : ℝ) (h : (x^2 + 3*x)^2 + 2*(x^2 + 3*x) - 3 = 0) : x^2 + 3*x = 1 :=
sorry

end NUMINAMATH_GPT_x_sq_plus_3x_eq_1_l2309_230951


namespace NUMINAMATH_GPT_certain_number_divisibility_l2309_230902

theorem certain_number_divisibility {n : ℕ} (h : ∃ count : ℕ, count = 50 ∧ (count = (300 / (2 * n)))) : n = 3 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_divisibility_l2309_230902


namespace NUMINAMATH_GPT_correct_equation_option_l2309_230901

theorem correct_equation_option :
  (∀ (x : ℝ), (x = 4 → false) ∧ (x = -4 → false)) →
  (∀ (y : ℝ), (y = 12 → true) ∧ (y = -12 → false)) →
  (∀ (z : ℝ), (z = -7 → false) ∧ (z = 7 → true)) →
  (∀ (w : ℝ), (w = 2 → true)) →
  ∃ (option : ℕ), option = 4 := 
by
  sorry

end NUMINAMATH_GPT_correct_equation_option_l2309_230901


namespace NUMINAMATH_GPT_no_solutions_to_equation_l2309_230905

theorem no_solutions_to_equation (a b c : ℤ) : a^2 + b^2 - 8 * c ≠ 6 := 
by 
-- sorry to skip the proof part
sorry

end NUMINAMATH_GPT_no_solutions_to_equation_l2309_230905


namespace NUMINAMATH_GPT_odd_function_has_zero_l2309_230985

variable {R : Type} [LinearOrderedField R]

def is_odd_function (f : R → R) := ∀ x : R, f (-x) = -f x

theorem odd_function_has_zero {f : R → R} (h : is_odd_function f) : ∃ x : R, f x = 0 :=
sorry

end NUMINAMATH_GPT_odd_function_has_zero_l2309_230985


namespace NUMINAMATH_GPT_trains_meet_in_16_67_seconds_l2309_230967

noncomputable def TrainsMeetTime (length1 length2 distance initial_speed1 initial_speed2 : ℝ) : ℝ := 
  let speed1 := initial_speed1 * 1000 / 3600
  let speed2 := initial_speed2 * 1000 / 3600
  let relativeSpeed := speed1 + speed2
  let totalDistance := distance + length1 + length2
  totalDistance / relativeSpeed

theorem trains_meet_in_16_67_seconds : 
  TrainsMeetTime 100 200 450 90 72 = 16.67 := 
by 
  sorry

end NUMINAMATH_GPT_trains_meet_in_16_67_seconds_l2309_230967


namespace NUMINAMATH_GPT_polynomial_expansion_a6_l2309_230904

theorem polynomial_expansion_a6 :
  let p := x^2 + x^7
  ∃ (a : ℕ → ℝ), p = a 0 + a 1 * (x + 1) + a 2 * (x + 1)^2 + a 3 * (x + 1)^3 + a 4 * (x + 1)^4 + a 5 * (x + 1)^5 + a 6 * (x + 1)^6 + a 7 * (x + 1)^7 ∧ a 6 = -7 := 
sorry

end NUMINAMATH_GPT_polynomial_expansion_a6_l2309_230904


namespace NUMINAMATH_GPT_a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l2309_230995

variables (a b c d : ℝ)

-- Given conditions
def first_condition : Prop := a + b = c + d
def second_condition : Prop := a^3 + b^3 = c^3 + d^3

-- Proof problem for part (a)
theorem a_b_fifth_power_equals_c_d_fifth_power 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : a^5 + b^5 = c^5 + d^5 := 
sorry

-- Proof problem for part (b)
theorem cannot_conclude_fourth_powers 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : ¬ (a^4 + b^4 = c^4 + d^4) :=
sorry

end NUMINAMATH_GPT_a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l2309_230995


namespace NUMINAMATH_GPT_altitude_segments_of_acute_triangle_l2309_230911

/-- If two altitudes of an acute triangle divide the sides into segments of lengths 5, 3, 2, and x units,
then x is equal to 10. -/
theorem altitude_segments_of_acute_triangle (a b c d e : ℝ) (h1 : a = 5) (h2 : b = 3) (h3 : c = 2) (h4 : d = x) :
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_altitude_segments_of_acute_triangle_l2309_230911


namespace NUMINAMATH_GPT_no_sequence_of_14_consecutive_divisible_by_some_prime_le_11_l2309_230932

theorem no_sequence_of_14_consecutive_divisible_by_some_prime_le_11 :
  ¬ ∃ n : ℕ, ∀ k : ℕ, k < 14 → ∃ p ∈ [2, 3, 5, 7, 11], (n + k) % p = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_sequence_of_14_consecutive_divisible_by_some_prime_le_11_l2309_230932


namespace NUMINAMATH_GPT_perimeter_of_square_36_l2309_230900

variable (a s P : ℕ)

def is_square_area : Prop := a = s * s
def is_square_perimeter : Prop := P = 4 * s
def condition : Prop := 5 * a = 10 * P + 45

theorem perimeter_of_square_36 (h1 : is_square_area a s) (h2 : is_square_perimeter P s) (h3 : condition a P) : P = 36 := 
by
  sorry

end NUMINAMATH_GPT_perimeter_of_square_36_l2309_230900


namespace NUMINAMATH_GPT_prime_power_divides_power_of_integer_l2309_230948

theorem prime_power_divides_power_of_integer 
    {p a n : ℕ} 
    (hp : Nat.Prime p)
    (ha_pos : 0 < a) 
    (hn_pos : 0 < n) 
    (h : p ∣ a^n) :
    p^n ∣ a^n := 
by 
  sorry

end NUMINAMATH_GPT_prime_power_divides_power_of_integer_l2309_230948


namespace NUMINAMATH_GPT_final_surface_area_l2309_230958

theorem final_surface_area 
  (original_cube_volume : ℕ)
  (small_cube_volume : ℕ)
  (remaining_cubes : ℕ)
  (removed_cubes : ℕ)
  (per_face_expose_area : ℕ)
  (initial_surface_area_per_cube : ℕ)
  (total_cubes : ℕ)
  (shared_internal_faces_area : ℕ)
  (final_surface_area : ℕ) :
  original_cube_volume = 12 * 12 * 12 →
  small_cube_volume = 3 * 3 * 3 →
  total_cubes = 64 →
  removed_cubes = 14 →
  remaining_cubes = total_cubes - removed_cubes →
  initial_surface_area_per_cube = 6 * 3 * 3 →
  per_face_expose_area = 6 * 4 →
  final_surface_area = remaining_cubes * (initial_surface_area_per_cube + per_face_expose_area) - shared_internal_faces_area →
  (remaining_cubes * (initial_surface_area_per_cube + per_face_expose_area) - shared_internal_faces_area) = 2820 :=
sorry

end NUMINAMATH_GPT_final_surface_area_l2309_230958


namespace NUMINAMATH_GPT_ratio_Theresa_Timothy_2010_l2309_230998

def Timothy_movies_2009 : Nat := 24
def Timothy_movies_2010 := Timothy_movies_2009 + 7
def Theresa_movies_2009 := Timothy_movies_2009 / 2
def total_movies := 129
def Timothy_total_movies := Timothy_movies_2009 + Timothy_movies_2010
def Theresa_total_movies := total_movies - Timothy_total_movies
def Theresa_movies_2010 := Theresa_total_movies - Theresa_movies_2009

theorem ratio_Theresa_Timothy_2010 :
  (Theresa_movies_2010 / Timothy_movies_2010) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_Theresa_Timothy_2010_l2309_230998


namespace NUMINAMATH_GPT_remainder_x1002_div_x2_minus_1_mul_x_plus_1_l2309_230936

noncomputable def polynomial_div_remainder (a b : Polynomial ℝ) : Polynomial ℝ := sorry

theorem remainder_x1002_div_x2_minus_1_mul_x_plus_1 :
  polynomial_div_remainder (Polynomial.X ^ 1002) ((Polynomial.X ^ 2 - 1) * (Polynomial.X + 1)) = 1 :=
by sorry

end NUMINAMATH_GPT_remainder_x1002_div_x2_minus_1_mul_x_plus_1_l2309_230936


namespace NUMINAMATH_GPT_smallest_integer_for_inequality_l2309_230916

theorem smallest_integer_for_inequality :
  ∃ x : ℤ, x^2 < 2 * x + 1 ∧ ∀ y : ℤ, y^2 < 2 * y + 1 → x ≤ y := sorry

end NUMINAMATH_GPT_smallest_integer_for_inequality_l2309_230916


namespace NUMINAMATH_GPT_distribute_problems_l2309_230966

theorem distribute_problems :
  let n_problems := 7
  let n_friends := 12
  (n_friends ^ n_problems) = 35831808 :=
by 
  sorry

end NUMINAMATH_GPT_distribute_problems_l2309_230966


namespace NUMINAMATH_GPT_average_salary_proof_l2309_230999

noncomputable def average_salary_of_all_workers (tech_workers : ℕ) (tech_avg_sal : ℕ) (total_workers : ℕ) (non_tech_avg_sal : ℕ) : ℕ :=
  let non_tech_workers := total_workers - tech_workers
  let total_tech_salary := tech_workers * tech_avg_sal
  let total_non_tech_salary := non_tech_workers * non_tech_avg_sal
  let total_salary := total_tech_salary + total_non_tech_salary
  total_salary / total_workers

theorem average_salary_proof : average_salary_of_all_workers 7 14000 28 6000 = 8000 := by
  sorry

end NUMINAMATH_GPT_average_salary_proof_l2309_230999


namespace NUMINAMATH_GPT_repeating_decimal_sum_l2309_230946

noncomputable def repeating_decimal_four : ℚ := 0.44444 -- 0.\overline{4}
noncomputable def repeating_decimal_seven : ℚ := 0.77777 -- 0.\overline{7}

-- Proving that the sum of these repeating decimals is equivalent to the fraction 11/9.
theorem repeating_decimal_sum : repeating_decimal_four + repeating_decimal_seven = 11/9 := by
  -- Placeholder to skip the actual proof
  sorry

end NUMINAMATH_GPT_repeating_decimal_sum_l2309_230946


namespace NUMINAMATH_GPT_tank_capacity_l2309_230975

theorem tank_capacity (C : ℝ) (rate_leak : ℝ) (rate_inlet : ℝ) (combined_rate_empty : ℝ) :
  rate_leak = C / 3 ∧ rate_inlet = 6 * 60 ∧ combined_rate_empty = C / 12 →
  C = 864 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_tank_capacity_l2309_230975


namespace NUMINAMATH_GPT_Jungkook_has_most_apples_l2309_230920

-- Conditions
def Yoongi_apples : ℕ := 4
def Jungkook_apples_initial : ℕ := 6
def Jungkook_apples_additional : ℕ := 3
def Jungkook_total_apples : ℕ := Jungkook_apples_initial + Jungkook_apples_additional
def Yuna_apples : ℕ := 5

-- Statement (to prove)
theorem Jungkook_has_most_apples : Jungkook_total_apples > Yoongi_apples ∧ Jungkook_total_apples > Yuna_apples := by
  sorry

end NUMINAMATH_GPT_Jungkook_has_most_apples_l2309_230920


namespace NUMINAMATH_GPT_cricket_runs_l2309_230993

variable (A B C D E : ℕ)

theorem cricket_runs
  (h1 : (A + B + C + D + E) = 180)
  (h2 : D = E + 5)
  (h3 : A = E + 8)
  (h4 : B = D + E)
  (h5 : B + C = 107) :
  E = 20 := by
  sorry

end NUMINAMATH_GPT_cricket_runs_l2309_230993


namespace NUMINAMATH_GPT_cost_per_scarf_l2309_230910

-- Define the cost of each earring
def cost_of_earring : ℕ := 6000

-- Define the number of earrings
def num_earrings : ℕ := 2

-- Define the cost of the iPhone
def cost_of_iphone : ℕ := 2000

-- Define the number of scarves
def num_scarves : ℕ := 4

-- Define the total value of the swag bag
def total_swag_bag_value : ℕ := 20000

-- Define the total value of diamond earrings and the iPhone
def total_value_of_earrings_and_iphone : ℕ := (num_earrings * cost_of_earring) + cost_of_iphone

-- Define the total value of the scarves
def total_value_of_scarves : ℕ := total_swag_bag_value - total_value_of_earrings_and_iphone

-- Define the cost of each designer scarf
def cost_of_each_scarf : ℕ := total_value_of_scarves / num_scarves

-- Prove that each designer scarf costs $1,500
theorem cost_per_scarf : cost_of_each_scarf = 1500 := by
  sorry

end NUMINAMATH_GPT_cost_per_scarf_l2309_230910


namespace NUMINAMATH_GPT_number_of_real_roots_of_cubic_l2309_230955

-- Define the real number coefficients
variables (a b c d : ℝ)

-- Non-zero condition on coefficients
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

-- Statement of the problem: The cubic polynomial typically has 3 real roots
theorem number_of_real_roots_of_cubic :
  ∃ (x : ℝ), (x ^ 3 + x * (c ^ 2 - d ^ 2 - b * d) - (b ^ 2) * c = 0) := by
  sorry

end NUMINAMATH_GPT_number_of_real_roots_of_cubic_l2309_230955


namespace NUMINAMATH_GPT_complex_mul_example_l2309_230945

theorem complex_mul_example (i : ℝ) (h : i^2 = -1) : (⟨2, 2 * i⟩ : ℂ) * (⟨1, -2 * i⟩) = ⟨6, -2 * i⟩ :=
by
  sorry

end NUMINAMATH_GPT_complex_mul_example_l2309_230945


namespace NUMINAMATH_GPT_prove_equivalence_l2309_230961

variable (x : ℝ)

def operation1 (x : ℝ) : ℝ := 8 - x

def operation2 (x : ℝ) : ℝ := x - 8

theorem prove_equivalence : operation2 (operation1 14) = -14 := by
  sorry

end NUMINAMATH_GPT_prove_equivalence_l2309_230961


namespace NUMINAMATH_GPT_only_n_equal_3_exists_pos_solution_l2309_230984

theorem only_n_equal_3_exists_pos_solution :
  ∀ (n : ℕ), (∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2) ↔ n = 3 := 
by
  sorry

end NUMINAMATH_GPT_only_n_equal_3_exists_pos_solution_l2309_230984


namespace NUMINAMATH_GPT_math_problem_l2309_230906

theorem math_problem : (4 + 6 + 7) * 2 - 2 + (3 / 3) = 33 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l2309_230906


namespace NUMINAMATH_GPT_tim_total_points_l2309_230907

-- Definitions based on the conditions
def points_single : ℕ := 1000
def points_tetris : ℕ := 8 * points_single
def singles_scored : ℕ := 6
def tetrises_scored : ℕ := 4

-- Theorem stating the total points scored by Tim
theorem tim_total_points : singles_scored * points_single + tetrises_scored * points_tetris = 38000 := by
  sorry

end NUMINAMATH_GPT_tim_total_points_l2309_230907


namespace NUMINAMATH_GPT_tangent_line_parabola_l2309_230941

theorem tangent_line_parabola (d : ℝ) : 
    (∀ y : ℝ, (-4)^2 - 4 * (y^2 - 4 * y + 4 * d) = 0) ↔ d = 1 :=
by
    sorry

end NUMINAMATH_GPT_tangent_line_parabola_l2309_230941


namespace NUMINAMATH_GPT_angle_C_is_110_degrees_l2309_230978

def lines_are_parallel (l m : Type) : Prop := sorry
def angle_measure (A : Type) : ℝ := sorry
noncomputable def mangle (C : Type) : ℝ := sorry

theorem angle_C_is_110_degrees 
  (l m C D : Type) 
  (hlm : lines_are_parallel l m)
  (hCDl : lines_are_parallel C l)
  (hCDm : lines_are_parallel C m)
  (hA : angle_measure A = 100)
  (hB : angle_measure B = 150) :
  mangle C = 110 :=
by
  sorry

end NUMINAMATH_GPT_angle_C_is_110_degrees_l2309_230978


namespace NUMINAMATH_GPT_partial_fraction_decomposition_l2309_230994

theorem partial_fraction_decomposition (A B C : ℚ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
    (-x^2 + 5*x - 6) / (x^3 - x) = A / x + (B*x + C) / (x^2 - 1)) →
  A = 6 ∧ B = -7 ∧ C = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_partial_fraction_decomposition_l2309_230994


namespace NUMINAMATH_GPT_children_of_exceptions_l2309_230983

theorem children_of_exceptions (x y : ℕ) (h : 6 * x + 2 * y = 58) (hx : x = 8) : y = 5 :=
by
  sorry

end NUMINAMATH_GPT_children_of_exceptions_l2309_230983


namespace NUMINAMATH_GPT_carnival_candies_l2309_230969

theorem carnival_candies :
  ∃ (c : ℕ), c % 5 = 4 ∧ c % 6 = 3 ∧ c % 8 = 5 ∧ c < 150 ∧ c = 69 :=
by
  sorry

end NUMINAMATH_GPT_carnival_candies_l2309_230969


namespace NUMINAMATH_GPT_find_real_number_l2309_230992

theorem find_real_number (x : ℝ) (h1 : 0 < x) (h2 : ⌊x⌋ * x = 72) : x = 9 :=
sorry

end NUMINAMATH_GPT_find_real_number_l2309_230992


namespace NUMINAMATH_GPT_second_chapter_pages_is_80_l2309_230933

def first_chapter_pages : ℕ := 37
def second_chapter_pages : ℕ := first_chapter_pages + 43

theorem second_chapter_pages_is_80 : second_chapter_pages = 80 :=
by
  sorry

end NUMINAMATH_GPT_second_chapter_pages_is_80_l2309_230933


namespace NUMINAMATH_GPT_gcd_5039_3427_l2309_230988

def a : ℕ := 5039
def b : ℕ := 3427

theorem gcd_5039_3427 : Nat.gcd a b = 7 := by
  sorry

end NUMINAMATH_GPT_gcd_5039_3427_l2309_230988


namespace NUMINAMATH_GPT_negation_of_p_negation_of_q_l2309_230931

def p (x : ℝ) : Prop := x > 0 → x^2 - 5 * x ≥ -25 / 4

def even (n : ℕ) : Prop := ∃ k, n = 2 * k

def q : Prop := ∃ n, even n ∧ ∃ m, n = 3 * m

theorem negation_of_p : ¬(∀ x : ℝ, x > 0 → x^2 - 5 * x ≥ - 25 / 4) → ∃ x : ℝ, x > 0 ∧ x^2 - 5 * x < - 25 / 4 := 
by sorry

theorem negation_of_q : ¬ (∃ n : ℕ, even n ∧ ∃ m : ℕ, n = 3 * m) → ∀ n : ℕ, even n → ¬ (∃ m : ℕ, n = 3 * m) := 
by sorry

end NUMINAMATH_GPT_negation_of_p_negation_of_q_l2309_230931


namespace NUMINAMATH_GPT_geometric_sequence_problem_l2309_230937

theorem geometric_sequence_problem
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : q ≠ 1)
  (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 6)
  (h_sum_squares : a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 + a 5 ^ 2 + a 6 ^ 2 + a 7 ^ 2 = 18)
  (h_geom_seq : ∀ n : ℕ, a (n + 1) = a 1 * q ^ n) :
  a 1 - a 2 + a 3 - a 4 + a 5 - a 6 + a 7 = 3 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l2309_230937


namespace NUMINAMATH_GPT_club_boys_count_l2309_230908

theorem club_boys_count (B G : ℕ) (h1 : B + G = 30) (h2 : (1 / 3 : ℝ) * G + B = 18) : B = 12 :=
by
  -- We would proceed with the steps here, but add 'sorry' to indicate incomplete proof
  sorry

end NUMINAMATH_GPT_club_boys_count_l2309_230908


namespace NUMINAMATH_GPT_find_point_coordinates_l2309_230912

theorem find_point_coordinates (P : ℝ × ℝ)
  (h1 : P.1 < 0) -- Point P is in the second quadrant, so x < 0
  (h2 : P.2 > 0) -- Point P is in the second quadrant, so y > 0
  (h3 : abs P.2 = 4) -- distance from P to x-axis is 4
  (h4 : abs P.1 = 5) -- distance from P to y-axis is 5
  : P = (-5, 4) :=
by {
  -- point P is in the second quadrant, so x < 0 and y > 0
  -- |y| = 4 -> y = 4 
  -- |x| = 5 -> x = -5
  sorry
}

end NUMINAMATH_GPT_find_point_coordinates_l2309_230912


namespace NUMINAMATH_GPT_number_of_arrangements_l2309_230976

noncomputable def arrangements_nonadjacent_teachers (A : ℕ → ℕ → ℕ) : ℕ :=
  let students_arrangements := A 8 8
  let gaps_count := 9
  let teachers_arrangements := A gaps_count 2
  students_arrangements * teachers_arrangements

theorem number_of_arrangements (A : ℕ → ℕ → ℕ) :
  arrangements_nonadjacent_teachers A = A 8 8 * A 9 2 := 
  sorry

end NUMINAMATH_GPT_number_of_arrangements_l2309_230976


namespace NUMINAMATH_GPT_find_percentage_find_percentage_as_a_percentage_l2309_230938

variable (P : ℝ)

theorem find_percentage (h : P / 2 = 0.02) : P = 0.04 :=
by
  sorry

theorem find_percentage_as_a_percentage (h : P / 2 = 0.02) : P = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_find_percentage_as_a_percentage_l2309_230938


namespace NUMINAMATH_GPT_solve_fraction_eq_zero_l2309_230944

theorem solve_fraction_eq_zero (x : ℝ) (h : (x - 3) / (2 * x + 5) = 0) (h2 : 2 * x + 5 ≠ 0) : x = 3 :=
sorry

end NUMINAMATH_GPT_solve_fraction_eq_zero_l2309_230944


namespace NUMINAMATH_GPT_find_triangle_sides_l2309_230953

-- Define the conditions and translate them into Lean 4
theorem find_triangle_sides :
  (∃ a b c: ℝ, a + b + c = 40 ∧ a^2 + b^2 = c^2 ∧ 
   (a + 4)^2 + (b + 1)^2 = (c + 3)^2 ∧ 
   a = 8 ∧ b = 15 ∧ c = 17) :=
by 
  sorry

end NUMINAMATH_GPT_find_triangle_sides_l2309_230953


namespace NUMINAMATH_GPT_number_of_keyboards_l2309_230930

-- Definitions based on conditions
def keyboard_cost : ℕ := 20
def printer_cost : ℕ := 70
def printers_bought : ℕ := 25
def total_cost : ℕ := 2050

-- The variable we want to prove
variable (K : ℕ)

-- The main theorem statement
theorem number_of_keyboards (K : ℕ) (keyboard_cost printer_cost printers_bought total_cost : ℕ) :
  keyboard_cost * K + printer_cost * printers_bought = total_cost → K = 15 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_number_of_keyboards_l2309_230930


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_add_1_l2309_230914

def q (x : ℝ) : ℝ := 2 * x^4 - 3 * x^3 + 4 * x^2 - 5 * x + 6

theorem remainder_when_divided_by_x_add_1 :
  q 2 = 6 → q (-1) = 20 :=
by
  intro hq2
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_add_1_l2309_230914


namespace NUMINAMATH_GPT_inequality_solution_l2309_230926

theorem inequality_solution (a x : ℝ) (h₁ : 0 < a) : 
  (0 < a ∧ a < 1 → 2 < x ∧ x < (a-2)/(a-1) → (a * (x - 1)) / (x-2) > 1) ∧ 
  (a = 1 → 2 < x → (a * (x - 1)) / (x-2) > 1 ∧ true) ∧ 
  (a > 1 → (2 < x ∨ x < (a-2)/(a-1)) → (a * (x - 1)) / (x-2) > 1) := 
sorry

end NUMINAMATH_GPT_inequality_solution_l2309_230926


namespace NUMINAMATH_GPT_no_valid_pairs_l2309_230949

/-- 
Statement: There are no pairs of positive integers (a, b) such that
a * b + 100 = 25 * lcm(a, b) + 15 * gcd(a, b).
-/
theorem no_valid_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  a * b + 100 ≠ 25 * Nat.lcm a b + 15 * Nat.gcd a b :=
sorry

end NUMINAMATH_GPT_no_valid_pairs_l2309_230949


namespace NUMINAMATH_GPT_seat_to_right_proof_l2309_230935

def Xiaofang_seat : ℕ × ℕ := (3, 5)

def seat_to_right (seat : ℕ × ℕ) : ℕ × ℕ :=
  (seat.1 + 1, seat.2)

theorem seat_to_right_proof : seat_to_right Xiaofang_seat = (4, 5) := by
  unfold Xiaofang_seat
  unfold seat_to_right
  sorry

end NUMINAMATH_GPT_seat_to_right_proof_l2309_230935
