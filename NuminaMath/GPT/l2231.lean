import Mathlib

namespace NUMINAMATH_GPT_experts_win_probability_l2231_223143

noncomputable def probability_of_experts_winning (p : ℝ) (q : ℝ) (needed_expert_wins : ℕ) (needed_audience_wins : ℕ) : ℝ :=
  p ^ 4 + 4 * (p ^ 3 * q)

-- Probability values
def p : ℝ := 0.6
def q : ℝ := 1 - p

-- Number of wins needed
def needed_expert_wins : ℕ := 3
def needed_audience_wins : ℕ := 2

theorem experts_win_probability :
  probability_of_experts_winning p q needed_expert_wins needed_audience_wins = 0.4752 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_experts_win_probability_l2231_223143


namespace NUMINAMATH_GPT_find_n_in_range_l2231_223138

theorem find_n_in_range : ∃ n, 5 ≤ n ∧ n ≤ 10 ∧ n ≡ 10543 [MOD 7] ∧ n = 8 := 
by
  sorry

end NUMINAMATH_GPT_find_n_in_range_l2231_223138


namespace NUMINAMATH_GPT_max_intersections_cos_circle_l2231_223142

theorem max_intersections_cos_circle :
  let circle := λ x y => (x - 4)^2 + y^2 = 25
  let cos_graph := λ x => (x, Real.cos x)
  ∀ x y, (circle x y ∧ y = Real.cos x) → (∃ (p : ℕ), p ≤ 8) := sorry

end NUMINAMATH_GPT_max_intersections_cos_circle_l2231_223142


namespace NUMINAMATH_GPT_triangle_side_length_condition_l2231_223146

theorem triangle_side_length_condition (a : ℝ) (h₁ : a > 0) (h₂ : a + 2 > a + 5) (h₃ : a + 5 > a + 2) (h₄ : a + 2 + a + 5 > a) : a > 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_condition_l2231_223146


namespace NUMINAMATH_GPT_even_binomial_coefficients_l2231_223155

theorem even_binomial_coefficients (n : ℕ) (h_pos: 0 < n) : 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 → 2 ∣ Nat.choose n k) ↔ ∃ k : ℕ, n = 2^k :=
by
  sorry

end NUMINAMATH_GPT_even_binomial_coefficients_l2231_223155


namespace NUMINAMATH_GPT_speed_conversion_l2231_223133

theorem speed_conversion (s : ℚ) (h : s = 13 / 48) : 
  ((13 / 48) * 3.6 = 0.975) :=
by
  sorry

end NUMINAMATH_GPT_speed_conversion_l2231_223133


namespace NUMINAMATH_GPT_find_a_l2231_223117

noncomputable def point1 : ℝ × ℝ := (-3, 6)
noncomputable def point2 : ℝ × ℝ := (2, -1)

theorem find_a (a : ℝ) :
  let direction : ℝ × ℝ := (point2.1 - point1.1, point2.2 - point1.2)
  direction = (5, -7) →
  let normalized_direction : ℝ × ℝ := (direction.1 / -7, direction.2 / -7)
  normalized_direction = (a, -1) →
  a = -5 / 7 :=
by 
  intros 
  sorry

end NUMINAMATH_GPT_find_a_l2231_223117


namespace NUMINAMATH_GPT_downstream_speed_is_40_l2231_223137

variable (Vu : ℝ) (Vs : ℝ) (Vd : ℝ)

theorem downstream_speed_is_40 (h1 : Vu = 26) (h2 : Vs = 33) :
  Vd = 40 :=
by
  sorry

end NUMINAMATH_GPT_downstream_speed_is_40_l2231_223137


namespace NUMINAMATH_GPT_M_inter_N_eq_interval_l2231_223171

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

theorem M_inter_N_eq_interval : (M ∩ N) = {x | 0 ≤ x ∧ x < 1} := 
  sorry

end NUMINAMATH_GPT_M_inter_N_eq_interval_l2231_223171


namespace NUMINAMATH_GPT_max_min_values_of_g_l2231_223175

noncomputable def g (x : ℝ) : ℝ := (Real.sin x)^8 + 8 * (Real.cos x)^8

theorem max_min_values_of_g :
  (∀ x : ℝ, g x ≤ 8) ∧ (∀ x : ℝ, g x ≥ 8 / 27) :=
by
  sorry

end NUMINAMATH_GPT_max_min_values_of_g_l2231_223175


namespace NUMINAMATH_GPT_Einstein_sold_25_cans_of_soda_l2231_223165

def sell_snacks_proof : Prop :=
  let pizza_price := 12
  let fries_price := 0.30
  let soda_price := 2
  let goal := 500
  let pizza_boxes := 15
  let fries_packs := 40
  let still_needed := 258
  let earned_from_pizza := pizza_boxes * pizza_price
  let earned_from_fries := fries_packs * fries_price
  let total_earned := earned_from_pizza + earned_from_fries
  let total_have := goal - still_needed
  let earned_from_soda := total_have - total_earned
  let cans_of_soda_sold := earned_from_soda / soda_price
  cans_of_soda_sold = 25

theorem Einstein_sold_25_cans_of_soda : sell_snacks_proof := by
  sorry

end NUMINAMATH_GPT_Einstein_sold_25_cans_of_soda_l2231_223165


namespace NUMINAMATH_GPT_find_m_range_l2231_223147

variable {R : Type*} [LinearOrderedField R]
variable (f : R → R)
variable (m : R)

-- Define that the function f is monotonically increasing
def monotonically_increasing (f : R → R) : Prop :=
  ∀ ⦃x y : R⦄, x ≤ y → f x ≤ f y

-- Lean statement for the proof problem
theorem find_m_range (h1 : monotonically_increasing f) (h2 : f (2 * m - 3) > f (-m)) : m > 1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_range_l2231_223147


namespace NUMINAMATH_GPT_remainder_count_l2231_223191

theorem remainder_count (n : ℕ) (h : n > 5) : 
  ∃ l : List ℕ, l.length = 5 ∧ ∀ x ∈ l, x ∣ 42 ∧ x > 5 := 
sorry

end NUMINAMATH_GPT_remainder_count_l2231_223191


namespace NUMINAMATH_GPT_rectangle_pairs_l2231_223132

theorem rectangle_pairs :
  {p : ℕ × ℕ | 0 < p.1 ∧ 0 < p.2 ∧ p.1 * p.2 = 18} = {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)} :=
by { sorry }

end NUMINAMATH_GPT_rectangle_pairs_l2231_223132


namespace NUMINAMATH_GPT_largest_of_eight_consecutive_integers_l2231_223100

theorem largest_of_eight_consecutive_integers (n : ℕ) 
  (h : 8 * n + 28 = 3652) : n + 7 = 460 := by 
  sorry

end NUMINAMATH_GPT_largest_of_eight_consecutive_integers_l2231_223100


namespace NUMINAMATH_GPT_find_percentage_l2231_223134

variable (X P : ℝ)

theorem find_percentage (h₁ : 0.20 * X = 400) (h₂ : (P / 100) * X = 2400) : P = 120 :=
by
  -- The proof is intentionally left out
  sorry

end NUMINAMATH_GPT_find_percentage_l2231_223134


namespace NUMINAMATH_GPT_angle_same_terminal_side_l2231_223119

theorem angle_same_terminal_side (α : ℝ) : 
  (∃ k : ℤ, α = k * 360 - 100) ↔ (∃ k : ℤ, α = k * 360 + (-100)) :=
sorry

end NUMINAMATH_GPT_angle_same_terminal_side_l2231_223119


namespace NUMINAMATH_GPT_coupon_savings_inequalities_l2231_223198

variable {P : ℝ} (p : ℝ) (hP : P = 150 + p) (hp_pos : p > 0)
variable (ha : 0.15 * P > 30) (hb : 0.15 * P > 0.20 * p)
variable (cA_saving : ℝ := 0.15 * P)
variable (cB_saving : ℝ := 30)
variable (cC_saving : ℝ := 0.20 * p)

theorem coupon_savings_inequalities (h1 : 0.15 * P - 30 > 0) (h2 : 0.15 * P - 0.20 * (P - 150) > 0) :
  let x := 200
  let y := 600
  y - x = 400 :=
by
  sorry

end NUMINAMATH_GPT_coupon_savings_inequalities_l2231_223198


namespace NUMINAMATH_GPT_probability_A_not_losing_l2231_223183

theorem probability_A_not_losing (P_draw P_win : ℚ) (h1 : P_draw = 1/2) (h2 : P_win = 1/3) : 
  P_draw + P_win = 5/6 :=
by
  sorry

end NUMINAMATH_GPT_probability_A_not_losing_l2231_223183


namespace NUMINAMATH_GPT_eval_expr_l2231_223185

theorem eval_expr : 8^3 + 8^3 + 8^3 + 8^3 - 2^6 * 2^3 = 1536 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_eval_expr_l2231_223185


namespace NUMINAMATH_GPT_min_distance_exists_l2231_223105

open Real

-- Define the distance formula function
noncomputable def distance (x : ℝ) : ℝ :=
sqrt ((x - 1) ^ 2 + (3 - 2 * x) ^ 2 + (3 * x - 3) ^ 2)

theorem min_distance_exists :
  ∃ (x : ℝ), distance x = sqrt (14 * x^2 - 32 * x + 19) ∧
               ∀ y, distance y ≥ (sqrt 35) / 7 :=
sorry

end NUMINAMATH_GPT_min_distance_exists_l2231_223105


namespace NUMINAMATH_GPT_P_work_time_l2231_223163

theorem P_work_time (T : ℝ) (hT : T > 0) : 
  (1 / T + 1 / 6 = 1 / 2.4) → T = 4 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_P_work_time_l2231_223163


namespace NUMINAMATH_GPT_parallelogram_side_sum_l2231_223189

variable (x y : ℚ)

theorem parallelogram_side_sum :
  4 * x - 1 = 10 →
  5 * y + 3 = 12 →
  x + y = 91 / 20 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_parallelogram_side_sum_l2231_223189


namespace NUMINAMATH_GPT_difference_between_multiplication_and_subtraction_l2231_223127

theorem difference_between_multiplication_and_subtraction (x : ℤ) (h1 : x = 11) :
  (3 * x) - (26 - x) = 18 := by
  sorry

end NUMINAMATH_GPT_difference_between_multiplication_and_subtraction_l2231_223127


namespace NUMINAMATH_GPT_train_length_l2231_223199

theorem train_length (L S : ℝ) 
  (h1 : L = S * 40) 
  (h2 : L + 1800 = S * 120) : 
  L = 900 := 
by
  sorry

end NUMINAMATH_GPT_train_length_l2231_223199


namespace NUMINAMATH_GPT_prime_factors_sum_l2231_223128

theorem prime_factors_sum (w x y z t : ℕ) 
  (h : 2^w * 3^x * 5^y * 7^z * 11^t = 107100) : 
  2 * w + 3 * x + 5 * y + 7 * z + 11 * t = 38 :=
sorry

end NUMINAMATH_GPT_prime_factors_sum_l2231_223128


namespace NUMINAMATH_GPT_jason_seashells_remaining_l2231_223124

-- Define the initial number of seashells Jason found
def initial_seashells : ℕ := 49

-- Define the number of seashells Jason gave to Tim
def seashells_given_to_tim : ℕ := 13

-- Define the number of seashells Jason now has
def seashells_now : ℕ := initial_seashells - seashells_given_to_tim

-- The theorem to prove: 
theorem jason_seashells_remaining : seashells_now = 36 := 
by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_jason_seashells_remaining_l2231_223124


namespace NUMINAMATH_GPT_andrey_boris_denis_eat_candies_l2231_223112

def andrey_boris_condition (a b : ℕ) : Prop :=
  a = 4 ∧ b = 3

def andrey_denis_condition (a d : ℕ) : Prop :=
  a = 6 ∧ d = 7

def total_candies_condition (total : ℕ) : Prop :=
  total = 70

theorem andrey_boris_denis_eat_candies :
  ∃ (a b d : ℕ), andrey_boris_condition a b ∧ andrey_denis_condition a d ∧ 
                  (total_candies_condition (2 * (12 + 9 + 14)) ∧ 
                   2 * 12 = 24 ∧ 2 * 9 = 18 ∧ 2 * 14 = 28) →
                  (a = 24 ∧ b = 18 ∧ d = 28) :=
by
  sorry

end NUMINAMATH_GPT_andrey_boris_denis_eat_candies_l2231_223112


namespace NUMINAMATH_GPT_program_final_value_l2231_223130

-- Define the program execution in a Lean function
def program_result (i : ℕ) (S : ℕ) : ℕ :=
  if i < 9 then S
  else program_result (i - 1) (S * i)

-- Initial conditions
def initial_i := 11
def initial_S := 1

-- The theorem to prove
theorem program_final_value : program_result initial_i initial_S = 990 := by
  sorry

end NUMINAMATH_GPT_program_final_value_l2231_223130


namespace NUMINAMATH_GPT_cows_horses_ratio_l2231_223190

theorem cows_horses_ratio (cows horses : ℕ) (h : cows = 21) (ratio : cows / horses = 7 / 2) : horses = 6 :=
sorry

end NUMINAMATH_GPT_cows_horses_ratio_l2231_223190


namespace NUMINAMATH_GPT_original_time_taken_by_bullet_train_is_50_minutes_l2231_223162

-- Define conditions as assumptions
variables (T D : ℝ) (h0 : D = 48 * T) (h1 : D = 60 * (40 / 60))

-- Define the theorem we want to prove
theorem original_time_taken_by_bullet_train_is_50_minutes :
  T = 50 / 60 :=
by
  sorry

end NUMINAMATH_GPT_original_time_taken_by_bullet_train_is_50_minutes_l2231_223162


namespace NUMINAMATH_GPT_perimeter_shaded_area_is_942_l2231_223151

-- Definition involving the perimeter of the shaded area of the circles
noncomputable def perimeter_shaded_area (s : ℝ) : ℝ := 
  4 * 75 * 3.14

-- Main theorem stating that if the side length of the octagon is 100 cm,
-- then the perimeter of the shaded area is 942 cm.
theorem perimeter_shaded_area_is_942 :
  perimeter_shaded_area 100 = 942 := 
  sorry

end NUMINAMATH_GPT_perimeter_shaded_area_is_942_l2231_223151


namespace NUMINAMATH_GPT_sequence_x_value_l2231_223173

theorem sequence_x_value (x : ℕ) (h1 : 3 - 1 = 2) (h2 : 6 - 3 = 3) (h3 : 10 - 6 = 4) (h4 : x - 10 = 5) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_sequence_x_value_l2231_223173


namespace NUMINAMATH_GPT_range_of_x_l2231_223180

theorem range_of_x (x y : ℝ) (h1 : 4 * x + y = 3) (h2 : -2 < y ∧ y ≤ 7) : 1 ≤ x ∧ x < 5 / 4 := 
  sorry

end NUMINAMATH_GPT_range_of_x_l2231_223180


namespace NUMINAMATH_GPT_stratified_sampling_total_sample_size_l2231_223107

-- Definitions based on conditions
def pure_milk_brands : ℕ := 30
def yogurt_brands : ℕ := 10
def infant_formula_brands : ℕ := 35
def adult_milk_powder_brands : ℕ := 25
def sampled_infant_formula_brands : ℕ := 7

-- The goal is to prove that the total sample size n is 20.
theorem stratified_sampling_total_sample_size : 
  let total_brands := pure_milk_brands + yogurt_brands + infant_formula_brands + adult_milk_powder_brands
  let sampling_fraction := sampled_infant_formula_brands / infant_formula_brands
  let pure_milk_samples := pure_milk_brands * sampling_fraction
  let yogurt_samples := yogurt_brands * sampling_fraction
  let adult_milk_samples := adult_milk_powder_brands * sampling_fraction
  let n := pure_milk_samples + yogurt_samples + sampled_infant_formula_brands + adult_milk_samples
  n = 20 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_total_sample_size_l2231_223107


namespace NUMINAMATH_GPT_second_field_full_rows_l2231_223135

theorem second_field_full_rows 
    (rows_field1 : ℕ) (cobs_per_row : ℕ) (total_cobs : ℕ)
    (H1 : rows_field1 = 13)
    (H2 : cobs_per_row = 4)
    (H3 : total_cobs = 116) : 
    (total_cobs - rows_field1 * cobs_per_row) / cobs_per_row = 16 :=
by sorry

end NUMINAMATH_GPT_second_field_full_rows_l2231_223135


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l2231_223122

def sequence_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

def abs_condition (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > abs (a n)

theorem sufficient_but_not_necessary (a : ℕ → ℝ) :
  (abs_condition a → sequence_increasing a) ∧ ¬ (sequence_increasing a → abs_condition a) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l2231_223122


namespace NUMINAMATH_GPT_problem_statement_l2231_223186

-- Define a : ℝ such that (a + 1/a)^3 = 7
variables (a : ℝ) (h : (a + 1/a)^3 = 7)

-- Goal: Prove that a^4 + 1/a^4 = 1519/81
theorem problem_statement (a : ℝ) (h : (a + 1/a)^3 = 7) : a^4 + 1/a^4 = 1519 / 81 := 
sorry

end NUMINAMATH_GPT_problem_statement_l2231_223186


namespace NUMINAMATH_GPT_complex_quadrant_l2231_223188

theorem complex_quadrant (z : ℂ) (h : (2 - I) * z = 1 + I) : 
  0 < z.re ∧ 0 < z.im := 
by 
  -- Proof will be provided here 
  sorry

end NUMINAMATH_GPT_complex_quadrant_l2231_223188


namespace NUMINAMATH_GPT_factorize_expression_l2231_223149

theorem factorize_expression (x y : ℝ) : x^2 - 1 + 2 * x * y + y^2 = (x + y + 1) * (x + y - 1) :=
by sorry

end NUMINAMATH_GPT_factorize_expression_l2231_223149


namespace NUMINAMATH_GPT_tan_value_l2231_223113

theorem tan_value (α : ℝ) (h1 : α ∈ (Set.Ioo (π/2) π)) (h2 : Real.sin α = 4/5) : Real.tan α = -4/3 :=
sorry

end NUMINAMATH_GPT_tan_value_l2231_223113


namespace NUMINAMATH_GPT_area_BEIH_l2231_223148

noncomputable def point (x y : ℝ) : ℝ × ℝ := (x, y)

noncomputable def area_quad (B E I H : ℝ × ℝ) : ℝ :=
  (1/2) * ((B.1 * E.2 + E.1 * I.2 + I.1 * H.2 + H.1 * B.2) - (B.2 * E.1 + E.2 * I.1 + I.2 * H.1 + H.2 * B.1))

theorem area_BEIH :
  let A : ℝ × ℝ := point 0 3
  let B : ℝ × ℝ := point 0 0
  let C : ℝ × ℝ := point 3 0
  let D : ℝ × ℝ := point 3 3
  let E : ℝ × ℝ := point 0 2
  let F : ℝ × ℝ := point 1 0
  let I : ℝ × ℝ := point (3/10) 2.1
  let H : ℝ × ℝ := point (3/4) (3/4)
  area_quad B E I H = 1.0125 :=
by
  sorry

end NUMINAMATH_GPT_area_BEIH_l2231_223148


namespace NUMINAMATH_GPT_infinite_n_divisible_by_p_l2231_223103

theorem infinite_n_divisible_by_p (p : ℕ) (hp : Nat.Prime p) : 
  ∃ᶠ n in Filter.atTop, p ∣ (2^n - n) :=
by
  sorry

end NUMINAMATH_GPT_infinite_n_divisible_by_p_l2231_223103


namespace NUMINAMATH_GPT_uniq_increasing_seq_l2231_223140

noncomputable def a (n : ℕ) : ℕ := n -- The correct sequence a_n = n

theorem uniq_increasing_seq (a : ℕ → ℕ)
  (h1 : a 2 = 2)
  (h2 : ∀ n m : ℕ, a (n * m) = a n * a m)
  (h_inc : ∀ n m : ℕ, n < m → a n < a m) : ∀ n : ℕ, a n = n := by
  -- Here we would place the proof, skipping it for now with sorry
  sorry

end NUMINAMATH_GPT_uniq_increasing_seq_l2231_223140


namespace NUMINAMATH_GPT_proof_minimum_value_l2231_223125

noncomputable def minimum_value_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : Prop :=
  (1 / a + a / b) ≥ 1 + 2 * Real.sqrt 2

theorem proof_minimum_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : minimum_value_inequality a b h1 h2 h3 :=
  by
    sorry

end NUMINAMATH_GPT_proof_minimum_value_l2231_223125


namespace NUMINAMATH_GPT_milk_water_ratio_l2231_223160

theorem milk_water_ratio (x y : ℝ) (h1 : 5 * x + 2 * y = 4 * x + 7 * y) :
  x / y = 5 :=
by 
  sorry

end NUMINAMATH_GPT_milk_water_ratio_l2231_223160


namespace NUMINAMATH_GPT_find_f_1002_l2231_223177

noncomputable def f : ℕ → ℝ :=
  sorry

theorem find_f_1002 (f : ℕ → ℝ) 
  (h : ∀ a b n : ℕ, a + b = 2^n → f a + f b = n^2) :
  f 1002 = 21 :=
sorry

end NUMINAMATH_GPT_find_f_1002_l2231_223177


namespace NUMINAMATH_GPT_perfect_square_values_l2231_223150

theorem perfect_square_values :
  ∀ n : ℕ, 0 < n → (∃ k : ℕ, (n^2 + 11 * n - 4) * n.factorial + 33 * 13^n + 4 = k^2) ↔ n = 1 ∨ n = 2 :=
by sorry

end NUMINAMATH_GPT_perfect_square_values_l2231_223150


namespace NUMINAMATH_GPT_georgina_parrot_days_l2231_223109

theorem georgina_parrot_days
  (total_phrases : ℕ)
  (phrases_per_week : ℕ)
  (initial_phrases : ℕ)
  (phrases_now : total_phrases = 17)
  (teaching_rate : phrases_per_week = 2)
  (initial_known : initial_phrases = 3) :
  (49 : ℕ) = (((17 - 3) / 2) * 7) :=
by
  -- proof will be here
  sorry

end NUMINAMATH_GPT_georgina_parrot_days_l2231_223109


namespace NUMINAMATH_GPT_sufficient_not_necessary_l2231_223129

theorem sufficient_not_necessary (a b : ℝ) :
  (a^2 + b^2 = 0 → ab = 0) ∧ (ab = 0 → ¬(a^2 + b^2 = 0)) := 
by
  have h1 : (a^2 + b^2 = 0 → ab = 0) := sorry
  have h2 : (ab = 0 → ¬(a^2 + b^2 = 0)) := sorry
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_sufficient_not_necessary_l2231_223129


namespace NUMINAMATH_GPT_liz_spent_total_l2231_223161

-- Definitions:
def recipe_book_cost : ℕ := 6
def baking_dish_cost : ℕ := 2 * recipe_book_cost
def ingredient_cost : ℕ := 3
def number_of_ingredients : ℕ := 5
def apron_cost : ℕ := recipe_book_cost + 1

-- Total cost calculation:
def total_cost : ℕ :=
  recipe_book_cost + baking_dish_cost + (number_of_ingredients * ingredient_cost) + apron_cost

-- Theorem Statement:
theorem liz_spent_total : total_cost = 40 := by
  sorry

end NUMINAMATH_GPT_liz_spent_total_l2231_223161


namespace NUMINAMATH_GPT_minimum_dwarfs_l2231_223111

theorem minimum_dwarfs (n : ℕ) (C : ℕ → Prop) (h_nonempty : ∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :
  ∃ m, 10 ≤ m ∧ (∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :=
sorry

end NUMINAMATH_GPT_minimum_dwarfs_l2231_223111


namespace NUMINAMATH_GPT_average_monthly_sales_booster_club_l2231_223174

noncomputable def monthly_sales : List ℕ := [80, 100, 75, 95, 110, 180, 90, 115, 130, 200, 160, 140]

noncomputable def average_sales (sales : List ℕ) : ℝ :=
  (sales.foldr (λ x acc => x + acc) 0 : ℕ) / sales.length

theorem average_monthly_sales_booster_club : average_sales monthly_sales = 122.92 := by
  sorry

end NUMINAMATH_GPT_average_monthly_sales_booster_club_l2231_223174


namespace NUMINAMATH_GPT_max_value_g_eq_3_in_interval_l2231_223166

noncomputable def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_value_g_eq_3_in_interval : 
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → g x ≤ 3) ∧ (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ g x = 3) :=
by
  sorry

end NUMINAMATH_GPT_max_value_g_eq_3_in_interval_l2231_223166


namespace NUMINAMATH_GPT_units_digit_of_product_of_seven_consecutive_l2231_223193

theorem units_digit_of_product_of_seven_consecutive (n : ℕ) : 
  ∃ d ∈ [n, n+1, n+2, n+3, n+4, n+5, n+6], d % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_product_of_seven_consecutive_l2231_223193


namespace NUMINAMATH_GPT_num_ballpoint_pens_l2231_223179

-- Define the total number of school supplies
def total_school_supplies : ℕ := 60

-- Define the number of pencils
def num_pencils : ℕ := 5

-- Define the number of notebooks
def num_notebooks : ℕ := 10

-- Define the number of erasers
def num_erasers : ℕ := 32

-- Define the number of ballpoint pens and prove it equals 13
theorem num_ballpoint_pens : total_school_supplies - (num_pencils + num_notebooks + num_erasers) = 13 :=
by
sorry

end NUMINAMATH_GPT_num_ballpoint_pens_l2231_223179


namespace NUMINAMATH_GPT_sum_eq_product_l2231_223120

theorem sum_eq_product (a b c : ℝ) (h1 : 1 + b * c ≠ 0) (h2 : 1 + c * a ≠ 0) (h3 : 1 + a * b ≠ 0) :
  (b - c) / (1 + b * c) + (c - a) / (1 + c * a) + (a - b) / (1 + a * b) =
  ((b - c) * (c - a) * (a - b)) / ((1 + b * c) * (1 + c * a) * (1 + a * b)) :=
by
  sorry

end NUMINAMATH_GPT_sum_eq_product_l2231_223120


namespace NUMINAMATH_GPT_polynomial_roots_l2231_223131

theorem polynomial_roots :
  (∀ x : ℤ, (x^3 - 4*x^2 - 11*x + 24 = 0) ↔ (x = 4 ∨ x = 3 ∨ x = -1)) :=
sorry

end NUMINAMATH_GPT_polynomial_roots_l2231_223131


namespace NUMINAMATH_GPT_small_branches_per_branch_l2231_223106

theorem small_branches_per_branch (x : ℕ) (h1 : 1 + x + x^2 = 57) : x = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_small_branches_per_branch_l2231_223106


namespace NUMINAMATH_GPT_find_large_number_l2231_223169

theorem find_large_number (L S : ℕ) (h1 : L - S = 1311) (h2 : L = 11 * S + 11) : L = 1441 :=
sorry

end NUMINAMATH_GPT_find_large_number_l2231_223169


namespace NUMINAMATH_GPT_parabola_constants_sum_l2231_223115

-- Definition based on the given conditions
structure Parabola where
  a: ℝ
  b: ℝ
  c: ℝ
  vertex_x: ℝ
  vertex_y: ℝ
  point_x: ℝ
  point_y: ℝ

-- Definitions of the specific parabola based on the problem's conditions
noncomputable def givenParabola : Parabola := {
  a := -1/4,
  b := -5/2,
  c := -1/4,
  vertex_x := 6,
  vertex_y := -5,
  point_x := 2,
  point_y := -1
}

-- Theorem proving the required value of a + b + c
theorem parabola_constants_sum : givenParabola.a + givenParabola.b + givenParabola.c = -3.25 :=
  by
  sorry

end NUMINAMATH_GPT_parabola_constants_sum_l2231_223115


namespace NUMINAMATH_GPT_rainfall_second_week_l2231_223197

theorem rainfall_second_week (x : ℝ) 
  (h1 : x + 1.5 * x = 25) :
  1.5 * x = 15 :=
by
  sorry

end NUMINAMATH_GPT_rainfall_second_week_l2231_223197


namespace NUMINAMATH_GPT_num_even_divisors_of_8_l2231_223110

def factorial (n : Nat) : Nat :=
  match n with
  | 0     => 1
  | Nat.succ n' => Nat.succ n' * factorial n'

-- Define the prime factorization of 8!
def prime_factors_eight_factorial : Nat := 2^7 * 3^2 * 5 * 7

-- Definition of an even divisor of 8!
def is_even_divisor (d : Nat) : Prop :=
  d ∣ prime_factors_eight_factorial ∧ 2 ∣ d

-- Calculation of number of even divisors of 8!
def num_even_divisors_8! : Nat :=
  7 * 3 * 2 * 2

theorem num_even_divisors_of_8! :
  num_even_divisors_8! = 84 :=
sorry

end NUMINAMATH_GPT_num_even_divisors_of_8_l2231_223110


namespace NUMINAMATH_GPT_normal_level_short_of_capacity_l2231_223104

noncomputable def total_capacity (water_amount : ℕ) (percentage : ℝ) : ℝ :=
  water_amount / percentage

noncomputable def normal_level (water_amount : ℕ) : ℕ :=
  water_amount / 2

theorem normal_level_short_of_capacity (water_amount : ℕ) (percentage : ℝ) (capacity : ℝ) (normal : ℕ) : 
  water_amount = 30 ∧ percentage = 0.75 ∧ capacity = total_capacity water_amount percentage ∧ normal = normal_level water_amount →
  (capacity - ↑normal) = 25 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_normal_level_short_of_capacity_l2231_223104


namespace NUMINAMATH_GPT_find_S6_l2231_223195

variable (a_n : ℕ → ℝ) -- Assume a_n gives the nth term of an arithmetic sequence.
variable (S_n : ℕ → ℝ) -- Assume S_n gives the sum of the first n terms of the sequence.

-- Conditions:
axiom S_2_eq : S_n 2 = 2
axiom S_4_eq : S_n 4 = 10

-- Define what it means to find S_6
theorem find_S6 : S_n 6 = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_S6_l2231_223195


namespace NUMINAMATH_GPT_ratio_of_arithmetic_sequences_l2231_223136

-- Definitions for the conditions
variables {a_n b_n : ℕ → ℝ}
variables {S_n T_n : ℕ → ℝ}
variables (d_a d_b : ℝ)

-- Arithmetic sequences conditions
def is_arithmetic_sequence (u_n : ℕ → ℝ) (t : ℝ) (d : ℝ) : Prop :=
  ∀ (n : ℕ), u_n n = t + n * d

-- Sum of first n terms conditions
def sum_of_first_n_terms (u_n : ℕ → ℝ) (Sn : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), Sn n = n * (u_n 1 + u_n (n-1)) / 2

-- Main theorem statement
theorem ratio_of_arithmetic_sequences (h1 : is_arithmetic_sequence a_n (a_n 0) d_a)
                                     (h2 : is_arithmetic_sequence b_n (b_n 0) d_b)
                                     (h3 : sum_of_first_n_terms a_n S_n)
                                     (h4 : sum_of_first_n_terms b_n T_n)
                                     (h5 : ∀ n, (S_n n) / (T_n n) = (2 * n) / (3 * n + 1)) :
                                     ∀ n, (a_n n) / (b_n n) = (2 * n - 1) / (3 * n - 1) := sorry

end NUMINAMATH_GPT_ratio_of_arithmetic_sequences_l2231_223136


namespace NUMINAMATH_GPT_determinant_evaluation_l2231_223181

theorem determinant_evaluation (x z : ℝ) :
  (Matrix.det ![
    ![1, x, z],
    ![1, x + z, z],
    ![1, x, x + z]
  ]) = x * z - z * z := 
sorry

end NUMINAMATH_GPT_determinant_evaluation_l2231_223181


namespace NUMINAMATH_GPT_eggs_supplied_l2231_223157

-- Define the conditions
def daily_eggs_first_store (D : ℕ) : ℕ := 12 * D
def daily_eggs_second_store : ℕ := 30
def total_weekly_eggs (D : ℕ) : ℕ := 7 * (daily_eggs_first_store D + daily_eggs_second_store)

-- Statement: prove that if the total number of eggs supplied in a week is 630,
-- then Mark supplies 5 dozen eggs to the first store each day.
theorem eggs_supplied (D : ℕ) (h : total_weekly_eggs D = 630) : D = 5 :=
by
  sorry

end NUMINAMATH_GPT_eggs_supplied_l2231_223157


namespace NUMINAMATH_GPT_total_students_is_88_l2231_223144

def orchestra_students : Nat := 20
def band_students : Nat := 2 * orchestra_students
def choir_boys : Nat := 12
def choir_girls : Nat := 16
def choir_students : Nat := choir_boys + choir_girls

def total_students : Nat := orchestra_students + band_students + choir_students

theorem total_students_is_88 : total_students = 88 := by
  sorry

end NUMINAMATH_GPT_total_students_is_88_l2231_223144


namespace NUMINAMATH_GPT_exponent_sum_equality_l2231_223187

theorem exponent_sum_equality {a : ℕ} (h1 : 2^12 + 1 = 17 * a) (h2: a = 2^8 + 2^7 + 2^6 + 2^5 + 2^0) : 
  ∃ a1 a2 a3 a4 a5 : ℕ, 
    a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ 
    2^a1 + 2^a2 + 2^a3 + 2^a4 + 2^a5 = a ∧ 
    a1 = 0 ∧ a2 = 5 ∧ a3 = 6 ∧ a4 = 7 ∧ a5 = 8 ∧ 
    5 = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_exponent_sum_equality_l2231_223187


namespace NUMINAMATH_GPT_multiplications_in_three_hours_l2231_223118

theorem multiplications_in_three_hours :
  let rate := 15000  -- multiplications per second
  let seconds_in_three_hours := 3 * 3600  -- seconds in three hours
  let total_multiplications := rate * seconds_in_three_hours
  total_multiplications = 162000000 :=
by
  let rate := 15000
  let seconds_in_three_hours := 3 * 3600
  let total_multiplications := rate * seconds_in_three_hours
  have h : total_multiplications = 162000000 := sorry
  exact h

end NUMINAMATH_GPT_multiplications_in_three_hours_l2231_223118


namespace NUMINAMATH_GPT_right_triangle_side_length_l2231_223139

theorem right_triangle_side_length (a c b : ℕ) (h1 : a = 3) (h2 : c = 5) (h3 : c^2 = a^2 + b^2) : b = 4 :=
sorry

end NUMINAMATH_GPT_right_triangle_side_length_l2231_223139


namespace NUMINAMATH_GPT_line_equation_through_point_line_equation_sum_of_intercepts_l2231_223158

theorem line_equation_through_point (x y : ℝ) (h : y = 2 * x + 5)
  (hx : x = -2) (hy : y = 1) : 2 * x - y + 5 = 0 :=
by {
  sorry
}

theorem line_equation_sum_of_intercepts (x y : ℝ) (h : y = 2 * x + 6)
  (hx : x = -3) (hy : y = 3) : 2 * x - y + 6 = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_line_equation_through_point_line_equation_sum_of_intercepts_l2231_223158


namespace NUMINAMATH_GPT_problem_1_problem_2_l2231_223172

def f (a x : ℝ) : ℝ := |a - 3 * x| - |2 + x|

theorem problem_1 (x : ℝ) : f 2 x ≤ 3 ↔ -3 / 4 ≤ x ∧ x ≤ 7 / 2 := by
  sorry

theorem problem_2 (a x : ℝ) : f a x ≥ 1 - a + 2 * |2 + x| → a ≥ -5 / 2 := by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2231_223172


namespace NUMINAMATH_GPT_range_of_a_l2231_223194

noncomputable def f (x : ℝ) : ℝ := 2 * x + 1 / Real.exp x - Real.exp x

theorem range_of_a (a : ℝ) (h : f (a - 1) + f (2 * a ^ 2) ≤ 0) : 
  a ∈ Set.Iic (-1) ∪ Set.Ici (1 / 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l2231_223194


namespace NUMINAMATH_GPT_triangle_altitude_l2231_223101

theorem triangle_altitude {A b h : ℝ} (hA : A = 720) (hb : b = 40) (hArea : A = 1 / 2 * b * h) : h = 36 :=
by
  sorry

end NUMINAMATH_GPT_triangle_altitude_l2231_223101


namespace NUMINAMATH_GPT_find_positive_integer_pairs_l2231_223164

theorem find_positive_integer_pairs :
  ∀ (m n : ℕ), m > 0 ∧ n > 0 → ∃ k : ℕ, (2^n - 13^m = k^3) ↔ (m = 2 ∧ n = 9) :=
by
  sorry

end NUMINAMATH_GPT_find_positive_integer_pairs_l2231_223164


namespace NUMINAMATH_GPT_quadratic_inequality_solution_range_l2231_223121

open Set Real

theorem quadratic_inequality_solution_range
  (a : ℝ) : (∃ (x1 x2 : ℤ), x1 ≠ x2 ∧ (∀ x : ℝ, x^2 - a * x + 2 * a < 0 ↔ ↑x1 < x ∧ x < ↑x2)) ↔ 
    (a ∈ Icc (-1 : ℝ) ((-1:ℝ)/3)) ∨ (a ∈ Ioo (25 / 3 : ℝ) 9) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_range_l2231_223121


namespace NUMINAMATH_GPT_line_through_point_intersecting_circle_eq_l2231_223102

theorem line_through_point_intersecting_circle_eq :
  ∃ k l : ℝ, (x + 2*y + 9 = 0 ∨ 2*x - y + 3 = 0) ∧ 
    ∀ L : ℝ × ℝ,  
      (L = (-3, -3)) ∧ (x^2 + y^2 + 4*y - 21 = 0) → 
      (L = (-3,-3) → (x + 2*y + 9 = 0 ∨ 2*x - y + 3 = 0)) := 
sorry

end NUMINAMATH_GPT_line_through_point_intersecting_circle_eq_l2231_223102


namespace NUMINAMATH_GPT_trigonometric_identity_l2231_223141

theorem trigonometric_identity 
  (x : ℝ) 
  (h : Real.sin (x + Real.pi / 3) = 1 / 3) :
  Real.sin (5 * Real.pi / 3 - x) - Real.cos (2 * x - Real.pi / 3) = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2231_223141


namespace NUMINAMATH_GPT_q1_correct_q2_correct_l2231_223126

-- Defining the necessary operations
def q1_lhs := 8 / (-2) - (-4) * (-3)
def q2_lhs := (-2) ^ 3 / 4 * (5 - (-3) ^ 2)

-- Theorem statements to prove that they are equal to 8
theorem q1_correct : q1_lhs = 8 := sorry
theorem q2_correct : q2_lhs = 8 := sorry

end NUMINAMATH_GPT_q1_correct_q2_correct_l2231_223126


namespace NUMINAMATH_GPT_kate_change_l2231_223114

def candyCost : ℝ := 0.54
def amountGiven : ℝ := 1.00
def change (amountGiven candyCost : ℝ) : ℝ := amountGiven - candyCost

theorem kate_change : change amountGiven candyCost = 0.46 := by
  sorry

end NUMINAMATH_GPT_kate_change_l2231_223114


namespace NUMINAMATH_GPT_max_x_value_l2231_223123

theorem max_x_value (x y z : ℝ) (h1 : x + y + z = 7) (h2 : x * y + x * z + y * z = 12) : x ≤ 1 :=
by sorry

end NUMINAMATH_GPT_max_x_value_l2231_223123


namespace NUMINAMATH_GPT_dividend_calculation_l2231_223192

theorem dividend_calculation :
  let divisor := 12
  let quotient := 909809
  let dividend := divisor * quotient
  dividend = 10917708 :=
by
  let divisor := 12
  let quotient := 909809
  let dividend := divisor * quotient
  show dividend = 10917708
  sorry

end NUMINAMATH_GPT_dividend_calculation_l2231_223192


namespace NUMINAMATH_GPT_books_left_over_l2231_223184

def total_books (box_count : ℕ) (books_per_box : ℤ) : ℤ :=
  box_count * books_per_box

theorem books_left_over
  (box_count : ℕ)
  (books_per_box : ℤ)
  (new_box_capacity : ℤ)
  (books_total : ℤ := total_books box_count books_per_box) :
  box_count = 1500 →
  books_per_box = 35 →
  new_box_capacity = 43 →
  books_total % new_box_capacity = 40 :=
by
  intros
  sorry

end NUMINAMATH_GPT_books_left_over_l2231_223184


namespace NUMINAMATH_GPT_remainder_of_x_pow_15_minus_1_div_x_plus_1_is_neg_2_l2231_223170

theorem remainder_of_x_pow_15_minus_1_div_x_plus_1_is_neg_2 :
  (x^15 - 1) % (x + 1) = -2 := 
sorry

end NUMINAMATH_GPT_remainder_of_x_pow_15_minus_1_div_x_plus_1_is_neg_2_l2231_223170


namespace NUMINAMATH_GPT_greatest_possible_multiple_of_4_l2231_223196

theorem greatest_possible_multiple_of_4 (x : ℕ) (h1 : x % 4 = 0) (h2 : x^2 < 400) : x ≤ 16 :=
by 
sorry

end NUMINAMATH_GPT_greatest_possible_multiple_of_4_l2231_223196


namespace NUMINAMATH_GPT_solve_ineq_case1_solve_ineq_case2_l2231_223116

theorem solve_ineq_case1 {a x : ℝ} (ha_pos : 0 < a) (ha_lt_one : a < 1) : 
  a^(x + 5) < a^(4 * x - 1) ↔ x < 2 :=
sorry

theorem solve_ineq_case2 {a x : ℝ} (ha_gt_one : a > 1) : 
  a^(x + 5) < a^(4 * x - 1) ↔ x > 2 :=
sorry

end NUMINAMATH_GPT_solve_ineq_case1_solve_ineq_case2_l2231_223116


namespace NUMINAMATH_GPT_raman_profit_percentage_l2231_223156

theorem raman_profit_percentage
  (cost1 weight1 rate1 : ℕ) (cost2 weight2 rate2 : ℕ) (total_cost_mix total_weight mixing_rate selling_rate profit profit_percentage : ℕ)
  (h_cost1 : cost1 = weight1 * rate1)
  (h_cost2 : cost2 = weight2 * rate2)
  (h_total_cost_mix : total_cost_mix = cost1 + cost2)
  (h_total_weight : total_weight = weight1 + weight2)
  (h_mixing_rate : mixing_rate = total_cost_mix / total_weight)
  (h_selling_price : selling_rate * total_weight = profit + total_cost_mix)
  (h_profit : profit = selling_rate * total_weight - total_cost_mix)
  (h_profit_percentage : profit_percentage = (profit * 100) / total_cost_mix)
  (h_weight1 : weight1 = 54)
  (h_rate1 : rate1 = 150)
  (h_weight2 : weight2 = 36)
  (h_rate2 : rate2 = 125)
  (h_selling_rate_value : selling_rate = 196) :
  profit_percentage = 40 :=
sorry

end NUMINAMATH_GPT_raman_profit_percentage_l2231_223156


namespace NUMINAMATH_GPT_unit_cubes_with_paint_l2231_223153

/-- Conditions:
1. Cubes with each side one inch long are glued together to form a larger cube.
2. The larger cube's face is painted with red color and the entire assembly is taken apart.
3. 23 small cubes are found with no paints on them.
-/
theorem unit_cubes_with_paint (n : ℕ) (h1 : n^3 - (n - 2)^3 = 23) (h2 : n = 4) :
    n^3 - 23 = 41 :=
by
  sorry

end NUMINAMATH_GPT_unit_cubes_with_paint_l2231_223153


namespace NUMINAMATH_GPT_solve_n_l2231_223159

/-
Define the condition for the problem.
Given condition: \(\frac{1}{n+1} + \frac{2}{n+1} + \frac{n}{n+1} = 4\)
-/

noncomputable def condition (n : ℚ) : Prop :=
  (1 / (n + 1) + 2 / (n + 1) + n / (n + 1)) = 4

/-
The theorem to prove: Value of \( n \) that satisfies the condition is \( n = -\frac{1}{3} \)
-/
theorem solve_n : ∃ n : ℚ, condition n ∧ n = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_n_l2231_223159


namespace NUMINAMATH_GPT_weight_of_new_person_l2231_223152

def total_weight_increase (num_people : ℕ) (weight_increase_per_person : ℝ) : ℝ :=
  num_people * weight_increase_per_person

def new_person_weight (old_person_weight : ℝ) (total_weight_increase : ℝ) : ℝ :=
  old_person_weight + total_weight_increase

theorem weight_of_new_person :
  let old_person_weight := 50
  let num_people := 8
  let weight_increase_per_person := 2.5
  new_person_weight old_person_weight (total_weight_increase num_people weight_increase_per_person) = 70 := 
by
  sorry

end NUMINAMATH_GPT_weight_of_new_person_l2231_223152


namespace NUMINAMATH_GPT_original_price_of_sarees_l2231_223168

theorem original_price_of_sarees (P : ℝ) (h1 : 0.95 * 0.80 * P = 133) : P = 175 :=
sorry

end NUMINAMATH_GPT_original_price_of_sarees_l2231_223168


namespace NUMINAMATH_GPT_range_of_m_l2231_223178

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (m+1)*x^2 - m*x + m - 1 ≥ 0) ↔ m ≥ (2*Real.sqrt 3)/3 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l2231_223178


namespace NUMINAMATH_GPT_goose_eggs_count_l2231_223145

theorem goose_eggs_count 
  (E : ℕ) 
  (hatch_rate : ℚ)
  (survive_first_month_rate : ℚ)
  (survive_first_year_rate : ℚ)
  (geese_survived_first_year : ℕ)
  (no_more_than_one_goose_per_egg : Prop) 
  (hatch_eq : hatch_rate = 2/3) 
  (survive_first_month_eq : survive_first_month_rate = 3/4) 
  (survive_first_year_eq : survive_first_year_rate = 2/5) 
  (geese_survived_eq : geese_survived_first_year = 130):
  E = 650 :=
by
  sorry

end NUMINAMATH_GPT_goose_eggs_count_l2231_223145


namespace NUMINAMATH_GPT_trader_loss_percent_l2231_223176

theorem trader_loss_percent :
  let SP1 : ℝ := 404415
  let SP2 : ℝ := 404415
  let gain_percent : ℝ := 15 / 100
  let loss_percent : ℝ := 15 / 100
  let CP1 : ℝ := SP1 / (1 + gain_percent)
  let CP2 : ℝ := SP2 / (1 - loss_percent)
  let TCP : ℝ := CP1 + CP2
  let TSP : ℝ := SP1 + SP2
  let overall_loss : ℝ := TSP - TCP
  let overall_loss_percent : ℝ := (overall_loss / TCP) * 100
  overall_loss_percent = -2.25 := 
sorry

end NUMINAMATH_GPT_trader_loss_percent_l2231_223176


namespace NUMINAMATH_GPT_honey_teas_l2231_223154

-- Definitions corresponding to the conditions
def evening_cups := 2
def evening_servings_per_cup := 2
def morning_cups := 1
def morning_servings_per_cup := 1
def afternoon_cups := 1
def afternoon_servings_per_cup := 1
def servings_per_ounce := 6
def container_ounces := 16

-- Calculation for total servings of honey per day and total days until the container is empty
theorem honey_teas :
  (container_ounces * servings_per_ounce) / 
  (evening_cups * evening_servings_per_cup +
   morning_cups * morning_servings_per_cup +
   afternoon_cups * afternoon_servings_per_cup) = 16 :=
by
  sorry

end NUMINAMATH_GPT_honey_teas_l2231_223154


namespace NUMINAMATH_GPT_range_of_2a_plus_b_l2231_223182

theorem range_of_2a_plus_b (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : -2 < b ∧ b < 2) :
  0 < 2 * a + b ∧ 2 * a + b < 10 :=
sorry

end NUMINAMATH_GPT_range_of_2a_plus_b_l2231_223182


namespace NUMINAMATH_GPT_input_for_output_16_l2231_223108

theorem input_for_output_16 (x : ℝ) (y : ℝ) : 
  (y = (if x < 0 then (x + 1)^2 else (x - 1)^2)) → 
  y = 16 → 
  (x = 5 ∨ x = -5) :=
by sorry

end NUMINAMATH_GPT_input_for_output_16_l2231_223108


namespace NUMINAMATH_GPT_ab_value_l2231_223167

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a ^ 2 + b ^ 2 = 35) : a * b = 13 :=
by
  sorry

end NUMINAMATH_GPT_ab_value_l2231_223167
