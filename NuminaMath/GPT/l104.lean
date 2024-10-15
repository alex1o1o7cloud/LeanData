import Mathlib

namespace NUMINAMATH_GPT_scarlet_savings_l104_10473

theorem scarlet_savings :
  ∀ (initial_savings cost_of_earrings cost_of_necklace amount_left : ℕ),
    initial_savings = 80 →
    cost_of_earrings = 23 →
    cost_of_necklace = 48 →
    amount_left = initial_savings - (cost_of_earrings + cost_of_necklace) →
    amount_left = 9 :=
by
  intros initial_savings cost_of_earrings cost_of_necklace amount_left h_is h_earrings h_necklace h_left
  rw [h_is, h_earrings, h_necklace] at h_left
  exact h_left

end NUMINAMATH_GPT_scarlet_savings_l104_10473


namespace NUMINAMATH_GPT_lollipop_distribution_l104_10488

theorem lollipop_distribution :
  let n := 42
  let initial_lollipops := 650
  let required_lollipops := n * (n + 1) / 2
  (required_lollipops - initial_lollipops) = 253 :=
by
  let n := 42
  let initial_lollipops := 650
  let required_lollipops := n * (n + 1) / 2
  have h : required_lollipops = 903 := by norm_num
  have h2 : (required_lollipops - initial_lollipops) = 253 := by norm_num
  exact h2

end NUMINAMATH_GPT_lollipop_distribution_l104_10488


namespace NUMINAMATH_GPT_fifth_term_arithmetic_sequence_l104_10459

variable (x y : ℝ)

def a1 := x + 2 * y^2
def a2 := x - 2 * y^2
def a3 := x + 3 * y
def a4 := x - 4 * y
def d := a2 - a1

theorem fifth_term_arithmetic_sequence : y = -1/2 → 
  x - 10 * y^2 - 4 * y^2 = x - 7/2 := by
  sorry

end NUMINAMATH_GPT_fifth_term_arithmetic_sequence_l104_10459


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l104_10456

theorem problem_part1 (α : ℝ) (h : Real.tan α = -2) :
    (3 * Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α) = -4 / 7 := 
    sorry

theorem problem_part2 (α : ℝ) (h : Real.tan α = -2) :
    3 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = -5 := 
    sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l104_10456


namespace NUMINAMATH_GPT_log_27_3_l104_10434

noncomputable def log_base (a b : ℝ) : ℝ := Real.log a / Real.log b

theorem log_27_3 :
  log_base 3 27 = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_log_27_3_l104_10434


namespace NUMINAMATH_GPT_countSumPairs_correct_l104_10457

def countSumPairs (n : ℕ) : ℕ :=
  n / 2

theorem countSumPairs_correct (n : ℕ) : countSumPairs n = n / 2 := by
  sorry

end NUMINAMATH_GPT_countSumPairs_correct_l104_10457


namespace NUMINAMATH_GPT_fraction_of_succeeding_number_l104_10421

theorem fraction_of_succeeding_number (N : ℝ) (hN : N = 24.000000000000004) :
  ∃ f : ℝ, (1 / 4) * N > f * (N + 1) + 1 ∧ f = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_succeeding_number_l104_10421


namespace NUMINAMATH_GPT_min_value_of_quadratic_l104_10410

noncomputable def quadratic_min_value (x : ℕ) : ℝ :=
  3 * (x : ℝ)^2 - 12 * x + 800

theorem min_value_of_quadratic : (∀ x : ℕ, quadratic_min_value x ≥ 788) ∧ (quadratic_min_value 2 = 788) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l104_10410


namespace NUMINAMATH_GPT_value_of_five_l104_10491

variable (f : ℝ → ℝ)

-- Conditions of the problem
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)
def periodic_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 2) = f (x)

theorem value_of_five (hf_odd : odd_function f) (hf_periodic : periodic_function f) : f 5 = 0 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_five_l104_10491


namespace NUMINAMATH_GPT_fewest_toothpicks_proof_l104_10478

noncomputable def fewest_toothpicks_to_remove (total_toothpicks : ℕ) (additional_row_and_column : ℕ) (triangles : ℕ) (upward_triangles : ℕ) (downward_triangles : ℕ) (max_destroyed_per_toothpick : ℕ) (horizontal_toothpicks : ℕ) : ℕ :=
  horizontal_toothpicks

theorem fewest_toothpicks_proof 
  (total_toothpicks : ℕ := 40) 
  (additional_row_and_column : ℕ := 1) 
  (triangles : ℕ := 35) 
  (upward_triangles : ℕ := 15) 
  (downward_triangles : ℕ := 10)
  (max_destroyed_per_toothpick : ℕ := 1)
  (horizontal_toothpicks : ℕ := 15) :
  fewest_toothpicks_to_remove total_toothpicks additional_row_and_column triangles upward_triangles downward_triangles max_destroyed_per_toothpick horizontal_toothpicks = 15 := 
by 
  sorry

end NUMINAMATH_GPT_fewest_toothpicks_proof_l104_10478


namespace NUMINAMATH_GPT_solution_set_of_inequality_g_geq_2_l104_10497

-- Definition of the function f
def f (x a : ℝ) := |x - a|

-- Definition of the function g
def g (x a : ℝ) := f x a + f (x + 2) a

-- Proof Problem I
theorem solution_set_of_inequality (a : ℝ) (x : ℝ) :
  a = -1 → (f x a ≥ 4 - |2 * x - 1|) ↔ (x ≤ -4/3 ∨ x ≥ 4/3) :=
by sorry

-- Proof Problem II
theorem g_geq_2 (a : ℝ) (x : ℝ) :
  (∀ x, f x a ≤ 1 → (0 ≤ x ∧ x ≤ 2)) → a = 1 → g x a ≥ 2 :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_g_geq_2_l104_10497


namespace NUMINAMATH_GPT_Q_transform_l104_10420

def rotate_180_clockwise (p q : ℝ × ℝ) : ℝ × ℝ :=
  let (px, py) := p
  let (qx, qy) := q
  (2 * px - qx, 2 * py - qy)

def reflect_y_equals_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (px, py) := p
  (py, px)

def Q := (8, -11) -- from the reverse transformations

theorem Q_transform (c d : ℝ) :
  (reflect_y_equals_x (rotate_180_clockwise (2, -3) (c, d)) = (5, -4)) → (d - c = -19) :=
by sorry

end NUMINAMATH_GPT_Q_transform_l104_10420


namespace NUMINAMATH_GPT_volume_of_rectangular_prism_l104_10490

-- Definition of the given conditions
variables (a b c : ℝ)

def condition1 : Prop := a * b = 24
def condition2 : Prop := b * c = 15
def condition3 : Prop := a * c = 10

-- The statement we want to prove
theorem volume_of_rectangular_prism
  (h1 : condition1 a b)
  (h2 : condition2 b c)
  (h3 : condition3 a c) :
  a * b * c = 60 :=
by sorry

end NUMINAMATH_GPT_volume_of_rectangular_prism_l104_10490


namespace NUMINAMATH_GPT_possible_values_of_x_l104_10495

-- Definitions representing the initial conditions
def condition1 (x : ℕ) : Prop := 203 % x = 13
def condition2 (x : ℕ) : Prop := 298 % x = 13

-- Main theorem statement
theorem possible_values_of_x (x : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 19 ∨ x = 95 := 
by
  sorry

end NUMINAMATH_GPT_possible_values_of_x_l104_10495


namespace NUMINAMATH_GPT_fewer_cans_collected_today_than_yesterday_l104_10423

theorem fewer_cans_collected_today_than_yesterday :
  let sarah_yesterday := 50
  let lara_yesterday := sarah_yesterday + 30
  let sarah_today := 40
  let lara_today := 70
  let total_yesterday := sarah_yesterday + lara_yesterday
  let total_today := sarah_today + lara_today
  total_yesterday - total_today = 20 :=
by
  sorry

end NUMINAMATH_GPT_fewer_cans_collected_today_than_yesterday_l104_10423


namespace NUMINAMATH_GPT_jason_initial_money_l104_10476

theorem jason_initial_money (M : ℝ) 
  (h1 : M - (M / 4 + 10 + (2 / 5 * (3 / 4 * M - 10) + 8)) = 130) : 
  M = 320 :=
by
  sorry

end NUMINAMATH_GPT_jason_initial_money_l104_10476


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l104_10472

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (a1 : ℝ)
  (d : ℝ)
  (h1 : d = 2)
  (h2 : ∀ n : ℕ, a n = a1 + (n - 1) * d)
  (h3 :  ∀ n : ℕ, S n = (n * (2 * a1 + (n - 1) * d)) / 2)
  (h4 : S 6 = 3 * S 3) :
  a 9 = 20 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l104_10472


namespace NUMINAMATH_GPT_luke_fish_fillets_l104_10428

theorem luke_fish_fillets : 
  (∃ (catch_rate : ℕ) (days : ℕ) (fillets_per_fish : ℕ), catch_rate = 2 ∧ days = 30 ∧ fillets_per_fish = 2 → 
  (catch_rate * days * fillets_per_fish = 120)) :=
by
  sorry

end NUMINAMATH_GPT_luke_fish_fillets_l104_10428


namespace NUMINAMATH_GPT_profit_percentage_l104_10404

theorem profit_percentage (C S : ℝ) (h1 : C > 0) (h2 : S > 0)
  (h3 : S - 1.25 * C = 0.7023809523809523 * S) :
  ((S - C) / C) * 100 = 320 := by
sorry

end NUMINAMATH_GPT_profit_percentage_l104_10404


namespace NUMINAMATH_GPT_sandy_correct_value_t_l104_10477

theorem sandy_correct_value_t (p q r s : ℕ) (t : ℕ) 
  (hp : p = 2) (hq : q = 4) (hr : r = 6) (hs : s = 8)
  (expr1 : p + q - r + s - t = p + (q - (r + (s - t)))) :
  t = 8 := 
by
  sorry

end NUMINAMATH_GPT_sandy_correct_value_t_l104_10477


namespace NUMINAMATH_GPT_decompose_five_eighths_l104_10402

theorem decompose_five_eighths : 
  ∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ (5 : ℚ) / 8 = 1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) := 
by
  sorry

end NUMINAMATH_GPT_decompose_five_eighths_l104_10402


namespace NUMINAMATH_GPT_missing_number_is_correct_l104_10461

theorem missing_number_is_correct (mean : ℝ) (observed_numbers : List ℝ) (total_obs : ℕ) (x : ℝ) :
  mean = 14.2 →
  observed_numbers = [8, 13, 21, 7, 23] →
  total_obs = 6 →
  (mean * total_obs = x + observed_numbers.sum) →
  x = 13.2 :=
by
  intros h_mean h_obs h_total h_sum
  sorry

end NUMINAMATH_GPT_missing_number_is_correct_l104_10461


namespace NUMINAMATH_GPT_multiply_scientific_notation_l104_10400

theorem multiply_scientific_notation (a b : ℝ) (e1 e2 : ℤ) 
  (h1 : a = 2) (h2 : b = 8) (h3 : e1 = 3) (h4 : e2 = 3) :
  (a * 10^e1) * (b * 10^e2) = 1.6 * 10^7 :=
by
  simp [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_multiply_scientific_notation_l104_10400


namespace NUMINAMATH_GPT_jo_thinking_greatest_integer_l104_10462

theorem jo_thinking_greatest_integer :
  ∃ n : ℕ, n < 150 ∧ 
           (∃ k : ℤ, n = 9 * k - 2) ∧ 
           (∃ m : ℤ, n = 11 * m - 4) ∧ 
           (∀ N : ℕ, (N < 150 ∧ 
                      (∃ K : ℤ, N = 9 * K - 2) ∧ 
                      (∃ M : ℤ, N = 11 * M - 4)) → N ≤ n) 
:= by
  sorry

end NUMINAMATH_GPT_jo_thinking_greatest_integer_l104_10462


namespace NUMINAMATH_GPT_angle_triple_supplementary_l104_10406

theorem angle_triple_supplementary (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
  sorry

end NUMINAMATH_GPT_angle_triple_supplementary_l104_10406


namespace NUMINAMATH_GPT_point_after_rotation_l104_10487

-- Definitions based on conditions
def point_N : ℝ × ℝ := (-1, -2)
def origin_O : ℝ × ℝ := (0, 0)
def rotation_180 (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, -P.2)

-- The statement to be proved
theorem point_after_rotation :
  rotation_180 point_N = (1, 2) :=
by
  sorry

end NUMINAMATH_GPT_point_after_rotation_l104_10487


namespace NUMINAMATH_GPT_option_d_correct_factorization_l104_10438

theorem option_d_correct_factorization (x : ℝ) : 
  -8 * x ^ 2 + 8 * x - 2 = -2 * (2 * x - 1) ^ 2 :=
by 
  sorry

end NUMINAMATH_GPT_option_d_correct_factorization_l104_10438


namespace NUMINAMATH_GPT_range_of_m_l104_10401

theorem range_of_m (m : ℝ) (h : ∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 + x2 = -(m + 2) ∧ x1 * x2 = m + 5) : -5 < m ∧ m < -2 := 
sorry

end NUMINAMATH_GPT_range_of_m_l104_10401


namespace NUMINAMATH_GPT_simplify_complex_fraction_l104_10496

theorem simplify_complex_fraction :
  (⟨-4, -6⟩ : ℂ) / (⟨5, -2⟩ : ℂ) = ⟨-(32 : ℚ) / 21, -(38 : ℚ) / 21⟩ := 
sorry

end NUMINAMATH_GPT_simplify_complex_fraction_l104_10496


namespace NUMINAMATH_GPT_complement_union_eq_l104_10452

open Set

-- Define the universe and sets P and Q
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 3, 5}
def Q : Set ℕ := {1, 2, 4}

-- State the theorem
theorem complement_union_eq :
  ((U \ P) ∪ Q) = {1, 2, 4, 6} := by
  sorry

end NUMINAMATH_GPT_complement_union_eq_l104_10452


namespace NUMINAMATH_GPT_election_winner_votes_l104_10470

theorem election_winner_votes (V : ℝ) (h1 : 0.62 * V - 0.38 * V = 360) :
  0.62 * V = 930 :=
by {
  sorry
}

end NUMINAMATH_GPT_election_winner_votes_l104_10470


namespace NUMINAMATH_GPT_compute_z_pow_8_l104_10418

noncomputable def z : ℂ := (1 - Real.sqrt 3 * Complex.I) / 2

theorem compute_z_pow_8 : z ^ 8 = -(1 + Real.sqrt 3 * Complex.I) / 2 :=
by
  sorry

end NUMINAMATH_GPT_compute_z_pow_8_l104_10418


namespace NUMINAMATH_GPT_least_prime_factor_of_5_to_the_3_minus_5_to_the_2_l104_10441

theorem least_prime_factor_of_5_to_the_3_minus_5_to_the_2 : 
  Nat.minFac (5^3 - 5^2) = 2 := by
  sorry

end NUMINAMATH_GPT_least_prime_factor_of_5_to_the_3_minus_5_to_the_2_l104_10441


namespace NUMINAMATH_GPT_cakes_served_yesterday_l104_10448

theorem cakes_served_yesterday (lunch_cakes dinner_cakes total_cakes served_yesterday : ℕ)
  (h1 : lunch_cakes = 5)
  (h2 : dinner_cakes = 6)
  (h3 : total_cakes = 14)
  (h4 : total_cakes = lunch_cakes + dinner_cakes + served_yesterday) :
  served_yesterday = 3 := 
by 
  sorry

end NUMINAMATH_GPT_cakes_served_yesterday_l104_10448


namespace NUMINAMATH_GPT_greatest_two_digit_multiple_of_17_is_85_l104_10464

theorem greatest_two_digit_multiple_of_17_is_85 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 17 = 0) → m ≤ n) ∧ n = 85 :=
sorry

end NUMINAMATH_GPT_greatest_two_digit_multiple_of_17_is_85_l104_10464


namespace NUMINAMATH_GPT_cotton_equals_iron_l104_10429

theorem cotton_equals_iron (cotton_weight : ℝ) (iron_weight : ℝ)
  (h_cotton : cotton_weight = 1)
  (h_iron : iron_weight = 4) :
  (4 / 5) * cotton_weight = (1 / 5) * iron_weight :=
by
  rw [h_cotton, h_iron]
  simp
  sorry

end NUMINAMATH_GPT_cotton_equals_iron_l104_10429


namespace NUMINAMATH_GPT_proof_set_intersection_l104_10486

noncomputable def U := ℝ
noncomputable def M := {x : ℝ | 0 ≤ x ∧ x < 5}
noncomputable def N := {x : ℝ | x ≥ 2}
noncomputable def compl_U_N := {x : ℝ | x < 2}
noncomputable def intersection := { x : ℝ | 0 ≤ x ∧ x < 2 }

theorem proof_set_intersection : ((compl_U_N ∩ M) = {x : ℝ | 0 ≤ x ∧ x < 2}) :=
by
  sorry

end NUMINAMATH_GPT_proof_set_intersection_l104_10486


namespace NUMINAMATH_GPT_quadratic_m_condition_l104_10442

theorem quadratic_m_condition (m : ℝ) (h_eq : (m - 2) * x ^ (m ^ 2 - 2) - m * x + 1 = 0) (h_pow : m ^ 2 - 2 = 2) :
  m = -2 :=
by sorry

end NUMINAMATH_GPT_quadratic_m_condition_l104_10442


namespace NUMINAMATH_GPT_total_worth_is_correct_l104_10408

-- Define the conditions
def rows : ℕ := 4
def gold_bars_per_row : ℕ := 20
def worth_per_gold_bar : ℕ := 20000

-- Define the calculated values
def total_gold_bars : ℕ := rows * gold_bars_per_row
def total_worth_of_gold_bars : ℕ := total_gold_bars * worth_per_gold_bar

-- Theorem statement to prove the correct total worth
theorem total_worth_is_correct : total_worth_of_gold_bars = 1600000 := by
  sorry

end NUMINAMATH_GPT_total_worth_is_correct_l104_10408


namespace NUMINAMATH_GPT_initial_welders_count_l104_10424

theorem initial_welders_count (W : ℕ) (h1: (1 + 16 * (W - 9) / W = 8)) : W = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_welders_count_l104_10424


namespace NUMINAMATH_GPT_quadratic_root_range_l104_10443

theorem quadratic_root_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, (x₁ > 0) ∧ (x₂ < 0) ∧ (x₁^2 + 2 * (a - 1) * x₁ + 2 * a + 6 = 0) ∧ (x₂^2 + 2 * (a - 1) * x₂ + 2 * a + 6 = 0)) → a < -3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_range_l104_10443


namespace NUMINAMATH_GPT_find_Y_length_l104_10435

theorem find_Y_length (Y : ℝ) : 
  (3 + 2 + 3 + 4 + Y = 7 + 4 + 2) → Y = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_Y_length_l104_10435


namespace NUMINAMATH_GPT_acme_horseshoes_production_l104_10411

theorem acme_horseshoes_production
  (profit : ℝ)
  (initial_outlay : ℝ)
  (cost_per_set : ℝ)
  (selling_price : ℝ)
  (number_of_sets : ℕ) :
  profit = selling_price * number_of_sets - (initial_outlay + cost_per_set * number_of_sets) →
  profit = 15337.5 →
  initial_outlay = 12450 →
  cost_per_set = 20.75 →
  selling_price = 50 →
  number_of_sets = 950 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end NUMINAMATH_GPT_acme_horseshoes_production_l104_10411


namespace NUMINAMATH_GPT_find_the_number_l104_10485

-- Statement
theorem find_the_number (x : ℤ) (h : 2 * x = 3 * x - 25) : x = 25 :=
  sorry

end NUMINAMATH_GPT_find_the_number_l104_10485


namespace NUMINAMATH_GPT_base8_arithmetic_l104_10482

-- Define the numbers in base 8
def num1 : ℕ := 0o453
def num2 : ℕ := 0o267
def num3 : ℕ := 0o512
def expected_result : ℕ := 0o232

-- Prove that (num1 + num2) - num3 = expected_result in base 8
theorem base8_arithmetic : ((num1 + num2) - num3) = expected_result := by
  sorry

end NUMINAMATH_GPT_base8_arithmetic_l104_10482


namespace NUMINAMATH_GPT_find_number_l104_10458

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 8) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l104_10458


namespace NUMINAMATH_GPT_paint_house_18_women_4_days_l104_10449

theorem paint_house_18_women_4_days :
  (∀ (m1 m2 : ℕ) (d1 d2 : ℕ), m1 * d1 = m2 * d2) →
  (12 * 6 = 72) →
  (72 = 18 * d) →
  d = 4.0 :=
by
  sorry

end NUMINAMATH_GPT_paint_house_18_women_4_days_l104_10449


namespace NUMINAMATH_GPT_find_a_plus_b_l104_10471

-- Define the constants and conditions
variables (a b c : ℤ)
variables (a_cond : 0 ≤ a ∧ a < 5) (b_cond : 0 ≤ b ∧ b < 13)
variables (frac_decomp : (1 : ℚ) / 2015 = (a : ℚ) / 5 + (b : ℚ) / 13 + (c : ℚ) / 31)

-- State the theorem
theorem find_a_plus_b (a b c : ℤ) (a_cond : 0 ≤ a ∧ a < 5) (b_cond : 0 ≤ b ∧ b < 13) (frac_decomp : (1 : ℚ) / 2015 = (a : ℚ) / 5 + (b : ℚ) / 13 + (c : ℚ) / 31) :
  a + b = 14 := 
sorry

end NUMINAMATH_GPT_find_a_plus_b_l104_10471


namespace NUMINAMATH_GPT_LCM_of_36_and_220_l104_10446

theorem LCM_of_36_and_220:
  let A := 36
  let B := 220
  let productAB := A * B
  let HCF := 4
  let LCM := (A * B) / HCF
  LCM = 1980 := 
by
  sorry

end NUMINAMATH_GPT_LCM_of_36_and_220_l104_10446


namespace NUMINAMATH_GPT_pyramid_vertices_l104_10483

theorem pyramid_vertices (n : ℕ) (h : 2 * n = 14) : n + 1 = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_pyramid_vertices_l104_10483


namespace NUMINAMATH_GPT_Alan_total_cost_is_84_l104_10455

theorem Alan_total_cost_is_84 :
  let D := 2 * 12
  let A := 12
  let cost_other := 2 * D + A
  let M := 0.4 * cost_other
  2 * D + A + M = 84 := by
    sorry

end NUMINAMATH_GPT_Alan_total_cost_is_84_l104_10455


namespace NUMINAMATH_GPT_solve_for_b_l104_10416

theorem solve_for_b (b : ℝ) : 
  (∀ x y, 3 * y - 2 * x + 6 = 0 ↔ y = (2 / 3) * x - 2) → 
  (∀ x y, 4 * y + b * x + 3 = 0 ↔ y = -(b / 4) * x - 3 / 4) → 
  (∀ m1 m2, (m1 = (2 / 3)) → (m2 = -(b / 4)) → m1 * m2 = -1) → 
  b = 6 :=
sorry

end NUMINAMATH_GPT_solve_for_b_l104_10416


namespace NUMINAMATH_GPT_dog_food_bags_needed_l104_10465

theorem dog_food_bags_needed
  (cup_weight: ℝ)
  (dogs: ℕ)
  (cups_per_day: ℕ)
  (days_in_month: ℕ)
  (bag_weight: ℝ)
  (hcw: cup_weight = 1/4)
  (hd: dogs = 2)
  (hcd: cups_per_day = 6 * 2)
  (hdm: days_in_month = 30)
  (hbw: bag_weight = 20) :
  (dogs * cups_per_day * days_in_month * cup_weight) / bag_weight = 9 :=
by
  sorry

end NUMINAMATH_GPT_dog_food_bags_needed_l104_10465


namespace NUMINAMATH_GPT_tangent_line_value_of_a_l104_10454

theorem tangent_line_value_of_a (a : ℝ) :
  (∃ (m : ℝ), (2 * m - 1 = a * m + Real.log m) ∧ (a + 1 / m = 2)) → a = 1 :=
by 
sorry

end NUMINAMATH_GPT_tangent_line_value_of_a_l104_10454


namespace NUMINAMATH_GPT_no_valid_triangle_exists_l104_10498

-- Variables representing the sides and altitudes of the triangle
variables (a b c h_a h_b h_c : ℕ)

-- Definition of the perimeter condition
def perimeter_condition : Prop := a + b + c = 1995

-- Definition of integer altitudes condition (simplified)
def integer_altitudes_condition : Prop := 
  ∃ (h_a h_b h_c : ℕ), (h_a * 4 * a ^ 2 = 2 * a ^ 2 * b ^ 2 + 2 * a ^ 2 * c ^ 2 + 2 * c ^ 2 * b ^ 2 - a ^ 4 - b ^ 4 - c ^ 4)

-- The main theorem to prove no valid triangle exists
theorem no_valid_triangle_exists : ¬ (∃ (a b c : ℕ), perimeter_condition a b c ∧ integer_altitudes_condition a b c) :=
sorry

end NUMINAMATH_GPT_no_valid_triangle_exists_l104_10498


namespace NUMINAMATH_GPT_remainder_of_7_pow_145_mod_12_l104_10450

theorem remainder_of_7_pow_145_mod_12 : (7 ^ 145) % 12 = 7 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_7_pow_145_mod_12_l104_10450


namespace NUMINAMATH_GPT_possible_values_of_n_l104_10439

theorem possible_values_of_n (n : ℕ) (h_pos : 0 < n) (h_prime_n : Nat.Prime n) (h_prime_double_sub1 : Nat.Prime (2 * n - 1)) (h_prime_quad_sub1 : Nat.Prime (4 * n - 1)) :
  n = 2 ∨ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_n_l104_10439


namespace NUMINAMATH_GPT_burger_cost_l104_10431

theorem burger_cost (b s : ℕ) (h1 : 3 * b + 2 * s = 385) (h2 : 2 * b + 3 * s = 360) : b = 87 :=
sorry

end NUMINAMATH_GPT_burger_cost_l104_10431


namespace NUMINAMATH_GPT_evaluate_expression_l104_10494

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 6
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 6

theorem evaluate_expression :
  ( (1 / a) + (1 / b) + (1 / c) + (1 / d) ) ^ 2 = 96 / 529 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l104_10494


namespace NUMINAMATH_GPT_profit_percentage_l104_10417

-- Definitions and conditions
variable (SP : ℝ) (CP : ℝ)
variable (h : CP = 0.98 * SP)

-- Lean statement to prove the profit percentage is 2.04%
theorem profit_percentage (h : CP = 0.98 * SP) : (SP - CP) / CP * 100 = 2.04 := 
sorry

end NUMINAMATH_GPT_profit_percentage_l104_10417


namespace NUMINAMATH_GPT_sin_squared_value_l104_10451

theorem sin_squared_value (x : ℝ) (h : Real.tan x = 1 / 2) : 
  Real.sin (π / 4 + x) ^ 2 = 9 / 10 :=
by
  -- Proof part, skipped.
  sorry

end NUMINAMATH_GPT_sin_squared_value_l104_10451


namespace NUMINAMATH_GPT_annie_ride_miles_l104_10432

noncomputable def annie_ride_distance : ℕ := 14

theorem annie_ride_miles
  (mike_base_rate : ℝ := 2.5)
  (mike_per_mile_rate : ℝ := 0.25)
  (mike_miles : ℕ := 34)
  (annie_base_rate : ℝ := 2.5)
  (annie_bridge_toll : ℝ := 5.0)
  (annie_per_mile_rate : ℝ := 0.25)
  (annie_miles : ℕ := annie_ride_distance)
  (mike_cost : ℝ := mike_base_rate + mike_per_mile_rate * mike_miles)
  (annie_cost : ℝ := annie_base_rate + annie_bridge_toll + annie_per_mile_rate * annie_miles) :
  mike_cost = annie_cost → annie_miles = 14 := 
by
  sorry

end NUMINAMATH_GPT_annie_ride_miles_l104_10432


namespace NUMINAMATH_GPT_container_volume_ratio_l104_10481

theorem container_volume_ratio
  (A B : ℝ)
  (h : (5 / 6) * A = (3 / 4) * B) :
  (A / B = 9 / 10) :=
sorry

end NUMINAMATH_GPT_container_volume_ratio_l104_10481


namespace NUMINAMATH_GPT_probability_computation_l104_10489

noncomputable def probability_two_equal_three : ℚ :=
  let p_one_digit : ℚ := 3 / 4
  let p_two_digit : ℚ := 1 / 4
  let number_of_dice : ℕ := 5
  let ways_to_choose_two_digit := Nat.choose number_of_dice 2
  ways_to_choose_two_digit * (p_two_digit^2) * (p_one_digit^3)

theorem probability_computation :
  probability_two_equal_three = 135 / 512 :=
by
  sorry

end NUMINAMATH_GPT_probability_computation_l104_10489


namespace NUMINAMATH_GPT_eventually_periodic_sequence_l104_10468

theorem eventually_periodic_sequence
  (a : ℕ → ℕ)
  (h1 : ∀ n m : ℕ, 0 < n → 0 < m → a (n + 2 * m) ∣ (a n + a (n + m)))
  : ∃ N d : ℕ, 0 < N ∧ 0 < d ∧ ∀ n > N, a n = a (n + d) :=
sorry

end NUMINAMATH_GPT_eventually_periodic_sequence_l104_10468


namespace NUMINAMATH_GPT_gambler_final_amount_l104_10480

theorem gambler_final_amount :
  let initial_money := 100
  let win_multiplier := (3/2 : ℚ)
  let loss_multiplier := (1/2 : ℚ)
  let final_multiplier := (win_multiplier * loss_multiplier)^4
  let final_amount := initial_money * final_multiplier
  final_amount = (8100 / 256) :=
by
  sorry

end NUMINAMATH_GPT_gambler_final_amount_l104_10480


namespace NUMINAMATH_GPT_ratio_of_area_l104_10474

noncomputable def area_ratio (l w r : ℝ) : ℝ :=
  if h1 : 2 * l + 2 * w = 2 * Real.pi * r 
  ∧ l = 2 * w then 
    (l * w) / (Real.pi * r ^ 2) 
  else 
    0

theorem ratio_of_area (l w r : ℝ) 
  (h1 : 2 * l + 2 * w = 2 * Real.pi * r) 
  (h2 : l = 2 * w) :
  area_ratio l w r = 2 * Real.pi / 9 :=
by
  unfold area_ratio
  simp [h1, h2]
  sorry

end NUMINAMATH_GPT_ratio_of_area_l104_10474


namespace NUMINAMATH_GPT_length_QF_l104_10433

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 8 * x

def focus : ℝ × ℝ := (2, 0)

def directrix (x y : ℝ) : Prop := x = 1 -- Directrix of the given parabola

def point_on_directrix (P : ℝ × ℝ) : Prop := directrix P.1 P.2

def point_on_parabola (Q : ℝ × ℝ) : Prop := parabola Q.1 Q.2

def point_on_line_PF (P F Q : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, m ≠ 0 ∧ (Q.2 = m * (Q.1 - F.1) + F.2) ∧ point_on_parabola Q

def vector_equality (F P Q : ℝ × ℝ) : Prop :=
  (4 * (Q.1 - F.1), 4 * (Q.2 - F.2)) = (P.1 - F.1, P.2 - F.2)

theorem length_QF 
  (P Q : ℝ × ℝ)
  (hPd : point_on_directrix P)
  (hPQ : point_on_line_PF P focus Q)
  (hVec : vector_equality focus P Q) : 
  dist Q focus = 3 :=
by
  sorry

end NUMINAMATH_GPT_length_QF_l104_10433


namespace NUMINAMATH_GPT_modulus_of_z_l104_10463

noncomputable def z : ℂ := sorry
def condition (z : ℂ) : Prop := z * (1 - Complex.I) = 2 * Complex.I

theorem modulus_of_z (hz : condition z) : Complex.abs z = Real.sqrt 2 := sorry

end NUMINAMATH_GPT_modulus_of_z_l104_10463


namespace NUMINAMATH_GPT_impossible_even_n_m_if_n3_plus_m3_is_odd_l104_10469

theorem impossible_even_n_m_if_n3_plus_m3_is_odd
  (n m : ℤ) (h : (n^3 + m^3) % 2 = 1) : ¬((n % 2 = 0) ∧ (m % 2 = 0)) := by
  sorry

end NUMINAMATH_GPT_impossible_even_n_m_if_n3_plus_m3_is_odd_l104_10469


namespace NUMINAMATH_GPT_fuel_ethanol_problem_l104_10447

theorem fuel_ethanol_problem (x : ℝ) (h : 0.12 * x + 0.16 * (200 - x) = 28) : x = 100 := 
by
  sorry

end NUMINAMATH_GPT_fuel_ethanol_problem_l104_10447


namespace NUMINAMATH_GPT_alberto_biked_more_than_bjorn_l104_10499

-- Define the distances traveled by Bjorn and Alberto after 5 hours.
def b_distance : ℝ := 75
def a_distance : ℝ := 100

-- Statement to prove the distance difference after 5 hours.
theorem alberto_biked_more_than_bjorn : a_distance - b_distance = 25 := 
by
  -- Proof is skipped, focusing only on the statement.
  sorry

end NUMINAMATH_GPT_alberto_biked_more_than_bjorn_l104_10499


namespace NUMINAMATH_GPT_total_red_cards_l104_10430

theorem total_red_cards (num_standard_decks : ℕ) (num_special_decks : ℕ)
  (red_standard_deck : ℕ) (additional_red_special_deck : ℕ)
  (total_decks : ℕ) (h1 : num_standard_decks = 5)
  (h2 : num_special_decks = 10)
  (h3 : red_standard_deck = 26)
  (h4 : additional_red_special_deck = 4)
  (h5 : total_decks = num_standard_decks + num_special_decks) :
  num_standard_decks * red_standard_deck +
  num_special_decks * (red_standard_deck + additional_red_special_deck) = 430 := by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_total_red_cards_l104_10430


namespace NUMINAMATH_GPT_sum_of_coordinates_l104_10479

theorem sum_of_coordinates {g h : ℝ → ℝ} 
  (h₁ : g 4 = 5)
  (h₂ : ∀ x, h x = (g x)^2) :
  4 + h 4 = 29 := by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_l104_10479


namespace NUMINAMATH_GPT_tank_capacity_is_48_l104_10415

-- Define the conditions
def num_4_liter_bucket_used : ℕ := 12
def num_3_liter_bucket_used : ℕ := num_4_liter_bucket_used + 4

-- Define the capacities of the buckets and the tank
def bucket_4_liters_capacity : ℕ := 4 * num_4_liter_bucket_used
def bucket_3_liters_capacity : ℕ := 3 * num_3_liter_bucket_used

-- Tank capacity
def tank_capacity : ℕ := 48

-- Statement to prove
theorem tank_capacity_is_48 : 
    bucket_4_liters_capacity = tank_capacity ∧
    bucket_3_liters_capacity = tank_capacity := by
  sorry

end NUMINAMATH_GPT_tank_capacity_is_48_l104_10415


namespace NUMINAMATH_GPT_quadratic_solution1_quadratic_solution2_l104_10403

theorem quadratic_solution1 (x : ℝ) :
  (x^2 + 4 * x - 4 = 0) ↔ (x = -2 + 2 * Real.sqrt 2 ∨ x = -2 - 2 * Real.sqrt 2) :=
by sorry

theorem quadratic_solution2 (x : ℝ) :
  ((x - 1)^2 = 2 * (x - 1)) ↔ (x = 1 ∨ x = 3) :=
by sorry

end NUMINAMATH_GPT_quadratic_solution1_quadratic_solution2_l104_10403


namespace NUMINAMATH_GPT_quadratic_completing_square_l104_10425

theorem quadratic_completing_square :
  ∃ (a b c : ℚ), a = 12 ∧ b = 6 ∧ c = 1296 ∧ 12 + 6 + 1296 = 1314 ∧
  (12 * (x + b)^2 + c = 12 * x^2 + 144 * x + 1728) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_completing_square_l104_10425


namespace NUMINAMATH_GPT_primes_less_than_200_with_ones_digit_3_l104_10444

theorem primes_less_than_200_with_ones_digit_3 : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, Prime n ∧ n < 200 ∧ n % 10 = 3) ∧ S.card = 12 := 
by
  sorry

end NUMINAMATH_GPT_primes_less_than_200_with_ones_digit_3_l104_10444


namespace NUMINAMATH_GPT_angles_on_axes_correct_l104_10475

-- Definitions for angles whose terminal sides lie on x-axis and y-axis.
def angles_on_x_axis (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi
def angles_on_y_axis (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi + Real.pi / 2

-- Combined definition for angles on the coordinate axes using Lean notation
def angles_on_axes (α : ℝ) : Prop := ∃ n : ℤ, α = n * (Real.pi / 2)

-- Theorem stating that angles on the coordinate axes are of the form nπ/2.
theorem angles_on_axes_correct : ∀ α : ℝ, (angles_on_x_axis α ∨ angles_on_y_axis α) ↔ angles_on_axes α := 
sorry -- Proof is omitted.

end NUMINAMATH_GPT_angles_on_axes_correct_l104_10475


namespace NUMINAMATH_GPT_find_k_l104_10484

theorem find_k (k x y : ℕ) (h : k * 2 + 1 = 5) : k = 2 :=
by {
  -- Proof will go here
  sorry
}

end NUMINAMATH_GPT_find_k_l104_10484


namespace NUMINAMATH_GPT_mars_bars_count_l104_10414

theorem mars_bars_count (total_candy_bars snickers butterfingers : ℕ) (h_total : total_candy_bars = 12) (h_snickers : snickers = 3) (h_butterfingers : butterfingers = 7) :
  total_candy_bars - (snickers + butterfingers) = 2 :=
by sorry

end NUMINAMATH_GPT_mars_bars_count_l104_10414


namespace NUMINAMATH_GPT_hourly_wage_increase_is_10_percent_l104_10419

theorem hourly_wage_increase_is_10_percent :
  ∀ (H W : ℝ), 
    ∀ (H' : ℝ), H' = H * (1 - 0.09090909090909092) →
    (H * W = H' * W') →
    (W' = (100 * W) / 90) := by
  sorry

end NUMINAMATH_GPT_hourly_wage_increase_is_10_percent_l104_10419


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l104_10460

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h3 : a 1 + 4 * d = (a 0 + 16 * d) * (a 0 + 4 * d) / a 0 ) :
  (a 1 + 4 * d) / a 0 = 3 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l104_10460


namespace NUMINAMATH_GPT_problem_solution_l104_10445

theorem problem_solution : (3106 - 2935 + 17)^2 / 121 = 292 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l104_10445


namespace NUMINAMATH_GPT_product_of_reals_condition_l104_10413

theorem product_of_reals_condition (x : ℝ) (h : x + 1/x = 3 * x) : 
  ∃ x1 x2 : ℝ, x1 + 1/x1 = 3 * x1 ∧ x2 + 1/x2 = 3 * x2 ∧ x1 * x2 = -1/2 := 
sorry

end NUMINAMATH_GPT_product_of_reals_condition_l104_10413


namespace NUMINAMATH_GPT_license_plate_combinations_l104_10492

-- Definitions based on the conditions
def num_letters := 26
def num_digits := 10
def num_positions := 5

def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Main theorem statement
theorem license_plate_combinations :
  choose num_letters 2 * (num_letters - 2) * choose num_positions 2 * choose (num_positions - 2) 2 * num_digits * (num_digits - 1) * (num_digits - 2) = 7776000 :=
by
  sorry

end NUMINAMATH_GPT_license_plate_combinations_l104_10492


namespace NUMINAMATH_GPT_digit_five_occurrences_l104_10453

/-- 
  Define that a 24-hour digital clock display shows times containing at least one 
  occurrence of the digit '5' a total of 450 times in a 24-hour period.
--/
def contains_digit_five (n : Nat) : Prop := 
  n / 10 = 5 ∨ n % 10 = 5

def count_times_with_digit_five : Nat :=
  let hours_with_five := 2 * 60  -- 05:00-05:59 and 15:00-15:59, each hour has 60 minutes
  let remaining_hours := 22 * 15 -- 22 hours, each hour has 15 minutes
  hours_with_five + remaining_hours

theorem digit_five_occurrences : count_times_with_digit_five = 450 := by
  sorry

end NUMINAMATH_GPT_digit_five_occurrences_l104_10453


namespace NUMINAMATH_GPT_tom_speed_from_A_to_B_l104_10405

theorem tom_speed_from_A_to_B (D S : ℝ) (h1 : 2 * D = S * (3 * D / 36 - D / 20))
  (h2 : S * (3 * D / 36 - D / 20) = 3 * D / 36 ∨ 3 * D / 36 = S * (3 * D / 36 - D / 20))
  (h3 : D > 0) : S = 60 :=
by { sorry }

end NUMINAMATH_GPT_tom_speed_from_A_to_B_l104_10405


namespace NUMINAMATH_GPT_find_factor_l104_10412

theorem find_factor (f : ℝ) : (120 * f - 138 = 102) → f = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_factor_l104_10412


namespace NUMINAMATH_GPT_proposition_5_l104_10437

/-! 
  Proposition 5: If there are four points A, B, C, D in a plane, 
  then the vector addition relation: \overrightarrow{AC} + \overrightarrow{BD} = \overrightarrow{BC} + \overrightarrow{AD} must hold.
--/

variables {A B C D : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
variables (AC BD BC AD : A)

-- Theorem Statement in Lean 4
theorem proposition_5 (AC BD BC AD : A)
  : AC + BD = BC + AD := by
  -- Proof by congruence and equality, will add actual steps here
  sorry

end NUMINAMATH_GPT_proposition_5_l104_10437


namespace NUMINAMATH_GPT_proportion_of_second_prize_winners_l104_10422

-- conditions
variables (A B C : ℝ) -- A, B, and C represent the proportions of first, second, and third prize winners respectively.
variables (h1 : A + B = 3 / 4)
variables (h2 : B + C = 2 / 3)

-- statement
theorem proportion_of_second_prize_winners : B = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_proportion_of_second_prize_winners_l104_10422


namespace NUMINAMATH_GPT_find_multiplier_l104_10467

theorem find_multiplier (x n : ℤ) (h : 2 * n + 20 = x * n - 4) (hn : n = 4) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_multiplier_l104_10467


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l104_10407

variable (a b : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : b > a) (h2 : a > 0) : 
  (a * (b + 1) > a^2) ∧ ¬(∀ (a b : ℝ), a * (b + 1) > a^2 → b > a ∧ a > 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l104_10407


namespace NUMINAMATH_GPT_g_possible_values_l104_10493

noncomputable def g (x y z : ℝ) : ℝ :=
  (x + y) / x + (y + z) / y + (z + x) / z

theorem g_possible_values (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  6 ≤ g x y z :=
by
  sorry

end NUMINAMATH_GPT_g_possible_values_l104_10493


namespace NUMINAMATH_GPT_math_problem_l104_10466
noncomputable def sum_of_terms (a b c d : ℕ) : ℕ := a + b + c + d

theorem math_problem
  (x y : ℝ)
  (h₁ : x + y = 5)
  (h₂ : 5 * x * y = 7) :
  ∃ a b c d : ℕ, 
  x = (a + b * Real.sqrt c) / d ∧
  a = 25 ∧ b = 1 ∧ c = 485 ∧ d = 10 ∧ sum_of_terms a b c d = 521 := by
sorry

end NUMINAMATH_GPT_math_problem_l104_10466


namespace NUMINAMATH_GPT_smallest_positive_m_integral_solutions_l104_10426

theorem smallest_positive_m_integral_solutions (m : ℕ) :
  (∃ (x y : ℤ), 10 * x * x - m * x + 660 = 0 ∧ 10 * y * y - m * y + 660 = 0 ∧ x ≠ y)
  → m = 170 := sorry

end NUMINAMATH_GPT_smallest_positive_m_integral_solutions_l104_10426


namespace NUMINAMATH_GPT_largest_divisor_of_m_l104_10409

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : ∃ k : ℕ, m^3 = 847 * k) : ∃ d : ℕ, d = 77 ∧ ∀ x : ℕ, x > d → ¬ (x ∣ m) :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_m_l104_10409


namespace NUMINAMATH_GPT_find_2a_2b_2c_2d_l104_10440

open Int

theorem find_2a_2b_2c_2d (a b c d : ℤ) 
  (h1 : a - b + c = 7) 
  (h2 : b - c + d = 8) 
  (h3 : c - d + a = 4) 
  (h4 : d - a + b = 1) : 
  2*a + 2*b + 2*c + 2*d = 20 := 
sorry

end NUMINAMATH_GPT_find_2a_2b_2c_2d_l104_10440


namespace NUMINAMATH_GPT_disjoint_subsets_less_elements_l104_10427

open Nat

theorem disjoint_subsets_less_elements (m : ℕ) (A B : Finset ℕ) (hA : A ⊆ Finset.range (m + 1))
  (hB : B ⊆ Finset.range (m + 1)) (h_disjoint : Disjoint A B)
  (h_sum : A.sum id = B.sum id) : ↑(A.card) < m / Real.sqrt 2 ∧ ↑(B.card) < m / Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_disjoint_subsets_less_elements_l104_10427


namespace NUMINAMATH_GPT_son_l104_10436

variable (S M : ℕ)

theorem son's_age
  (h1 : M = S + 24)
  (h2 : M + 2 = 2 * (S + 2))
  : S = 22 :=
sorry

end NUMINAMATH_GPT_son_l104_10436
