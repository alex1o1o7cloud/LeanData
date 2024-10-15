import Mathlib

namespace NUMINAMATH_GPT_a_2_geometric_sequence_l1787_178724

theorem a_2_geometric_sequence (a : ℝ) (n : ℕ) (S : ℕ → ℝ)
  (h1 : ∀ n ≥ 2, S n = a * 3^n - 2) : S 2 = 12 :=
by 
  sorry

end NUMINAMATH_GPT_a_2_geometric_sequence_l1787_178724


namespace NUMINAMATH_GPT_rational_neither_positive_nor_fraction_l1787_178730

def is_rational (q : ℚ) : Prop :=
  q.floor = q

def is_integer (q : ℚ) : Prop :=
  ∃ n : ℤ, q = n

def is_fraction (q : ℚ) : Prop :=
  ∃ p q : ℤ, q ≠ 0 ∧ q = p / q

def is_positive (q : ℚ) : Prop :=
  q > 0

theorem rational_neither_positive_nor_fraction (q : ℚ) :
  (is_rational q) ∧ ¬(is_positive q) ∧ ¬(is_fraction q) ↔
  (is_integer q ∧ q ≤ 0) :=
sorry

end NUMINAMATH_GPT_rational_neither_positive_nor_fraction_l1787_178730


namespace NUMINAMATH_GPT_find_b_value_l1787_178749

theorem find_b_value (x y z : ℝ) (u t : ℕ) (h_pos_xyx : x > 0 ∧ y > 0 ∧ z > 0 ∧ u > 0 ∧ t > 0)
  (h1 : (x + y - z) / z = 1) (h2 : (x - y + z) / y = 1) (h3 : (-x + y + z) / x = 1) 
  (ha : (x + y) * (y + z) * (z + x) / (x * y * z) = 8) (hu_t : u + t + u * t = 34) : (u + t = 10) :=
by
  sorry

end NUMINAMATH_GPT_find_b_value_l1787_178749


namespace NUMINAMATH_GPT_solitaire_game_removal_l1787_178773

theorem solitaire_game_removal (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (∃ moves : ℕ, ∀ i : ℕ, i < moves → (i + 1) % 2 = (i % 2) + 1) ↔ (m % 2 = 1 ∨ n % 2 = 1) :=
sorry

end NUMINAMATH_GPT_solitaire_game_removal_l1787_178773


namespace NUMINAMATH_GPT_percent_answered_second_correctly_l1787_178770

theorem percent_answered_second_correctly
  (nA : ℝ) (nAB : ℝ) (n_neither : ℝ) :
  nA = 0.80 → nAB = 0.60 → n_neither = 0.05 → 
  (nA + nB - nAB + n_neither = 1) → 
  ((1 - n_neither) = nA + nB - nAB) → 
  nB = 0.75 :=
by
  intros h1 h2 h3 hUnion hInclusion
  sorry

end NUMINAMATH_GPT_percent_answered_second_correctly_l1787_178770


namespace NUMINAMATH_GPT_angles_symmetric_about_y_axis_l1787_178708

theorem angles_symmetric_about_y_axis (α β : ℝ) (k : ℤ) (h : β = (2 * ↑k + 1) * Real.pi - α) : 
  α + β = (2 * ↑k + 1) * Real.pi :=
sorry

end NUMINAMATH_GPT_angles_symmetric_about_y_axis_l1787_178708


namespace NUMINAMATH_GPT_range_of_k_l1787_178728

theorem range_of_k (k : ℝ) : (∃ x : ℝ, 2 * x - 5 * k = x + 4 ∧ x > 0) → k > -4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1787_178728


namespace NUMINAMATH_GPT_range_of_a_l1787_178767

noncomputable def f (a x : ℝ) : ℝ := 2 * x^2 - a * x + 5

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x → x ≤ y → f a x ≤ f a y) → a ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1787_178767


namespace NUMINAMATH_GPT_paige_finished_problems_at_school_l1787_178799

-- Definitions based on conditions
def math_problems : ℕ := 43
def science_problems : ℕ := 12
def total_problems : ℕ := math_problems + science_problems
def problems_left : ℕ := 11

-- The main theorem we need to prove
theorem paige_finished_problems_at_school : total_problems - problems_left = 44 := by
  sorry

end NUMINAMATH_GPT_paige_finished_problems_at_school_l1787_178799


namespace NUMINAMATH_GPT_factorize_expression_l1787_178761

variable (m n : ℝ)

theorem factorize_expression : 12 * m^2 * n - 12 * m * n + 3 * n = 3 * n * (2 * m - 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1787_178761


namespace NUMINAMATH_GPT_infinite_solutions_for_equation_l1787_178736

theorem infinite_solutions_for_equation :
  ∃ (x y z : ℤ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ ∀ (k : ℤ), (x^2 + y^5 = z^3) :=
sorry

end NUMINAMATH_GPT_infinite_solutions_for_equation_l1787_178736


namespace NUMINAMATH_GPT_green_pill_cost_l1787_178776

variable (x : ℝ) -- cost of a green pill in dollars
variable (y : ℝ) -- cost of a pink pill in dollars
variable (total_cost : ℝ) -- total cost for 21 days

theorem green_pill_cost
  (h1 : x = y + 2) -- a green pill costs $2 more than a pink pill
  (h2 : total_cost = 819) -- total cost for 21 days is $819
  (h3 : ∀ n, n = 21 ∧ total_cost / n = (x + y)) :
  x = 20.5 :=
by
  sorry

end NUMINAMATH_GPT_green_pill_cost_l1787_178776


namespace NUMINAMATH_GPT_length_of_first_train_is_140_l1787_178759

theorem length_of_first_train_is_140 
  (speed1 : ℝ) (speed2 : ℝ) (time_to_cross : ℝ) (length2 : ℝ) 
  (h1 : speed1 = 60) 
  (h2 : speed2 = 40) 
  (h3 : time_to_cross = 12.239020878329734) 
  (h4 : length2 = 200) : 
  ∃ (length1 : ℝ), length1 = 140 := 
by
  sorry

end NUMINAMATH_GPT_length_of_first_train_is_140_l1787_178759


namespace NUMINAMATH_GPT_discount_per_bear_l1787_178786

/-- Suppose the price of the first bear is $4.00 and Wally pays $354.00 for 101 bears.
 Prove that the discount per bear after the first bear is $0.50. -/
theorem discount_per_bear 
  (price_first : ℝ) (total_bears : ℕ) (total_paid : ℝ) (price_rest_bears : ℝ )
  (h1 : price_first = 4.0) (h2 : total_bears = 101) (h3 : total_paid = 354.0) : 
  (price_first + (total_bears - 1) * price_rest_bears - total_paid) / (total_bears - 1) = 0.50 :=
sorry

end NUMINAMATH_GPT_discount_per_bear_l1787_178786


namespace NUMINAMATH_GPT_surcharge_X_is_2_17_percent_l1787_178756

def priceX : ℝ := 575
def priceY : ℝ := 530
def surchargeY : ℝ := 0.03
def totalSaved : ℝ := 41.60

theorem surcharge_X_is_2_17_percent :
  let surchargeX := (2.17 / 100)
  let totalCostX := priceX + (priceX * surchargeX)
  let totalCostY := priceY + (priceY * surchargeY)
  (totalCostX - totalCostY = totalSaved) →
  surchargeX * 100 = 2.17 :=
by
  sorry

end NUMINAMATH_GPT_surcharge_X_is_2_17_percent_l1787_178756


namespace NUMINAMATH_GPT_passes_to_left_l1787_178740

theorem passes_to_left
  (total_passes passes_left passes_right passes_center : ℕ)
  (h1 : total_passes = 50)
  (h2 : passes_right = 2 * passes_left)
  (h3 : passes_center = passes_left + 2)
  (h4 : total_passes = passes_left + passes_right + passes_center) :
  passes_left = 12 :=
by
  sorry

end NUMINAMATH_GPT_passes_to_left_l1787_178740


namespace NUMINAMATH_GPT_molecular_weight_proof_l1787_178744

def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_I : ℝ := 126.90

def molecular_weight (n_N n_H n_I : ℕ) : ℝ :=
  n_N * atomic_weight_N + n_H * atomic_weight_H + n_I * atomic_weight_I

theorem molecular_weight_proof : molecular_weight 1 4 1 = 144.95 :=
by {
  sorry
}

end NUMINAMATH_GPT_molecular_weight_proof_l1787_178744


namespace NUMINAMATH_GPT_reverse_addition_unique_l1787_178784

theorem reverse_addition_unique (k : ℤ) (h t u : ℕ) (n : ℤ)
  (hk : 100 * h + 10 * t + u = k) 
  (h_k_range : 100 < k ∧ k < 1000)
  (h_reverse_addition : 100 * u + 10 * t + h = k + n)
  (digits_range : 1 ≤ h ∧ h ≤ 9 ∧ 0 ≤ t ∧ t ≤ 9 ∧ 0 ≤ u ∧ u ≤ 9) :
  n = 99 :=
sorry

end NUMINAMATH_GPT_reverse_addition_unique_l1787_178784


namespace NUMINAMATH_GPT_odd_function_properties_l1787_178764

theorem odd_function_properties
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ x y, 1 ≤ x ∧ x ≤ 3 ∧ 1 ≤ y ∧ y ≤ 3 ∧ x < y → f x < f y)
  (h_min_val : ∀ x, 1 ≤ x ∧ x ≤ 3 → 7 ≤ f x) :
  (∀ x y, -3 ≤ x ∧ x ≤ -1 ∧ -3 ≤ y ∧ y ≤ -1 ∧ x < y → f x < f y) ∧
  (∀ x, -3 ≤ x ∧ x ≤ -1 → f x ≤ -7) :=
sorry

end NUMINAMATH_GPT_odd_function_properties_l1787_178764


namespace NUMINAMATH_GPT_value_subtracted_from_result_l1787_178785

theorem value_subtracted_from_result (N V : ℕ) (hN : N = 1152) (h: (N / 6) - V = 3) : V = 189 :=
by
  sorry

end NUMINAMATH_GPT_value_subtracted_from_result_l1787_178785


namespace NUMINAMATH_GPT_find_loss_percentage_l1787_178727

theorem find_loss_percentage (CP SP_new : ℝ) (h1 : CP = 875) (h2 : SP_new = CP * 1.04) (h3 : SP_new = SP + 140) : 
  ∃ L : ℝ, SP = CP - (L / 100 * CP) → L = 12 := 
by 
  sorry

end NUMINAMATH_GPT_find_loss_percentage_l1787_178727


namespace NUMINAMATH_GPT_sum_seven_consecutive_l1787_178793

theorem sum_seven_consecutive (n : ℤ) : 
  (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 7 * n + 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_seven_consecutive_l1787_178793


namespace NUMINAMATH_GPT_least_product_of_distinct_primes_greater_than_30_l1787_178729

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Define the problem
theorem least_product_of_distinct_primes_greater_than_30 :
  ∃ p q : ℕ, p ≠ q ∧ 30 < p ∧ 30 < q ∧ is_prime p ∧ is_prime q ∧ p * q = 1147 :=
by
  sorry

end NUMINAMATH_GPT_least_product_of_distinct_primes_greater_than_30_l1787_178729


namespace NUMINAMATH_GPT_vertex_of_parabola_l1787_178760

theorem vertex_of_parabola (c d : ℝ) (h₁ : ∀ x, -x^2 + c*x + d ≤ 0 ↔ (x ≤ -1 ∨ x ≥ 7)) : 
  ∃ v : ℝ × ℝ, v = (3, 16) :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_l1787_178760


namespace NUMINAMATH_GPT_original_cost_of_dress_l1787_178769

theorem original_cost_of_dress (x : ℝ) 
  (h1 : x / 2 - 10 < x)
  (h2 : x - (x / 2 - 10) = 80) : 
  x = 140 := 
sorry

end NUMINAMATH_GPT_original_cost_of_dress_l1787_178769


namespace NUMINAMATH_GPT_problem_statement_l1787_178748

noncomputable def f (a x : ℝ) : ℝ := a^x + a^(-x)

theorem problem_statement (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : f a 1 = 3) :
  f a 0 + f a 1 + f a 2 = 12 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1787_178748


namespace NUMINAMATH_GPT_mala_usha_speed_ratio_l1787_178797

noncomputable def drinking_speed_ratio (M U : ℝ) (tM tU : ℝ) (fracU : ℝ) (total_bottle : ℝ) : ℝ :=
  let U_speed := fracU * total_bottle / tU
  let M_speed := (total_bottle - fracU * total_bottle) / tM
  M_speed / U_speed

theorem mala_usha_speed_ratio :
  drinking_speed_ratio (3/50) (1/50) 10 20 (4/10) 1 = 3 :=
by
  sorry

end NUMINAMATH_GPT_mala_usha_speed_ratio_l1787_178797


namespace NUMINAMATH_GPT_six_digit_squares_l1787_178775

theorem six_digit_squares (x y : ℕ) 
  (h1 : y < 1000)
  (h2 : (1000 * x + y) < 1000000)
  (h3 : y * (y - 1) = 1000 * x)
  (mod8 : y * (y - 1) ≡ 0 [MOD 8])
  (mod125 : y * (y - 1) ≡ 0 [MOD 125]) :
  (1000 * x + y = 390625 ∨ 1000 * x + y = 141376) :=
sorry

end NUMINAMATH_GPT_six_digit_squares_l1787_178775


namespace NUMINAMATH_GPT_cruise_liner_travelers_l1787_178701

theorem cruise_liner_travelers 
  (a : ℤ) 
  (h1 : 250 ≤ a) 
  (h2 : a ≤ 400) 
  (h3 : a % 15 = 7) 
  (h4 : a % 25 = -8) : 
  a = 292 ∨ a = 367 := sorry

end NUMINAMATH_GPT_cruise_liner_travelers_l1787_178701


namespace NUMINAMATH_GPT_find_first_offset_l1787_178754

theorem find_first_offset 
  (diagonal : ℝ) (second_offset : ℝ) (area : ℝ) (first_offset : ℝ)
  (h_diagonal : diagonal = 20)
  (h_second_offset : second_offset = 4)
  (h_area : area = 90)
  (h_area_formula : area = (diagonal * (first_offset + second_offset)) / 2) :
  first_offset = 5 :=
by 
  rw [h_diagonal, h_second_offset, h_area] at h_area_formula 
  -- This would be the place where you handle solving the formula using the given conditions
  sorry

end NUMINAMATH_GPT_find_first_offset_l1787_178754


namespace NUMINAMATH_GPT_simplify_expression_l1787_178795

theorem simplify_expression : (1 / (-8^2)^4) * (-8)^9 = -8 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1787_178795


namespace NUMINAMATH_GPT_ammonia_moles_l1787_178721

-- Definitions corresponding to the given conditions
def moles_KOH : ℚ := 3
def moles_NH4I : ℚ := 3

def balanced_equation (n_KOH n_NH4I : ℚ) : ℚ :=
  if n_KOH = n_NH4I then n_KOH else 0

-- Proof problem: Prove that the reaction produces 3 moles of NH3
theorem ammonia_moles (n_KOH n_NH4I : ℚ) (h1 : n_KOH = moles_KOH) (h2 : n_NH4I = moles_NH4I) :
  balanced_equation n_KOH n_NH4I = 3 :=
by 
  -- proof here 
  sorry

end NUMINAMATH_GPT_ammonia_moles_l1787_178721


namespace NUMINAMATH_GPT_profit_shares_difference_l1787_178747

theorem profit_shares_difference (total_profit : ℝ) (share_ratio_x share_ratio_y : ℝ) 
  (hx : share_ratio_x = 1/2) (hy : share_ratio_y = 1/3) (profit : ℝ):
  total_profit = 500 → profit = (total_profit * share_ratio_x) / ((share_ratio_x + share_ratio_y)) - (total_profit * share_ratio_y) / ((share_ratio_x + share_ratio_y)) → profit = 100 :=
by
  intros
  sorry

end NUMINAMATH_GPT_profit_shares_difference_l1787_178747


namespace NUMINAMATH_GPT_g_eq_g_g_l1787_178720

noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 1

theorem g_eq_g_g (x : ℝ) : 
  g (g x) = g x ↔ x = 2 + Real.sqrt ((11 + 2 * Real.sqrt 21) / 2) 
             ∨ x = 2 - Real.sqrt ((11 + 2 * Real.sqrt 21) / 2) 
             ∨ x = 2 + Real.sqrt ((11 - 2 * Real.sqrt 21) / 2) 
             ∨ x = 2 - Real.sqrt ((11 - 2 * Real.sqrt 21) / 2) := 
by
  sorry

end NUMINAMATH_GPT_g_eq_g_g_l1787_178720


namespace NUMINAMATH_GPT_students_wearing_specific_shirt_and_accessory_count_l1787_178750

theorem students_wearing_specific_shirt_and_accessory_count :
  let total_students := 1000
  let blue_shirt_percent := 0.40
  let red_shirt_percent := 0.25
  let green_shirt_percent := 0.20
  let blue_shirt_students := blue_shirt_percent * total_students
  let red_shirt_students := red_shirt_percent * total_students
  let green_shirt_students := green_shirt_percent * total_students
  let blue_shirt_stripes_percent := 0.30
  let blue_shirt_polka_dots_percent := 0.35
  let red_shirt_stripes_percent := 0.20
  let red_shirt_polka_dots_percent := 0.40
  let green_shirt_stripes_percent := 0.25
  let green_shirt_polka_dots_percent := 0.25
  let accessory_hat_percent := 0.15
  let accessory_scarf_percent := 0.10
  let red_polka_dot_students := red_shirt_polka_dots_percent * red_shirt_students
  let red_polka_dot_hat_students := accessory_hat_percent * red_polka_dot_students
  let green_no_pattern_students := green_shirt_students - (green_shirt_stripes_percent * green_shirt_students + green_shirt_polka_dots_percent * green_shirt_students)
  let green_no_pattern_scarf_students := accessory_scarf_percent * green_no_pattern_students
  red_polka_dot_hat_students + green_no_pattern_scarf_students = 25 := by
    sorry

end NUMINAMATH_GPT_students_wearing_specific_shirt_and_accessory_count_l1787_178750


namespace NUMINAMATH_GPT_bottle_caps_total_l1787_178783

-- Define the conditions
def groups : ℕ := 7
def caps_per_group : ℕ := 5

-- State the theorem
theorem bottle_caps_total : groups * caps_per_group = 35 :=
by
  sorry

end NUMINAMATH_GPT_bottle_caps_total_l1787_178783


namespace NUMINAMATH_GPT_no_real_coeff_quadratic_with_roots_sum_and_product_l1787_178771

theorem no_real_coeff_quadratic_with_roots_sum_and_product (a b c : ℝ) (h : a ≠ 0) :
  ¬ ∃ (α β : ℝ), (α = a + b + c) ∧ (β = a * b * c) ∧ (α + β = -b / a) ∧ (α * β = c / a) :=
by
  sorry

end NUMINAMATH_GPT_no_real_coeff_quadratic_with_roots_sum_and_product_l1787_178771


namespace NUMINAMATH_GPT_sqrt_eq_sum_iff_l1787_178765

open Real

theorem sqrt_eq_sum_iff (a b : ℝ) : sqrt (a^2 + b^2) = a + b ↔ (a * b = 0) ∧ (a + b ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_eq_sum_iff_l1787_178765


namespace NUMINAMATH_GPT_pete_backwards_speed_l1787_178781

variable (speed_pete_hands : ℕ) (speed_tracy_cartwheel : ℕ) (speed_susan_walk : ℕ) (speed_pete_backwards : ℕ)

axiom pete_hands_speed : speed_pete_hands = 2
axiom pete_hands_speed_quarter_tracy_cartwheel : speed_pete_hands = speed_tracy_cartwheel / 4
axiom tracy_cartwheel_twice_susan_walk : speed_tracy_cartwheel = 2 * speed_susan_walk
axiom pete_backwards_three_times_susan_walk : speed_pete_backwards = 3 * speed_susan_walk

theorem pete_backwards_speed : 
  speed_pete_backwards = 12 :=
by
  sorry

end NUMINAMATH_GPT_pete_backwards_speed_l1787_178781


namespace NUMINAMATH_GPT_area_of_moon_slice_l1787_178790

-- Definitions of the conditions
def larger_circle_radius := 5
def larger_circle_center := (2, 0)
def smaller_circle_radius := 2
def smaller_circle_center := (0, 0)

-- Prove the area of the moon slice
theorem area_of_moon_slice : 
  (1/4) * (larger_circle_radius^2 * Real.pi) - (1/4) * (smaller_circle_radius^2 * Real.pi) = (21 * Real.pi) / 4 :=
by
  sorry

end NUMINAMATH_GPT_area_of_moon_slice_l1787_178790


namespace NUMINAMATH_GPT_base_conversion_sum_l1787_178711

noncomputable def A : ℕ := 10

noncomputable def base11_to_nat (x y z : ℕ) : ℕ :=
  x * 11^2 + y * 11^1 + z * 11^0

noncomputable def base12_to_nat (x y z : ℕ) : ℕ :=
  x * 12^2 + y * 12^1 + z * 12^0

theorem base_conversion_sum :
  base11_to_nat 3 7 9 + base12_to_nat 3 9 A = 999 :=
by
  sorry

end NUMINAMATH_GPT_base_conversion_sum_l1787_178711


namespace NUMINAMATH_GPT_interest_rate_correct_l1787_178716

namespace InterestProblem

variable (P : ℤ) (SI : ℤ) (T : ℤ)

def rate_of_interest (P : ℤ) (SI : ℤ) (T : ℤ) : ℚ :=
  (SI * 100) / (P * T)

theorem interest_rate_correct :
  rate_of_interest 400 140 2 = 17.5 := by
  sorry

end InterestProblem

end NUMINAMATH_GPT_interest_rate_correct_l1787_178716


namespace NUMINAMATH_GPT_arithmetic_geometric_product_l1787_178792

theorem arithmetic_geometric_product :
  let a (n : ℕ) := 2 * n - 1
  let b (n : ℕ) := 2 ^ (n - 1)
  b (a 1) * b (a 3) * b (a 5) = 4096 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_product_l1787_178792


namespace NUMINAMATH_GPT_root_calculation_l1787_178706

theorem root_calculation :
  (Real.sqrt (Real.sqrt (0.00032 ^ (1 / 5)) ^ (1 / 4))) = 0.6687 :=
by
  sorry

end NUMINAMATH_GPT_root_calculation_l1787_178706


namespace NUMINAMATH_GPT_reciprocal_of_repeating_decimal_l1787_178709

theorem reciprocal_of_repeating_decimal : 
  (1 : ℚ) / (34 / 99 : ℚ) = 99 / 34 :=
by sorry

end NUMINAMATH_GPT_reciprocal_of_repeating_decimal_l1787_178709


namespace NUMINAMATH_GPT_minimum_value_fraction_l1787_178714

-- Define the conditions in Lean
theorem minimum_value_fraction
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (line_through_center : ∀ x y, x = 1 ∧ y = -2 → a * x - b * y - 1 = 0) :
  (2 / a + 1 / b) = 8 := 
sorry

end NUMINAMATH_GPT_minimum_value_fraction_l1787_178714


namespace NUMINAMATH_GPT_tan_sin_cos_ratio_l1787_178741

open Real

variable {α β : ℝ}

theorem tan_sin_cos_ratio (h1 : tan (α + β) = 2) (h2 : tan (α - β) = 3) :
  sin (2 * α) / cos (2 * β) = 5 / 7 := sorry

end NUMINAMATH_GPT_tan_sin_cos_ratio_l1787_178741


namespace NUMINAMATH_GPT_units_digit_product_l1787_178742

theorem units_digit_product :
  ((734^99 + 347^83) % 10) * ((956^75 - 214^61) % 10) % 10 = 4 := by
  sorry

end NUMINAMATH_GPT_units_digit_product_l1787_178742


namespace NUMINAMATH_GPT_Aiden_sleep_fraction_l1787_178707

theorem Aiden_sleep_fraction (minutes_slept : ℕ) (hour_minutes : ℕ) (h : minutes_slept = 15) (k : hour_minutes = 60) :
  (minutes_slept : ℚ) / hour_minutes = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_Aiden_sleep_fraction_l1787_178707


namespace NUMINAMATH_GPT_circle_standard_equation_l1787_178791

theorem circle_standard_equation (x y : ℝ) :
  let center_x := 2
  let center_y := -1
  let radius := 3
  (center_x = 2) ∧ (center_y = -1) ∧ (radius = 3) → (x - center_x) ^ 2 + (y - center_y) ^ 2 = radius ^ 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_circle_standard_equation_l1787_178791


namespace NUMINAMATH_GPT_smallest_a1_value_l1787_178777

noncomputable def a_seq (n : ℕ) : ℝ :=
if n = 0 then 29 / 98 else if n > 0 then 15 * a_seq (n - 1) - 2 * n else 0

theorem smallest_a1_value :
  (∃ f : ℕ → ℝ, (∀ n > 0, f n = 15 * f (n - 1) - 2 * n) ∧ (∀ n, f n > 0) ∧ (f 1 = 29 / 98)) :=
sorry

end NUMINAMATH_GPT_smallest_a1_value_l1787_178777


namespace NUMINAMATH_GPT_evaluate_expression_l1787_178702

theorem evaluate_expression : 
  ∃ q : ℤ, ∀ (a : ℤ), a = 2022 → (2023 : ℚ) / 2022 - (2022 : ℚ) / 2023 = 4045 / q :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1787_178702


namespace NUMINAMATH_GPT_unbroken_seashells_l1787_178762

theorem unbroken_seashells (total broken : ℕ) (h1 : total = 7) (h2 : broken = 4) : total - broken = 3 :=
by
  -- Proof goes here…
  sorry

end NUMINAMATH_GPT_unbroken_seashells_l1787_178762


namespace NUMINAMATH_GPT_percentage_of_boys_is_90_l1787_178746

variables (B G : ℕ)

def total_children : ℕ := 100
def future_total_children : ℕ := total_children + 100
def percentage_girls : ℕ := 5
def girls_after_increase : ℕ := future_total_children * percentage_girls / 100
def boys_after_increase : ℕ := total_children - girls_after_increase

theorem percentage_of_boys_is_90 :
  B + G = total_children →
  G = girls_after_increase →
  B = total_children - G →
  (B:ℚ) / total_children * 100 = 90 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_boys_is_90_l1787_178746


namespace NUMINAMATH_GPT_total_mass_grain_l1787_178719

-- Given: the mass of the grain is 0.5 tons, and this constitutes 0.2 of the total mass
theorem total_mass_grain (m : ℝ) (h : 0.2 * m = 0.5) : m = 2.5 :=
by {
    -- Proof steps would go here
    sorry
}

end NUMINAMATH_GPT_total_mass_grain_l1787_178719


namespace NUMINAMATH_GPT_floor_multiple_of_floor_l1787_178766

noncomputable def r : ℝ := sorry

theorem floor_multiple_of_floor (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : ∃ k, n = k * m) (hr : r ≥ 1) 
  (floor_multiple : ∀ (m n : ℕ), (∃ k : ℕ, n = k * m) → ∃ l, ⌊n * r⌋ = l * ⌊m * r⌋) :
  ∃ k : ℤ, r = k := 
sorry

end NUMINAMATH_GPT_floor_multiple_of_floor_l1787_178766


namespace NUMINAMATH_GPT_merchant_marked_price_l1787_178745

variable (L C M S : ℝ)

-- Conditions
def condition1 : Prop := C = 0.7 * L
def condition2 : Prop := C = 0.7 * S
def condition3 : Prop := S = 0.8 * M

-- The main statement
theorem merchant_marked_price (h1 : condition1 L C) (h2 : condition2 C S) (h3 : condition3 S M) : M = 1.25 * L :=
by
  sorry

end NUMINAMATH_GPT_merchant_marked_price_l1787_178745


namespace NUMINAMATH_GPT_sector_angle_l1787_178794

theorem sector_angle (r l : ℝ) (h1 : l + 2 * r = 6) (h2 : 1/2 * l * r = 2) : 
  l / r = 1 ∨ l / r = 4 := 
sorry

end NUMINAMATH_GPT_sector_angle_l1787_178794


namespace NUMINAMATH_GPT_max_value_func_l1787_178798

noncomputable def func (x : ℝ) : ℝ :=
  Real.sin x - Real.sqrt 3 * Real.cos x

theorem max_value_func : ∃ x : ℝ, func x = 2 :=
by
  -- proof steps will be provided here
  sorry

end NUMINAMATH_GPT_max_value_func_l1787_178798


namespace NUMINAMATH_GPT_gloves_selection_l1787_178788

theorem gloves_selection (total_pairs : ℕ) (total_gloves : ℕ) (num_to_select : ℕ) 
    (total_ways : ℕ) (no_pair_ways : ℕ) : 
    total_pairs = 4 → 
    total_gloves = 8 → 
    num_to_select = 4 → 
    total_ways = (Nat.choose total_gloves num_to_select) → 
    no_pair_ways = 2^total_pairs → 
    (total_ways - no_pair_ways) = 54 :=
by
  intros
  sorry

end NUMINAMATH_GPT_gloves_selection_l1787_178788


namespace NUMINAMATH_GPT_smallest_x_integer_value_l1787_178735

theorem smallest_x_integer_value (x : ℤ) (h : (x - 5) ∣ 58) : x = -53 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_integer_value_l1787_178735


namespace NUMINAMATH_GPT_quadratic_inequality_hold_l1787_178755

theorem quadratic_inequality_hold (α : ℝ) (h : 0 ≤ α ∧ α ≤ π) :
    (∀ x : ℝ, 8 * x^2 - (8 * Real.sin α) * x + Real.cos (2 * α) ≥ 0) ↔ 
    (α ∈ Set.Icc 0 (π / 6) ∨ α ∈ Set.Icc (5 * π / 6) π) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_hold_l1787_178755


namespace NUMINAMATH_GPT_more_non_representable_ten_digit_numbers_l1787_178722

-- Define the range of ten-digit numbers
def total_ten_digit_numbers : ℕ := 9 * 10^9

-- Define the range of five-digit numbers
def total_five_digit_numbers : ℕ := 90000

-- Calculate the number of pairs of five-digit numbers
def number_of_pairs_five_digit_numbers : ℕ :=
  total_five_digit_numbers * (total_five_digit_numbers + 1)

-- Problem statement
theorem more_non_representable_ten_digit_numbers:
  number_of_pairs_five_digit_numbers < total_ten_digit_numbers :=
by
  -- Proof is non-computable and should be added here
  sorry

end NUMINAMATH_GPT_more_non_representable_ten_digit_numbers_l1787_178722


namespace NUMINAMATH_GPT_cubes_sum_equiv_l1787_178732

theorem cubes_sum_equiv (h : 2^3 + 4^3 + 6^3 + 8^3 + 10^3 + 12^3 + 14^3 + 16^3 + 18^3 = 16200) :
  3^3 + 6^3 + 9^3 + 12^3 + 15^3 + 18^3 + 21^3 + 24^3 + 27^3 = 54675 := 
  sorry

end NUMINAMATH_GPT_cubes_sum_equiv_l1787_178732


namespace NUMINAMATH_GPT_square_area_with_circles_l1787_178725

theorem square_area_with_circles 
  (r : ℝ)
  (nrows : ℕ)
  (ncols : ℕ)
  (circle_radius : r = 3)
  (rows : nrows = 2)
  (columns : ncols = 3)
  (num_circles : nrows * ncols = 6)
  : ∃ (side_length area : ℝ), side_length = ncols * 2 * r ∧ area = side_length ^ 2 ∧ area = 324 := 
by sorry

end NUMINAMATH_GPT_square_area_with_circles_l1787_178725


namespace NUMINAMATH_GPT_inequality_holds_for_all_x_in_interval_l1787_178739

theorem inequality_holds_for_all_x_in_interval (a b : ℝ) :
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |x^2 + a * x + b| ≤ 1 / 8) ↔ (a = -1 ∧ b = 1 / 8) :=
sorry

end NUMINAMATH_GPT_inequality_holds_for_all_x_in_interval_l1787_178739


namespace NUMINAMATH_GPT_janeth_balloons_count_l1787_178787

-- Define the conditions
def bags_round_balloons : Nat := 5
def balloons_per_bag_round : Nat := 20
def bags_long_balloons : Nat := 4
def balloons_per_bag_long : Nat := 30
def burst_round_balloons : Nat := 5

-- Proof statement
theorem janeth_balloons_count:
  let total_round_balloons := bags_round_balloons * balloons_per_bag_round
  let total_long_balloons := bags_long_balloons * balloons_per_bag_long
  let total_balloons := total_round_balloons + total_long_balloons
  total_balloons - burst_round_balloons = 215 :=
by {
  sorry
}

end NUMINAMATH_GPT_janeth_balloons_count_l1787_178787


namespace NUMINAMATH_GPT_curve_equation_l1787_178782

noncomputable def satisfies_conditions (f : ℝ → ℝ) (M₀ : ℝ × ℝ) : Prop :=
  (f M₀.1 = M₀.2) ∧ 
  (∀ (x y : ℝ) (h_tangent : ∀ x y, y = (f x) → x * y - 2 * (f x) * x = 0),
    y = f x → x * y / (y / x) = 2 * x)

theorem curve_equation (f : ℝ → ℝ) :
  satisfies_conditions f (1, 4) →
  (∀ x : ℝ, f x * x = 4) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_curve_equation_l1787_178782


namespace NUMINAMATH_GPT_gini_coefficient_separate_gini_coefficient_combined_l1787_178768

-- Definitions based on provided conditions
def northern_residents : ℕ := 24
def southern_residents : ℕ := 6
def price_per_set : ℝ := 2000
def northern_PPC (x : ℝ) : ℝ := 13.5 - 9 * x
def southern_PPC (x : ℝ) : ℝ := 1.5 * x^2 - 24

-- Gini Coefficient when both regions operate separately
theorem gini_coefficient_separate : 
  ∃ G : ℝ, G = 0.2 :=
  sorry

-- Gini Coefficient change when blending productions as per Northern conditions
theorem gini_coefficient_combined :
  ∃ ΔG : ℝ, ΔG = 0.001 :=
  sorry

end NUMINAMATH_GPT_gini_coefficient_separate_gini_coefficient_combined_l1787_178768


namespace NUMINAMATH_GPT_Loris_needs_more_books_l1787_178789

noncomputable def books_needed (Loris Darryl Lamont : ℕ) :=
  (Lamont - Loris)

theorem Loris_needs_more_books
  (darryl_books: ℕ)
  (lamont_books: ℕ)
  (loris_books_total: ℕ)
  (total_books: ℕ)
  (h1: lamont_books = 2 * darryl_books)
  (h2: darryl_books = 20)
  (h3: loris_books_total + darryl_books + lamont_books = total_books)
  (h4: total_books = 97) :
  books_needed loris_books_total darryl_books lamont_books = 3 :=
sorry

end NUMINAMATH_GPT_Loris_needs_more_books_l1787_178789


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l1787_178774

noncomputable def f (n : Nat) : Set ℕ := sorry

theorem negation_of_universal_proposition :
  (¬ ∀ n : ℕ, f n ⊆ (Set.univ : Set ℕ) ∧ ∀ m ∈ f n, m ≤ n) ↔
  ∃ n_0 : ℕ, f n_0 ⊆ (Set.univ : Set ℕ) ∧ ∀ m ∈ f n_0, m ≤ n_0 :=
sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l1787_178774


namespace NUMINAMATH_GPT_value_of_x_plus_y_l1787_178757

theorem value_of_x_plus_y (x y : ℝ) (h1 : x^2 + x * y + 2 * y = 10) (h2 : y^2 + x * y + 2 * x = 14) :
  x + y = -6 ∨ x + y = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_plus_y_l1787_178757


namespace NUMINAMATH_GPT_incorrect_residual_plot_statement_l1787_178723

theorem incorrect_residual_plot_statement :
  ∀ (vertical_only_residual : Prop)
    (horizontal_any_of : Prop)
    (narrower_band_smaller_ssr : Prop)
    (narrower_band_smaller_corr : Prop)
    ,
    narrower_band_smaller_corr → False :=
  by intros vertical_only_residual horizontal_any_of narrower_band_smaller_ssr narrower_band_smaller_corr
     sorry

end NUMINAMATH_GPT_incorrect_residual_plot_statement_l1787_178723


namespace NUMINAMATH_GPT_eight_pow_n_over_three_eq_512_l1787_178713

theorem eight_pow_n_over_three_eq_512 : 8^(9/3) = 512 :=
by
  -- sorry skips the proof
  sorry

end NUMINAMATH_GPT_eight_pow_n_over_three_eq_512_l1787_178713


namespace NUMINAMATH_GPT_value_of_x_l1787_178796

theorem value_of_x (z : ℕ) (y : ℕ) (x : ℕ) 
  (h₁ : y = z / 5)
  (h₂ : x = y / 2)
  (h₃ : z = 60) : 
  x = 6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l1787_178796


namespace NUMINAMATH_GPT_average_weight_increase_l1787_178779

theorem average_weight_increase 
  (w_old : ℝ) (w_new : ℝ) (n : ℕ) 
  (h1 : w_old = 65) 
  (h2 : w_new = 93) 
  (h3 : n = 8) : 
  (w_new - w_old) / n = 3.5 := 
by 
  sorry

end NUMINAMATH_GPT_average_weight_increase_l1787_178779


namespace NUMINAMATH_GPT_time_for_each_student_l1787_178731

-- Define the conditions as variables
variables (num_students : ℕ) (period_length : ℕ) (num_periods : ℕ)
-- Assume the conditions from the problem
def conditions := num_students = 32 ∧ period_length = 40 ∧ num_periods = 4

-- Define the total time available
def total_time (num_periods period_length : ℕ) := num_periods * period_length

-- Define the time per student
def time_per_student (total_time num_students : ℕ) := total_time / num_students

-- State the theorem to be proven
theorem time_for_each_student : 
  conditions num_students period_length num_periods →
  time_per_student (total_time num_periods period_length) num_students = 5 := sorry

end NUMINAMATH_GPT_time_for_each_student_l1787_178731


namespace NUMINAMATH_GPT_sum_a_t_l1787_178700

theorem sum_a_t (a : ℝ) (t : ℝ) 
  (h₁ : a = 6)
  (h₂ : t = a^2 - 1) : a + t = 41 :=
by
  sorry

end NUMINAMATH_GPT_sum_a_t_l1787_178700


namespace NUMINAMATH_GPT_sum_S11_l1787_178726

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {a1 d : ℝ}

axiom arithmetic_sequence (n : ℕ) : a n = a1 + (n - 1) * d
axiom sum_of_first_n_terms (n : ℕ) : S n = n / 2 * (a 1 + a n)
axiom condition : a 3 + 4 = a 2 + a 7

theorem sum_S11 : S 11 = 44 := by
  sorry

end NUMINAMATH_GPT_sum_S11_l1787_178726


namespace NUMINAMATH_GPT_smallest_b_greater_than_l1787_178751

theorem smallest_b_greater_than (a b : ℤ) (h₁ : 9 < a) (h₂ : a < 21) (h₃ : 10 / b ≥ 2 / 3) (h₄ : b < 31) : 14 < b :=
sorry

end NUMINAMATH_GPT_smallest_b_greater_than_l1787_178751


namespace NUMINAMATH_GPT_cost_of_ticket_when_Matty_was_born_l1787_178743

theorem cost_of_ticket_when_Matty_was_born 
    (cost : ℕ → ℕ) 
    (h_halved : ∀ t : ℕ, cost (t + 10) = cost t / 2) 
    (h_age_30 : cost 30 = 125000) : 
    cost 0 = 1000000 := 
by 
  sorry

end NUMINAMATH_GPT_cost_of_ticket_when_Matty_was_born_l1787_178743


namespace NUMINAMATH_GPT_tetrahedron_sum_of_faces_l1787_178704

theorem tetrahedron_sum_of_faces (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
(h_sum_vertices : b * c * d + a * c * d + a * b * d + a * b * c = 770) :
  a + b + c + d = 57 :=
sorry

end NUMINAMATH_GPT_tetrahedron_sum_of_faces_l1787_178704


namespace NUMINAMATH_GPT_largest_result_is_0_point_1_l1787_178738

theorem largest_result_is_0_point_1 : 
  max (5 * Real.sqrt 2 - 7) (max (7 - 5 * Real.sqrt 2) (max (|1 - 1|) 0.1)) = 0.1 := 
by
  -- We will prove this by comparing each value to 0.1
  sorry

end NUMINAMATH_GPT_largest_result_is_0_point_1_l1787_178738


namespace NUMINAMATH_GPT_price_after_9_years_l1787_178717

-- Assume the initial conditions
def initial_price : ℝ := 640
def decrease_factor : ℝ := 0.75
def years : ℕ := 9
def period : ℕ := 3

-- Define the function to calculate the price after a certain number of years, given the period and decrease factor
def price_after_years (initial_price : ℝ) (decrease_factor : ℝ) (years : ℕ) (period : ℕ) : ℝ :=
  initial_price * (decrease_factor ^ (years / period))

-- State the theorem that we intend to prove
theorem price_after_9_years : price_after_years initial_price decrease_factor 9 period = 270 := by
  sorry

end NUMINAMATH_GPT_price_after_9_years_l1787_178717


namespace NUMINAMATH_GPT_least_number_divisible_by_38_and_3_remainder_1_exists_l1787_178710

theorem least_number_divisible_by_38_and_3_remainder_1_exists :
  ∃ n, n % 38 = 1 ∧ n % 3 = 1 ∧ ∀ m, m % 38 = 1 ∧ m % 3 = 1 → n ≤ m :=
sorry

end NUMINAMATH_GPT_least_number_divisible_by_38_and_3_remainder_1_exists_l1787_178710


namespace NUMINAMATH_GPT_smallest_a_for_f_iter_3_l1787_178737

def f (x : Int) : Int :=
  if x % 4 = 0 ∧ x % 9 = 0 then x / 36
  else if x % 9 = 0 then 4 * x
  else if x % 4 = 0 then 9 * x
  else x + 4

def f_iter (f : Int → Int) (a : Nat) (x : Int) : Int :=
  if a = 0 then x else f_iter f (a - 1) (f x)

theorem smallest_a_for_f_iter_3 (a : Nat) (h : a > 1) : 
  (∀b, b > 1 → b < a → f_iter f b 3 ≠ f 3) ∧ f_iter f a 3 = f 3 ↔ a = 9 := 
  by
  sorry

end NUMINAMATH_GPT_smallest_a_for_f_iter_3_l1787_178737


namespace NUMINAMATH_GPT_median_of_data_set_l1787_178705

def data_set := [2, 3, 3, 4, 6, 6, 8, 8]

def calculate_50th_percentile (l : List ℕ) : ℕ :=
  if H : l.length % 2 = 0 then
    (l.get ⟨l.length / 2 - 1, sorry⟩ + l.get ⟨l.length / 2, sorry⟩) / 2
  else
    l.get ⟨l.length / 2, sorry⟩

theorem median_of_data_set : calculate_50th_percentile data_set = 5 :=
by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_median_of_data_set_l1787_178705


namespace NUMINAMATH_GPT_sum_of_nonzero_perfect_squares_l1787_178733

theorem sum_of_nonzero_perfect_squares (p n : ℕ) (hp_prime : Nat.Prime p) 
    (hn_ge_p : n ≥ p) (h_perfect_square : ∃ k : ℕ, 1 + n * p = k^2) :
    ∃ (a : ℕ) (f : Fin p → ℕ), (∀ i, 0 < f i ∧ ∃ m, f i = m^2) ∧ (n + 1 = a + (Finset.univ.sum f)) :=
sorry

end NUMINAMATH_GPT_sum_of_nonzero_perfect_squares_l1787_178733


namespace NUMINAMATH_GPT_decrypt_encryption_l1787_178780

-- Encryption function description
def encrypt_digit (d : ℕ) : ℕ := 10 - (d * 7 % 10)

def encrypt_number (n : ℕ) : ℕ :=
  let digits := n.digits 10
  let encrypted_digits := digits.map encrypt_digit
  encrypted_digits.foldr (λ d acc => d + acc * 10) 0
  
noncomputable def digit_match (d: ℕ) : ℕ :=
  match d with
  | 0 => 0 | 1 => 3 | 2 => 8 | 3 => 1 | 4 => 6 | 5 => 5
  | 6 => 8 | 7 => 1 | 8 => 4 | 9 => 7 | _ => 0

theorem decrypt_encryption:
encrypt_number 891134 = 473392 :=
by
  sorry

end NUMINAMATH_GPT_decrypt_encryption_l1787_178780


namespace NUMINAMATH_GPT_rectangle_length_is_4_l1787_178758

theorem rectangle_length_is_4 (w l : ℝ) (h_length : l = w + 3) (h_area : l * w = 4) : l = 4 := 
sorry

end NUMINAMATH_GPT_rectangle_length_is_4_l1787_178758


namespace NUMINAMATH_GPT_solve_quadratic_l1787_178715

theorem solve_quadratic : ∀ (x : ℝ), x * (x + 1) = 2014 * 2015 ↔ (x = 2014 ∨ x = -2015) := by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l1787_178715


namespace NUMINAMATH_GPT_sum_place_values_of_specified_digits_l1787_178753

def numeral := 95378637153370261

def place_values_of_3s := [3 * 100000000000, 3 * 10]
def place_values_of_7s := [7 * 10000000000, 7 * 1000000, 7 * 100]
def place_values_of_5s := [5 * 10000000000000, 5 * 1000, 5 * 10000, 5 * 1]

def sum_place_values (lst : List ℕ) : ℕ :=
  lst.foldl (· + ·) 0

def sum_of_place_values := 
  sum_place_values place_values_of_3s + 
  sum_place_values place_values_of_7s + 
  sum_place_values place_values_of_5s

theorem sum_place_values_of_specified_digits :
  sum_of_place_values = 350077055735 :=
by
  sorry

end NUMINAMATH_GPT_sum_place_values_of_specified_digits_l1787_178753


namespace NUMINAMATH_GPT_smallest_a_condition_l1787_178734

theorem smallest_a_condition:
  ∃ a: ℝ, (∀ x y z: ℝ, (0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 1) → a * (x^2 + y^2 + z^2) + x * y * z ≥ 10 / 27) ∧ a = 2 / 9 :=
sorry

end NUMINAMATH_GPT_smallest_a_condition_l1787_178734


namespace NUMINAMATH_GPT_train_crosses_post_in_approximately_18_seconds_l1787_178712

noncomputable def train_length : ℕ := 300
noncomputable def platform_length : ℕ := 350
noncomputable def crossing_time_platform : ℕ := 39

noncomputable def combined_length : ℕ := train_length + platform_length
noncomputable def speed_train : ℝ := combined_length / crossing_time_platform

noncomputable def crossing_time_post : ℝ := train_length / speed_train

theorem train_crosses_post_in_approximately_18_seconds :
  abs (crossing_time_post - 18) < 1 :=
by
  admit

end NUMINAMATH_GPT_train_crosses_post_in_approximately_18_seconds_l1787_178712


namespace NUMINAMATH_GPT_school_distance_l1787_178763

theorem school_distance (T D : ℝ) (h1 : 5 * (T + 6) = 630) (h2 : 7 * (T - 30) = 630) :
  D = 630 :=
sorry

end NUMINAMATH_GPT_school_distance_l1787_178763


namespace NUMINAMATH_GPT_like_terms_exponents_l1787_178703

theorem like_terms_exponents (m n : ℕ) (h₁ : m + 3 = 5) (h₂ : 6 = 2 * n) : m^n = 8 :=
by
  sorry

end NUMINAMATH_GPT_like_terms_exponents_l1787_178703


namespace NUMINAMATH_GPT_rectangular_field_length_l1787_178752

theorem rectangular_field_length (w : ℝ) (h₁ : w * (w + 10) = 171) : w + 10 = 19 := 
by
  sorry

end NUMINAMATH_GPT_rectangular_field_length_l1787_178752


namespace NUMINAMATH_GPT_max_fruits_is_15_l1787_178718

def maxFruits (a m p : ℕ) : Prop :=
  3 * a + 4 * m + 5 * p = 50 ∧ a ≥ 1 ∧ m ≥ 1 ∧ p ≥ 1

theorem max_fruits_is_15 : ∃ a m p : ℕ, maxFruits a m p ∧ a + m + p = 15 := 
  sorry

end NUMINAMATH_GPT_max_fruits_is_15_l1787_178718


namespace NUMINAMATH_GPT_distance_traveled_by_both_cars_l1787_178772

def car_R_speed := 34.05124837953327
def car_P_speed := 44.05124837953327
def car_R_time := 8.810249675906654
def car_P_time := car_R_time - 2

def distance_car_R := car_R_speed * car_R_time
def distance_car_P := car_P_speed * car_P_time

theorem distance_traveled_by_both_cars :
  distance_car_R = 300 :=
by
  sorry

end NUMINAMATH_GPT_distance_traveled_by_both_cars_l1787_178772


namespace NUMINAMATH_GPT_expression_value_l1787_178778

theorem expression_value : 
  (2 ^ 1501 + 5 ^ 1502) ^ 2 - (2 ^ 1501 - 5 ^ 1502) ^ 2 = 20 * 10 ^ 1501 := 
by
  sorry

end NUMINAMATH_GPT_expression_value_l1787_178778
