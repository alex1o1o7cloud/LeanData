import Mathlib

namespace NUMINAMATH_GPT_jake_total_work_hours_l1671_167143

def initial_debt_A := 150
def payment_A := 60
def hourly_rate_A := 15
def remaining_debt_A := initial_debt_A - payment_A
def hours_to_work_A := remaining_debt_A / hourly_rate_A

def initial_debt_B := 200
def payment_B := 80
def hourly_rate_B := 20
def remaining_debt_B := initial_debt_B - payment_B
def hours_to_work_B := remaining_debt_B / hourly_rate_B

def initial_debt_C := 250
def payment_C := 100
def hourly_rate_C := 25
def remaining_debt_C := initial_debt_C - payment_C
def hours_to_work_C := remaining_debt_C / hourly_rate_C

def total_hours_to_work := hours_to_work_A + hours_to_work_B + hours_to_work_C

theorem jake_total_work_hours :
  total_hours_to_work = 18 :=
sorry

end NUMINAMATH_GPT_jake_total_work_hours_l1671_167143


namespace NUMINAMATH_GPT_largest_of_three_roots_l1671_167124

theorem largest_of_three_roots (p q r : ℝ) (hpqr_sum : p + q + r = 3) 
    (hpqr_prod_sum : p * q + p * r + q * r = -8) (hpqr_prod : p * q * r = -15) :
    max p (max q r) = 3 := 
sorry

end NUMINAMATH_GPT_largest_of_three_roots_l1671_167124


namespace NUMINAMATH_GPT_play_area_l1671_167182

theorem play_area (posts : ℕ) (space : ℝ) (extra_posts : ℕ) (short_posts long_posts : ℕ) (short_spaces long_spaces : ℕ) 
  (short_length long_length area : ℝ)
  (h1 : posts = 24) 
  (h2 : space = 5)
  (h3 : extra_posts = 6)
  (h4 : long_posts = short_posts + extra_posts)
  (h5 : 2 * short_posts + 2 * long_posts - 4 = posts)
  (h6 : short_spaces = short_posts - 1)
  (h7 : long_spaces = long_posts - 1)
  (h8 : short_length = short_spaces * space)
  (h9 : long_length = long_spaces * space)
  (h10 : area = short_length * long_length) :
  area = 675 := 
sorry

end NUMINAMATH_GPT_play_area_l1671_167182


namespace NUMINAMATH_GPT_meaningful_if_and_only_if_l1671_167169

theorem meaningful_if_and_only_if (x : ℝ) : (∃ y : ℝ, y = (1 / (x - 1))) ↔ x ≠ 1 :=
by 
  sorry

end NUMINAMATH_GPT_meaningful_if_and_only_if_l1671_167169


namespace NUMINAMATH_GPT_phone_numbers_count_l1671_167175

theorem phone_numbers_count : (2^5 = 32) :=
by sorry

end NUMINAMATH_GPT_phone_numbers_count_l1671_167175


namespace NUMINAMATH_GPT_sum_of_first_49_primes_l1671_167131

def first_49_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
                                   61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 
                                   137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 
                                   199, 211, 223, 227]

theorem sum_of_first_49_primes : first_49_primes.sum = 10787 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_sum_of_first_49_primes_l1671_167131


namespace NUMINAMATH_GPT_num_subsets_containing_6_l1671_167168

open Finset

-- Define the set S
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the subset containing number 6
def subsets_with_6 (s : Finset ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ x => 6 ∈ x)

-- Theorem: The number of subsets of {1, 2, 3, 4, 5, 6} containing the number 6 is 32
theorem num_subsets_containing_6 : (subsets_with_6 S).card = 32 := by
  sorry

end NUMINAMATH_GPT_num_subsets_containing_6_l1671_167168


namespace NUMINAMATH_GPT_find_units_digit_l1671_167100

theorem find_units_digit (A : ℕ) (h : 10 * A + 2 = 20 + A + 9) : A = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_units_digit_l1671_167100


namespace NUMINAMATH_GPT_volume_of_rectangular_solid_l1671_167118

theorem volume_of_rectangular_solid (a b c : ℝ) (h1 : a * b = Real.sqrt 2) (h2 : b * c = Real.sqrt 3) (h3 : c * a = Real.sqrt 6) : a * b * c = Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_volume_of_rectangular_solid_l1671_167118


namespace NUMINAMATH_GPT_arithmetic_sequence_theorem_l1671_167194

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ := (n * (a 1 + a n)) / 2

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_theorem (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : arithmetic_sequence a)
  (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h_a1_pos : a 1 > 0)
  (h_condition : -1 < a 7 / a 6 ∧ a 7 / a 6 < 0) :
  (∃ d, d < 0) ∧ (∀ n, S n > 0 → n ≤ 12) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_theorem_l1671_167194


namespace NUMINAMATH_GPT_infinite_sequence_domain_l1671_167144

def seq_domain (f : ℕ → ℕ) : Set ℕ := {n | 0 < n}

theorem infinite_sequence_domain (f : ℕ → ℕ) (a_n : ℕ → ℕ)
   (h : ∀ (n : ℕ), a_n n = f n) : 
   seq_domain f = {n | 0 < n} :=
sorry

end NUMINAMATH_GPT_infinite_sequence_domain_l1671_167144


namespace NUMINAMATH_GPT_solve_inequalities_l1671_167196

theorem solve_inequalities :
  {x : ℝ | 4 ≤ (2*x) / (3*x - 7) ∧ (2*x) / (3*x - 7) < 9} = {x : ℝ | (63 / 25) < x ∧ x ≤ 2.8} :=
by
  sorry

end NUMINAMATH_GPT_solve_inequalities_l1671_167196


namespace NUMINAMATH_GPT_initially_calculated_average_l1671_167170

theorem initially_calculated_average 
  (correct_sum : ℤ)
  (incorrect_diff : ℤ)
  (num_numbers : ℤ)
  (correct_average : ℤ)
  (h1 : correct_sum = correct_average * num_numbers)
  (h2 : incorrect_diff = 20)
  (h3 : num_numbers = 10)
  (h4 : correct_average = 18) :
  (correct_sum - incorrect_diff) / num_numbers = 16 := by
  sorry

end NUMINAMATH_GPT_initially_calculated_average_l1671_167170


namespace NUMINAMATH_GPT_min_x2_y2_l1671_167160

theorem min_x2_y2 (x y : ℝ) (h : 2 * (x^2 + y^2) = x^2 + y + x * y) : 
  (∃ x y, x = 0 ∧ y = 0) ∨ x^2 + y^2 >= 1 := 
sorry

end NUMINAMATH_GPT_min_x2_y2_l1671_167160


namespace NUMINAMATH_GPT_zero_lies_in_interval_l1671_167158

def f (x : ℝ) : ℝ := -|x - 5| + 2 * x - 1

theorem zero_lies_in_interval (k : ℤ) (h : ∃ x : ℝ, k < x ∧ x < k + 1 ∧ f x = 0) : k = 2 := 
sorry

end NUMINAMATH_GPT_zero_lies_in_interval_l1671_167158


namespace NUMINAMATH_GPT_find_c_l1671_167114

theorem find_c (c : ℝ) (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
  (hf : ∀ x, f x = 2 / (3 * x + c))
  (hfinv : ∀ x, f_inv x = (2 - 3 * x) / (3 * x)) :
  c = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l1671_167114


namespace NUMINAMATH_GPT_add_in_base_7_l1671_167136

theorem add_in_base_7 (X Y : ℕ) (h1 : (X + 5) % 7 = 0) (h2 : (Y + 2) % 7 = X) : X + Y = 2 :=
by
  sorry

end NUMINAMATH_GPT_add_in_base_7_l1671_167136


namespace NUMINAMATH_GPT_max_planes_determined_by_15_points_l1671_167137

theorem max_planes_determined_by_15_points : ∃ (n : ℕ), n = 455 ∧ 
  ∀ (P : Finset (Fin 15)), (15 + 1) ∣ (P.card * (P.card - 1) * (P.card - 2) / 6) → n = (15 * 14 * 13) / 6 :=
by
  sorry

end NUMINAMATH_GPT_max_planes_determined_by_15_points_l1671_167137


namespace NUMINAMATH_GPT_gain_percent_l1671_167134

theorem gain_percent (C S S_d : ℝ) 
  (h1 : 50 * C = 20 * S) 
  (h2 : S_d = S * (1 - 0.15)) : 
  ((S_d - C) / C) * 100 = 112.5 := 
by 
  sorry

end NUMINAMATH_GPT_gain_percent_l1671_167134


namespace NUMINAMATH_GPT_find_k_l1671_167139

theorem find_k (k : ℚ) :
  (5 + ∑' n : ℕ, (5 + 2*k*(n+1)) / 4^n) = 10 → k = 15/4 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1671_167139


namespace NUMINAMATH_GPT_color_schemes_equivalence_l1671_167187

noncomputable def number_of_non_equivalent_color_schemes (n : Nat) : Nat :=
  let total_ways := Nat.choose (n * n) 2
  -- Calculate the count for non-diametrically opposite positions (4 rotations)
  let non_diametric := (total_ways - 24) / 4
  -- Calculate the count for diametrically opposite positions (2 rotations)
  let diametric := 24 / 2
  -- Sum both counts
  non_diametric + diametric

theorem color_schemes_equivalence (n : Nat) (h : n = 7) : number_of_non_equivalent_color_schemes n = 300 :=
  by
    rw [h]
    sorry

end NUMINAMATH_GPT_color_schemes_equivalence_l1671_167187


namespace NUMINAMATH_GPT_poly_divisible_by_seven_l1671_167142

-- Define the given polynomial expression
def poly_expr (x n : ℕ) : ℕ := (1 + x)^n - 1

-- Define the proof statement
theorem poly_divisible_by_seven :
  ∀ x n : ℕ, x = 5 ∧ n = 4 → poly_expr x n % 7 = 0 :=
by
  intro x n h
  cases h
  sorry

end NUMINAMATH_GPT_poly_divisible_by_seven_l1671_167142


namespace NUMINAMATH_GPT_derivative_y_l1671_167117

noncomputable def y (x : ℝ) : ℝ := 
  Real.arcsin (1 / (2 * x + 3)) + 2 * Real.sqrt (x^2 + 3 * x + 2)

variable {x : ℝ}

theorem derivative_y :
  2 * x + 3 > 0 → 
  HasDerivAt y (4 * Real.sqrt (x^2 + 3 * x + 2) / (2 * x + 3)) x :=
by 
  sorry

end NUMINAMATH_GPT_derivative_y_l1671_167117


namespace NUMINAMATH_GPT_dandelion_seed_production_l1671_167113

theorem dandelion_seed_production :
  ∀ (initial_seeds : ℕ), initial_seeds = 50 →
  ∀ (germination_rate : ℚ), germination_rate = 1 / 2 →
  ∀ (new_seed_rate : ℕ), new_seed_rate = 50 →
  (initial_seeds * germination_rate * new_seed_rate) = 1250 :=
by
  intros initial_seeds h1 germination_rate h2 new_seed_rate h3
  sorry

end NUMINAMATH_GPT_dandelion_seed_production_l1671_167113


namespace NUMINAMATH_GPT_number_is_24point2_l1671_167101

noncomputable def certain_number (x : ℝ) : Prop :=
  0.12 * x = 2.904

theorem number_is_24point2 : certain_number 24.2 :=
by
  unfold certain_number
  sorry

end NUMINAMATH_GPT_number_is_24point2_l1671_167101


namespace NUMINAMATH_GPT_find_m_plus_n_l1671_167108

theorem find_m_plus_n (PQ QR RP : ℕ) (x y : ℕ) 
  (h1 : PQ = 26) 
  (h2 : QR = 29) 
  (h3 : RP = 25) 
  (h4 : PQ = x + y) 
  (h5 : QR = x + (QR - x))
  (h6 : RP = x + (RP - x)) : 
  30 = 29 + 1 :=
by
  -- assumptions already provided in problem statement
  sorry

end NUMINAMATH_GPT_find_m_plus_n_l1671_167108


namespace NUMINAMATH_GPT_number_difference_l1671_167159

theorem number_difference (a b : ℕ) (h1 : a + b = 25650) (h2 : a % 100 = 0) (h3 : b = a / 100) :
  a - b = 25146 :=
sorry

end NUMINAMATH_GPT_number_difference_l1671_167159


namespace NUMINAMATH_GPT_find_number_lemma_l1671_167140

theorem find_number_lemma (x : ℝ) (a b c d : ℝ) (h₁ : x = 5) 
  (h₂ : a = 0.47 * 1442) (h₃ : b = 0.36 * 1412) 
  (h₄ : c = a - b) (h₅ : d + c = x) : 
  d = -164.42 :=
by
  sorry

end NUMINAMATH_GPT_find_number_lemma_l1671_167140


namespace NUMINAMATH_GPT_max_hours_worked_l1671_167141

theorem max_hours_worked
  (r : ℝ := 8)  -- Regular hourly rate
  (h_r : ℝ := 20)  -- Hours at regular rate
  (r_o : ℝ := r + 0.25 * r)  -- Overtime hourly rate
  (E : ℝ := 410)  -- Total weekly earnings
  : (h_r + (E - r * h_r) / r_o) = 45 :=
by
  sorry

end NUMINAMATH_GPT_max_hours_worked_l1671_167141


namespace NUMINAMATH_GPT_consecutive_rolls_probability_l1671_167191

theorem consecutive_rolls_probability : 
  let total_outcomes := 36
  let consecutive_events := 10
  (consecutive_events / total_outcomes : ℚ) = 5 / 18 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_rolls_probability_l1671_167191


namespace NUMINAMATH_GPT_no_solution_of_abs_sum_l1671_167195

theorem no_solution_of_abs_sum (a : ℝ) : (∀ x : ℝ, |x - 2| + |x + 3| < a → false) ↔ a ≤ 5 := sorry

end NUMINAMATH_GPT_no_solution_of_abs_sum_l1671_167195


namespace NUMINAMATH_GPT_no_strictly_greater_polynomials_l1671_167197

noncomputable def transformation (P : Polynomial ℝ) (k : ℕ) (a : ℝ) : Polynomial ℝ := 
  P + Polynomial.monomial k (2 * a) - Polynomial.monomial (k + 1) a

theorem no_strictly_greater_polynomials (P Q : Polynomial ℝ) 
  (H1 : ∃ (n : ℕ) (a : ℝ), Q = transformation P n a)
  (H2 : ∃ (n : ℕ) (a : ℝ), P = transformation Q n a) : 
  ∃ x : ℝ, P.eval x = Q.eval x :=
sorry

end NUMINAMATH_GPT_no_strictly_greater_polynomials_l1671_167197


namespace NUMINAMATH_GPT_inequality_proof_l1671_167125

theorem inequality_proof (x y : ℝ) (hx : x > -1) (hy : y > -1) (hxy : x + y = 1) :
  (x / (y + 1) + y / (x + 1)) ≥ (2 / 3) ∧ (x = 1 / 2 ∧ y = 1 / 2 → x / (y + 1) + y / (x + 1) = 2 / 3) := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1671_167125


namespace NUMINAMATH_GPT_evaluate_fraction_l1671_167185

theorem evaluate_fraction (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : x - 1 / y ≠ 0) :
  (y - 1 / x) / (x - 1 / y) + y / x = 2 * y / x :=
by sorry

end NUMINAMATH_GPT_evaluate_fraction_l1671_167185


namespace NUMINAMATH_GPT_n_gon_angles_l1671_167112

theorem n_gon_angles (n : ℕ) (h1 : n > 7) (h2 : n < 12) : 
  (∃ x : ℝ, (150 * (n - 1) + x = 180 * (n - 2)) ∧ (x < 150)) :=
by {
  sorry
}

end NUMINAMATH_GPT_n_gon_angles_l1671_167112


namespace NUMINAMATH_GPT_store_A_cheaper_than_store_B_l1671_167172

noncomputable def store_A_full_price : ℝ := 125
noncomputable def store_A_discount_pct : ℝ := 0.08
noncomputable def store_B_full_price : ℝ := 130
noncomputable def store_B_discount_pct : ℝ := 0.10

noncomputable def final_price_A : ℝ :=
  store_A_full_price * (1 - store_A_discount_pct)

noncomputable def final_price_B : ℝ :=
  store_B_full_price * (1 - store_B_discount_pct)

theorem store_A_cheaper_than_store_B :
  final_price_B - final_price_A = 2 :=
by
  sorry

end NUMINAMATH_GPT_store_A_cheaper_than_store_B_l1671_167172


namespace NUMINAMATH_GPT_birth_rate_calculation_l1671_167178

theorem birth_rate_calculation (D : ℕ) (G : ℕ) (P : ℕ) (NetGrowth : ℕ) (B : ℕ) (h1 : D = 16) (h2 : G = 12) (h3 : P = 3000) (h4 : NetGrowth = G * P / 100) (h5 : NetGrowth = B - D) : B = 52 := by
  sorry

end NUMINAMATH_GPT_birth_rate_calculation_l1671_167178


namespace NUMINAMATH_GPT_incorrect_conclusion_l1671_167104

theorem incorrect_conclusion (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : 1/a < 1/b ∧ 1/b < 0) : ¬ (ab > b^2) :=
by
  { sorry }

end NUMINAMATH_GPT_incorrect_conclusion_l1671_167104


namespace NUMINAMATH_GPT_congruence_solution_count_l1671_167166

theorem congruence_solution_count :
  ∃! x : ℕ, x < 50 ∧ x + 20 ≡ 75 [MOD 43] := 
by
  sorry

end NUMINAMATH_GPT_congruence_solution_count_l1671_167166


namespace NUMINAMATH_GPT_parallel_line_through_point_l1671_167183

-- Problem: Prove the equation of the line that passes through the point (1, 1)
-- and is parallel to the line 2x - y + 1 = 0 is 2x - y - 1 = 0.

theorem parallel_line_through_point (x y : ℝ) (c : ℝ) :
  (2*x - y + 1 = 0) → (x = 1) → (y = 1) → (2*1 - 1 + c = 0) → c = -1 → (2*x - y - 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_parallel_line_through_point_l1671_167183


namespace NUMINAMATH_GPT_profit_made_after_two_years_l1671_167176

variable (present_value : ℝ) (depreciation_rate : ℝ) (selling_price : ℝ) 

def value_after_one_year (present_value depreciation_rate : ℝ) : ℝ :=
  present_value - (depreciation_rate * present_value)

def value_after_two_years (value_after_one_year : ℝ) (depreciation_rate : ℝ) : ℝ :=
  value_after_one_year - (depreciation_rate * value_after_one_year)

def profit (selling_price value_after_two_years : ℝ) : ℝ :=
  selling_price - value_after_two_years

theorem profit_made_after_two_years
  (h_present_value : present_value = 150000)
  (h_depreciation_rate : depreciation_rate = 0.22)
  (h_selling_price : selling_price = 115260) :
  profit selling_price (value_after_two_years (value_after_one_year present_value depreciation_rate) depreciation_rate) = 24000 := 
by
  sorry

end NUMINAMATH_GPT_profit_made_after_two_years_l1671_167176


namespace NUMINAMATH_GPT_trig_expression_simplification_l1671_167177

theorem trig_expression_simplification :
  ∃ a b : ℕ, 
  0 < b ∧ b < 90 ∧ 
  (1000 * Real.sin (10 * Real.pi / 180) * Real.cos (20 * Real.pi / 180) * Real.cos (30 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) = ↑a * Real.sin (b * Real.pi / 180)) ∧ 
  (100 * a + b = 12560) :=
sorry

end NUMINAMATH_GPT_trig_expression_simplification_l1671_167177


namespace NUMINAMATH_GPT_blue_paint_cans_needed_l1671_167186

theorem blue_paint_cans_needed (ratio_bg : ℤ × ℤ) (total_cans : ℤ) (r : ratio_bg = (4, 3)) (t : total_cans = 42) :
  let ratio_bw : ℚ := 4 / (4 + 3) 
  let blue_cans : ℚ := ratio_bw * total_cans 
  blue_cans = 24 :=
by
  sorry

end NUMINAMATH_GPT_blue_paint_cans_needed_l1671_167186


namespace NUMINAMATH_GPT_basketball_player_height_l1671_167123

noncomputable def player_height (H : ℝ) : Prop :=
  let reach := 22 / 12
  let jump := 32 / 12
  let total_rim_height := 10 + (6 / 12)
  H + reach + jump = total_rim_height

theorem basketball_player_height : ∃ H : ℝ, player_height H → H = 6 :=
by
  use 6
  sorry

end NUMINAMATH_GPT_basketball_player_height_l1671_167123


namespace NUMINAMATH_GPT_ratio_soda_water_l1671_167122

variables (W S : ℕ) (k : ℕ)

-- Conditions of the problem
def condition1 : Prop := S = k * W - 6
def condition2 : Prop := W + S = 54
def positive_integer_k : Prop := k > 0

-- The theorem we want to prove
theorem ratio_soda_water (h1 : condition1 W S k) (h2 : condition2 W S) (h3 : positive_integer_k k) : S / gcd S W = 4 ∧ W / gcd S W = 5 :=
sorry

end NUMINAMATH_GPT_ratio_soda_water_l1671_167122


namespace NUMINAMATH_GPT_actual_average_speed_l1671_167152

theorem actual_average_speed 
  (v t : ℝ)
  (h : v * t = (v + 21) * (2/3) * t) : 
  v = 42 :=
by
  sorry

end NUMINAMATH_GPT_actual_average_speed_l1671_167152


namespace NUMINAMATH_GPT_angle_difference_l1671_167149

theorem angle_difference (A B : ℝ) 
  (h1 : A = 85) 
  (h2 : A + B = 180) : B - A = 10 := 
by sorry

end NUMINAMATH_GPT_angle_difference_l1671_167149


namespace NUMINAMATH_GPT_jan_25_on_thursday_l1671_167121

/-- 
  Given that December 25 is on Monday,
  prove that January 25 in the following year falls on Thursday.
-/
theorem jan_25_on_thursday (day_of_week : Fin 7) (h : day_of_week = 0) : 
  ((day_of_week + 31) % 7 + 25) % 7 = 4 := 
sorry

end NUMINAMATH_GPT_jan_25_on_thursday_l1671_167121


namespace NUMINAMATH_GPT_find_X_value_l1671_167173

-- Given definitions and conditions
def X (n : ℕ) : ℕ := 3 + 2 * (n - 1)
def S (n : ℕ) : ℕ := n * (n + 2)

-- Proposition we need to prove
theorem find_X_value : ∃ n : ℕ, S n ≥ 10000 ∧ X n = 201 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_find_X_value_l1671_167173


namespace NUMINAMATH_GPT_simplify_expression_l1671_167193

theorem simplify_expression (y : ℝ) : 7 * y + 8 - 3 * y + 16 = 4 * y + 24 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1671_167193


namespace NUMINAMATH_GPT_dog_weight_ratio_l1671_167111

theorem dog_weight_ratio :
  ∀ (brown black white grey : ℕ),
    brown = 4 →
    black = brown + 1 →
    grey = black - 2 →
    (brown + black + white + grey) / 4 = 5 →
    white / brown = 2 :=
by
  intros brown black white grey h_brown h_black h_grey h_avg
  sorry

end NUMINAMATH_GPT_dog_weight_ratio_l1671_167111


namespace NUMINAMATH_GPT_distinct_solutions_l1671_167146

theorem distinct_solutions : 
  ∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (|x1 - 7| = 2 * |x1 + 1| + |x1 - 3| ∧ |x2 - 7| = 2 * |x2 + 1| + |x2 - 3|) := 
by
  sorry

end NUMINAMATH_GPT_distinct_solutions_l1671_167146


namespace NUMINAMATH_GPT_probability_of_winning_pair_is_correct_l1671_167119

noncomputable def probability_of_winning_pair : ℚ :=
  let total_cards := 10
  let red_cards := 5
  let blue_cards := 5
  let total_ways := Nat.choose total_cards 2 -- Combination C(10,2)
  let same_color_ways := Nat.choose red_cards 2 + Nat.choose blue_cards 2 -- Combination C(5,2) for each color
  let consecutive_pairs_per_color := 4
  let consecutive_ways := 2 * consecutive_pairs_per_color -- Two colors
  let favorable_ways := same_color_ways + consecutive_ways
  favorable_ways / total_ways

theorem probability_of_winning_pair_is_correct : 
  probability_of_winning_pair = 28 / 45 := sorry

end NUMINAMATH_GPT_probability_of_winning_pair_is_correct_l1671_167119


namespace NUMINAMATH_GPT_monotonicity_intervals_m0_monotonicity_intervals_m_positive_intersection_points_m1_inequality_a_b_l1671_167167

noncomputable def f (x m : ℝ) : ℝ := x - m * (x + 1) * Real.log (x + 1)

theorem monotonicity_intervals_m0 :
  ∀ x : ℝ, x > -1 → f x 0 = x - 0 * (x + 1) * Real.log (x + 1) ∧ f x 0 > 0 := 
sorry

theorem monotonicity_intervals_m_positive (m : ℝ) (hm : m > 0) :
  ∀ x : ℝ, x > -1 → 
  (f x m > f (x + e ^ ((1 - m) / m) - 1) m ∧ 
  f (x + e ^ ((1 - m) / m) - 1) m < f (x + e ^ ((1 - m) / m) - 1 + 1) m) :=
sorry

theorem intersection_points_m1 (t : ℝ) (hx_rng : -1 / 2 ≤ t ∧ t < 1) :
  (∃ x1 x2 : ℝ, x1 > -1/2 ∧ x1 ≤ 1 ∧ x2 > -1/2 ∧ x2 ≤ 1 ∧ f x1 1 = t ∧ f x2 1 = t) ↔ 
  (-1 / 2 + 1 / 2 * Real.log 2 ≤ t ∧ t < 0) :=
sorry

theorem inequality_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (1 + a) ^ b < (1 + b) ^ a :=
sorry

end NUMINAMATH_GPT_monotonicity_intervals_m0_monotonicity_intervals_m_positive_intersection_points_m1_inequality_a_b_l1671_167167


namespace NUMINAMATH_GPT_initial_speed_of_car_l1671_167174

-- Definition of conditions
def distance_from_A_to_B := 100  -- km
def time_remaining_first_reduction := 30 / 60  -- hours
def speed_reduction_first := 10  -- km/h
def time_remaining_second_reduction := 20 / 60  -- hours
def speed_reduction_second := 10  -- km/h
def additional_time_reduced_speeds := 5 / 60  -- hours

-- Variables for initial speed and intermediate distances
variables (v x : ℝ)

-- Proposition to prove the initial speed
theorem initial_speed_of_car :
  (100 - (v / 2 + x + 20)) / v + 
  (v / 2) / (v - 10) + 
  20 / (v - 20) - 
  20 / (v - 10) 
  = 5 / 60 →
  v = 100 :=
by
  sorry

end NUMINAMATH_GPT_initial_speed_of_car_l1671_167174


namespace NUMINAMATH_GPT_parabola_tangents_coprime_l1671_167165

theorem parabola_tangents_coprime {d e f : ℤ} (hd : d ≠ 0) (he : e ≠ 0)
  (h_coprime: Int.gcd (Int.gcd d e) f = 1)
  (h_tangent1 : d^2 - 4 * e * (2 * e - f) = 0)
  (h_tangent2 : (e + d)^2 - 4 * d * (8 * d - f) = 0) :
  d + e + f = 8 := by
  sorry

end NUMINAMATH_GPT_parabola_tangents_coprime_l1671_167165


namespace NUMINAMATH_GPT_total_fish_caught_l1671_167138

theorem total_fish_caught (leo_fish : ℕ) (agrey_fish : ℕ) 
  (h₁ : leo_fish = 40) (h₂ : agrey_fish = leo_fish + 20) : 
  leo_fish + agrey_fish = 100 := 
by 
  sorry

end NUMINAMATH_GPT_total_fish_caught_l1671_167138


namespace NUMINAMATH_GPT_total_sales_l1671_167109

theorem total_sales (S : ℕ) (h1 : (1 / 3 : ℚ) * S + (1 / 4 : ℚ) * S = (1 - (1 / 3 + 1 / 4)) * S + 15) : S = 36 :=
by
  sorry

end NUMINAMATH_GPT_total_sales_l1671_167109


namespace NUMINAMATH_GPT_distinct_roots_of_transformed_polynomial_l1671_167181

theorem distinct_roots_of_transformed_polynomial
  (a b c : ℝ)
  (h : ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
                    (a * x^5 + b * x^4 + c = 0) ∧ 
                    (a * y^5 + b * y^4 + c = 0) ∧ 
                    (a * z^5 + b * z^4 + c = 0)) :
  ∃ u v w : ℝ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ 
               (c * u^5 + b * u + a = 0) ∧ 
               (c * v^5 + b * v + a = 0) ∧ 
               (c * w^5 + b * w + a = 0) :=
  sorry

end NUMINAMATH_GPT_distinct_roots_of_transformed_polynomial_l1671_167181


namespace NUMINAMATH_GPT_valid_combinations_l1671_167190

-- Definitions based on conditions
def h : Nat := 4  -- number of herbs
def c : Nat := 6  -- number of crystals
def r : Nat := 3  -- number of negative reactions

-- Theorem statement based on the problem and solution
theorem valid_combinations : (h * c) - r = 21 := by
  sorry

end NUMINAMATH_GPT_valid_combinations_l1671_167190


namespace NUMINAMATH_GPT_triangle_square_ratio_l1671_167147

theorem triangle_square_ratio :
  ∀ (x y : ℝ), (x = 60 / 17) → (y = 780 / 169) → (x / y = 78 / 102) :=
by
  intros x y hx hy
  rw [hx, hy]
  -- the proof is skipped, as instructed
  sorry

end NUMINAMATH_GPT_triangle_square_ratio_l1671_167147


namespace NUMINAMATH_GPT_intersection_M_N_l1671_167192

-- Definitions based on the conditions
def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | x^2 - 25 < 0}

-- Theorem asserting the intersection of sets M and N
theorem intersection_M_N : M ∩ N = {x | 2 ≤ x ∧ x < 5} := 
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1671_167192


namespace NUMINAMATH_GPT_find_weight_of_silver_in_metal_bar_l1671_167120

noncomputable def weight_loss_ratio_tin : ℝ := 1.375 / 10
noncomputable def weight_loss_ratio_silver : ℝ := 0.375
noncomputable def ratio_tin_silver : ℝ := 0.6666666666666664

theorem find_weight_of_silver_in_metal_bar (T S : ℝ)
  (h1 : T + S = 70)
  (h2 : T / S = ratio_tin_silver)
  (h3 : weight_loss_ratio_tin * T + weight_loss_ratio_silver * S = 7) :
  S = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_weight_of_silver_in_metal_bar_l1671_167120


namespace NUMINAMATH_GPT_weeks_to_buy_iphone_l1671_167161

-- Definitions based on conditions
def iphone_cost : ℝ := 800
def trade_in_value : ℝ := 240
def earnings_per_week : ℝ := 80

-- Mathematically equivalent proof problem
theorem weeks_to_buy_iphone : 
  ∀ (iphone_cost trade_in_value earnings_per_week : ℝ), 
  (iphone_cost - trade_in_value) / earnings_per_week = 7 :=
by
  -- Using the given conditions directly.
  intros iphone_cost trade_in_value earnings_per_week
  sorry

end NUMINAMATH_GPT_weeks_to_buy_iphone_l1671_167161


namespace NUMINAMATH_GPT_max_sum_abc_l1671_167198

theorem max_sum_abc
  (a b c : ℤ)
  (A : Matrix (Fin 2) (Fin 2) ℚ)
  (hA1 : A = (1/7 : ℚ) • ![![(-5 : ℚ), a], ![b, c]])
  (hA2 : A * A = 2 • (1 : Matrix (Fin 2) (Fin 2) ℚ)) :
  a + b + c ≤ 79 :=
by
  sorry

end NUMINAMATH_GPT_max_sum_abc_l1671_167198


namespace NUMINAMATH_GPT_age_of_B_l1671_167153

theorem age_of_B (A B C : ℕ) (h1 : A = 2 * C + 2) (h2 : B = 2 * C) (h3 : A + B + C = 27) : B = 10 :=
by
  sorry

end NUMINAMATH_GPT_age_of_B_l1671_167153


namespace NUMINAMATH_GPT_logically_equivalent_to_original_l1671_167135

def original_statement (E W : Prop) : Prop := E → ¬ W
def statement_I (E W : Prop) : Prop := W → E
def statement_II (E W : Prop) : Prop := ¬ E → ¬ W
def statement_III (E W : Prop) : Prop := W → ¬ E
def statement_IV (E W : Prop) : Prop := ¬ E ∨ ¬ W

theorem logically_equivalent_to_original (E W : Prop) :
  (original_statement E W ↔ statement_III E W) ∧
  (original_statement E W ↔ statement_IV E W) :=
  sorry

end NUMINAMATH_GPT_logically_equivalent_to_original_l1671_167135


namespace NUMINAMATH_GPT_sum_of_squares_of_consecutive_integers_l1671_167115

theorem sum_of_squares_of_consecutive_integers (a : ℝ) (h : (a-1)*a*(a+1) = 36*a) :
  (a-1)^2 + a^2 + (a+1)^2 = 77 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_consecutive_integers_l1671_167115


namespace NUMINAMATH_GPT_parabola_focus_l1671_167171

-- Definitions and conditions from the original problem
def parabola_eq (x y : ℝ) : Prop := x^2 = (1/2) * y 

-- Define the problem to prove the coordinates of the focus
theorem parabola_focus (x y : ℝ) (h : parabola_eq x y) : (x = 0 ∧ y = 1/8) :=
sorry

end NUMINAMATH_GPT_parabola_focus_l1671_167171


namespace NUMINAMATH_GPT_right_triangle_ratio_is_4_l1671_167116

noncomputable def right_triangle_rectangle_ratio (b h xy : ℝ) : Prop :=
  (0.4 * (1/2) * b * h = 0.25 * xy) ∧ (xy = b * h) → (b / h = 4)

theorem right_triangle_ratio_is_4 (b h xy : ℝ) (h1 : 0.4 * (1/2) * b * h = 0.25 * xy)
(h2 : xy = b * h) : b / h = 4 :=
sorry

end NUMINAMATH_GPT_right_triangle_ratio_is_4_l1671_167116


namespace NUMINAMATH_GPT_probability_red_side_given_observed_l1671_167157

def total_cards : ℕ := 9
def black_black_cards : ℕ := 4
def black_red_cards : ℕ := 2
def red_red_cards : ℕ := 3

def red_sides : ℕ := red_red_cards * 2 + black_red_cards
def red_red_sides : ℕ := red_red_cards * 2
def probability_other_side_is_red (total_red_sides red_red_sides : ℕ) : ℚ :=
  red_red_sides / total_red_sides

theorem probability_red_side_given_observed :
  probability_other_side_is_red red_sides red_red_sides = 3 / 4 :=
by
  unfold red_sides
  unfold red_red_sides
  unfold probability_other_side_is_red
  sorry

end NUMINAMATH_GPT_probability_red_side_given_observed_l1671_167157


namespace NUMINAMATH_GPT_no_elimination_method_l1671_167130

theorem no_elimination_method
  (x y : ℤ)
  (h1 : x + 3 * y = 4)
  (h2 : 2 * x - y = 1) :
  ¬ (∀ z : ℤ, z = x + 3 * y - 3 * (2 * x - y)) →
  ∃ x y : ℤ, x + 3 * y - 3 * (2 * x - y) ≠ 0 := sorry

end NUMINAMATH_GPT_no_elimination_method_l1671_167130


namespace NUMINAMATH_GPT_pq_even_impossible_l1671_167150

theorem pq_even_impossible {p q : ℤ} (h : (p^2 + q^2 + p*q) % 2 = 1) : ¬(p % 2 = 0 ∧ q % 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_pq_even_impossible_l1671_167150


namespace NUMINAMATH_GPT_correct_calculation_l1671_167145

theorem correct_calculation : ∀ (a : ℝ), a^3 * a^2 = a^5 := 
by
  intro a
  sorry

end NUMINAMATH_GPT_correct_calculation_l1671_167145


namespace NUMINAMATH_GPT_correct_judgments_about_f_l1671_167107

-- Define the function f with its properties
variable {f : ℝ → ℝ} 

-- f is an even function
axiom even_function : ∀ x, f (-x) = f x

-- f satisfies f(x + 1) = -f(x)
axiom function_property : ∀ x, f (x + 1) = -f x

-- f is increasing on [-1, 0]
axiom increasing_on_interval : ∀ x y, -1 ≤ x → x ≤ y → y ≤ 0 → f x ≤ f y

theorem correct_judgments_about_f :
  (∀ x, f x = f (x + 2)) ∧
  (∀ x, f x = f (-x + 2)) ∧
  (f 2 = f 0) :=
by 
  sorry

end NUMINAMATH_GPT_correct_judgments_about_f_l1671_167107


namespace NUMINAMATH_GPT_find_x_values_l1671_167151

theorem find_x_values (x : ℝ) : 
  ((x + 1)^2 = 36 ∨ (x + 10)^3 = -27) ↔ (x = 5 ∨ x = -7 ∨ x = -13) :=
by
  sorry

end NUMINAMATH_GPT_find_x_values_l1671_167151


namespace NUMINAMATH_GPT_green_fish_always_15_l1671_167128

def total_fish (T : ℕ) : Prop :=
∃ (O B G : ℕ),
B = T / 2 ∧
O = B - 15 ∧
T = B + O + G ∧
G = 15

theorem green_fish_always_15 (T : ℕ) : total_fish T → ∃ G, G = 15 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_green_fish_always_15_l1671_167128


namespace NUMINAMATH_GPT_parabola_tangent_line_l1671_167148

noncomputable def gcd (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

theorem parabola_tangent_line (a b c : ℕ) (h1 : a^2 + (104 / 5) * a * b - 4 * b * c = 0)
  (h2 : b^2 - 5 * a^2 + 4 * a * c = 0) (hgcd : gcd a b c = 1) :
  a + b + c = 17 := by
  sorry

end NUMINAMATH_GPT_parabola_tangent_line_l1671_167148


namespace NUMINAMATH_GPT_solve_problem_l1671_167164

open Real

noncomputable def problem_statement : Prop :=
  ∃ (p q : ℝ), 1 < p ∧ p < q ∧ (1 / p + 1 / q = 1) ∧ (p * q = 8) ∧ (q = 4 + 2 * sqrt 2)
  
theorem solve_problem : problem_statement :=
sorry

end NUMINAMATH_GPT_solve_problem_l1671_167164


namespace NUMINAMATH_GPT_determine_x_l1671_167199

variable (A B C x : ℝ)
variable (hA : A = x)
variable (hB : B = 2 * x)
variable (hC : C = 45)
variable (hSum : A + B + C = 180)

theorem determine_x : x = 45 := 
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_determine_x_l1671_167199


namespace NUMINAMATH_GPT_smallest_hamburger_packages_l1671_167184

theorem smallest_hamburger_packages (h_num : ℕ) (b_num : ℕ) (h_bag_num : h_num = 10) (b_bag_num : b_num = 15) :
  ∃ (n : ℕ), n = 3 ∧ (n * h_num) = (2 * b_num) := by
  sorry

end NUMINAMATH_GPT_smallest_hamburger_packages_l1671_167184


namespace NUMINAMATH_GPT_arithmetic_sum_sequences_l1671_167110

theorem arithmetic_sum_sequences (a b : ℕ → ℕ) (h1 : ∀ n, a n = a 0 + n * (a 1 - a 0)) (h2 : ∀ n, b n = b 0 + n * (b 1 - b 0)) (h3 : a 2 + b 2 = 3) (h4 : a 4 + b 4 = 5): a 7 + b 7 = 8 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sum_sequences_l1671_167110


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1671_167154

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4) + f (x + 3 * Real.pi / 4)

theorem problem_1 : f (Real.pi / 2) = 1 := 
sorry

theorem problem_2 : (∃ p > 0, ∀ x, f (x + p) = f x) ∧ (∀ p, p > 0 ∧ (∀ x, f (x + p) = f x) → p ≥ 2 * Real.pi) := 
sorry

theorem problem_3 : ∃ x : ℝ, g x = -2 := 
sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1671_167154


namespace NUMINAMATH_GPT_sequence_formula_l1671_167133

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = -2) (h2 : a 2 = -1.2) :
  ∀ n, a n = 0.8 * n - 2.8 :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l1671_167133


namespace NUMINAMATH_GPT_total_flowers_l1671_167162

theorem total_flowers (initial_rosas_flowers andre_gifted_flowers : ℝ) 
  (h1 : initial_rosas_flowers = 67.0) 
  (h2 : andre_gifted_flowers = 90.0) : 
  initial_rosas_flowers + andre_gifted_flowers = 157.0 :=
  by
  sorry

end NUMINAMATH_GPT_total_flowers_l1671_167162


namespace NUMINAMATH_GPT_b_minus_a_l1671_167163

theorem b_minus_a (a b : ℕ) : (a * b = 2 * (a + b) + 12) → (b = 10) → (b - a = 6) :=
by
  sorry

end NUMINAMATH_GPT_b_minus_a_l1671_167163


namespace NUMINAMATH_GPT_small_gifts_combinations_large_gifts_combinations_l1671_167179

/-
  Definitions based on the given conditions:
  - 12 varieties of wrapping paper.
  - 3 colors of ribbon.
  - 6 types of gift cards.
  - Small gifts can use only 2 out of the 3 ribbon colors.
-/

def wrapping_paper_varieties : ℕ := 12
def ribbon_colors : ℕ := 3
def gift_card_types : ℕ := 6
def small_gift_ribbon_colors : ℕ := 2

/-
  Proof problems:

  - For small gifts, there are 12 * 2 * 6 combinations.
  - For large gifts, there are 12 * 3 * 6 combinations.
-/

theorem small_gifts_combinations :
  wrapping_paper_varieties * small_gift_ribbon_colors * gift_card_types = 144 :=
by
  sorry

theorem large_gifts_combinations :
  wrapping_paper_varieties * ribbon_colors * gift_card_types = 216 :=
by
  sorry

end NUMINAMATH_GPT_small_gifts_combinations_large_gifts_combinations_l1671_167179


namespace NUMINAMATH_GPT_boat_goes_6_km_upstream_l1671_167180

variable (speed_in_still_water : ℕ) (distance_downstream : ℕ) (time_downstream : ℕ) (effective_speed_downstream : ℕ) (speed_of_stream : ℕ)

-- Given conditions
def condition1 : Prop := speed_in_still_water = 11
def condition2 : Prop := distance_downstream = 16
def condition3 : Prop := time_downstream = 1
def condition4 : Prop := effective_speed_downstream = speed_in_still_water + speed_of_stream
def condition5 : Prop := effective_speed_downstream = 16

-- Prove that the boat goes 6 km against the stream in one hour.
theorem boat_goes_6_km_upstream : speed_of_stream = 5 →
  11 - 5 = 6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_boat_goes_6_km_upstream_l1671_167180


namespace NUMINAMATH_GPT_morgan_change_l1671_167155

-- Define the costs of the items and the amount paid
def hamburger_cost : ℕ := 4
def onion_rings_cost : ℕ := 2
def smoothie_cost : ℕ := 3
def amount_paid : ℕ := 20

-- Define total cost
def total_cost := hamburger_cost + onion_rings_cost + smoothie_cost

-- Define the change received
def change_received := amount_paid - total_cost

-- Statement of the problem in Lean 4
theorem morgan_change : change_received = 11 := by
  -- include proof steps here
  sorry

end NUMINAMATH_GPT_morgan_change_l1671_167155


namespace NUMINAMATH_GPT_value_of_expression_l1671_167156

theorem value_of_expression (x y : ℝ) (hy : y > 0) (h : x = 3 * y) :
  (x^y * y^x) / (y^y * x^x) = 3^(-2 * y) := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1671_167156


namespace NUMINAMATH_GPT_batsman_average_excluding_highest_and_lowest_l1671_167105

theorem batsman_average_excluding_highest_and_lowest (average : ℝ) (innings : ℕ) (highest_score : ℝ) (score_difference : ℝ) :
  average = 63 →
  innings = 46 →
  highest_score = 248 →
  score_difference = 150 →
  (average * innings - highest_score - (highest_score - score_difference)) / (innings - 2) = 58 :=
by
  intros h_average h_innings h_highest h_difference
  simp [h_average, h_innings, h_highest, h_difference]
  -- Here the detailed steps from the solution would come in to verify the simplification
  sorry

end NUMINAMATH_GPT_batsman_average_excluding_highest_and_lowest_l1671_167105


namespace NUMINAMATH_GPT_gnollish_valid_sentences_l1671_167106

def valid_sentences_count : ℕ :=
  let words := ["splargh", "glumph", "amr", "krack"]
  let total_words := 4
  let total_sentences := total_words ^ 3
  let invalid_splargh_glumph := 2 * total_words
  let invalid_amr_krack := 2 * total_words
  let total_invalid := invalid_splargh_glumph + invalid_amr_krack
  total_sentences - total_invalid

theorem gnollish_valid_sentences : valid_sentences_count = 48 :=
by
  sorry

end NUMINAMATH_GPT_gnollish_valid_sentences_l1671_167106


namespace NUMINAMATH_GPT_geometric_progression_common_ratio_l1671_167129

theorem geometric_progression_common_ratio (y r : ℝ) (h : (40 + y)^2 = (10 + y) * (90 + y)) :
  r = (40 + y) / (10 + y) → r = (90 + y) / (40 + y) → r = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_common_ratio_l1671_167129


namespace NUMINAMATH_GPT_cost_of_double_room_l1671_167189

theorem cost_of_double_room (total_rooms : ℕ) (cost_single_room : ℕ) (total_revenue : ℕ) 
  (double_rooms_booked : ℕ) (single_rooms_booked := total_rooms - double_rooms_booked) 
  (total_single_revenue := single_rooms_booked * cost_single_room) : 
  total_rooms = 260 → cost_single_room = 35 → total_revenue = 14000 → double_rooms_booked = 196 → 
  196 * 60 + 64 * 35 = total_revenue :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_cost_of_double_room_l1671_167189


namespace NUMINAMATH_GPT_emily_total_cost_l1671_167126

-- Definition of the monthly cell phone plan costs and usage details
def base_cost : ℝ := 30
def cost_per_text : ℝ := 0.10
def cost_per_extra_minute : ℝ := 0.15
def cost_per_extra_gb : ℝ := 5
def free_hours : ℝ := 25
def free_gb : ℝ := 15
def texts : ℝ := 150
def hours : ℝ := 26
def gb : ℝ := 16

-- Calculate the total cost
def total_cost : ℝ :=
  base_cost +
  (texts * cost_per_text) +
  ((hours - free_hours) * 60 * cost_per_extra_minute) +
  ((gb - free_gb) * cost_per_extra_gb)

-- The proof statement that Emily had to pay $59
theorem emily_total_cost :
  total_cost = 59 := by
  sorry

end NUMINAMATH_GPT_emily_total_cost_l1671_167126


namespace NUMINAMATH_GPT_juliet_age_l1671_167127

theorem juliet_age
    (M J R : ℕ)
    (h1 : J = M + 3)
    (h2 : J = R - 2)
    (h3 : M + R = 19) : J = 10 := by
  sorry

end NUMINAMATH_GPT_juliet_age_l1671_167127


namespace NUMINAMATH_GPT_angle_C_side_c_area_of_triangle_l1671_167132

open Real

variables (A B C a b c : Real)

noncomputable def acute_triangle (A B C a b c : ℝ) : Prop :=
  (A + B + C = π) ∧ (A > 0) ∧ (B > 0) ∧ (C > 0) ∧
  (A < π / 2) ∧ (B < π / 2) ∧ (C < π / 2) ∧
  (a^2 - 2 * sqrt 3 * a + 2 = 0) ∧
  (b^2 - 2 * sqrt 3 * b + 2 = 0) ∧
  (2 * sin (A + B) - sqrt 3 = 0)

noncomputable def length_side_c (a b : ℝ) : ℝ :=
  sqrt (a^2 + b^2 - 2 * a * b * cos (π / 3))

noncomputable def area_triangle (a b : ℝ) : ℝ := 
  (1 / 2) * a * b * sin (π / 3)

theorem angle_C (h : acute_triangle A B C a b c) : C = π / 3 :=
  sorry

theorem side_c (h : acute_triangle A B C a b c) : c = sqrt 6 :=
  sorry

theorem area_of_triangle (h : acute_triangle A B C a b c) : area_triangle a b = sqrt 3 / 2 :=
  sorry

end NUMINAMATH_GPT_angle_C_side_c_area_of_triangle_l1671_167132


namespace NUMINAMATH_GPT_arithmetic_progression_squares_l1671_167188

theorem arithmetic_progression_squares :
  ∃ (n : ℤ), ((3 * n^2 + 8 = 1111 * 5) ∧ (n-2, n, n+2) = (41, 43, 45)) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_squares_l1671_167188


namespace NUMINAMATH_GPT_speed_increase_impossible_l1671_167103

theorem speed_increase_impossible (v : ℝ) : v = 60 → (¬ ∃ v', (1 / (v' / 60) = 0)) :=
by sorry

end NUMINAMATH_GPT_speed_increase_impossible_l1671_167103


namespace NUMINAMATH_GPT_rhombus_area_of_square_4_l1671_167102

theorem rhombus_area_of_square_4 :
  let A := (0, 4)
  let B := (0, 0)
  let C := (4, 0)
  let D := (4, 4)
  let F := (0, 2)  -- Midpoint of AB
  let E := (4, 2)  -- Midpoint of CD
  let FG := 2 -- Half of the side of the square (since F and E are midpoints)
  let GH := 2
  let HE := 2
  let EF := 2
  let rhombus_FGEH_area := 1 / 2 * FG * EH
  rhombus_FGEH_area = 4 := sorry

end NUMINAMATH_GPT_rhombus_area_of_square_4_l1671_167102
