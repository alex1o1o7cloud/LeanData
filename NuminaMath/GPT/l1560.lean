import Mathlib

namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1560_156096

-- Define predicate conditions and solutions in Lean 4 for each problem

theorem problem1 (x : ℝ) :
  -2 * x^2 + 3 * x + 9 > 0 ↔ (-3 / 2 < x ∧ x < 3) := by
  sorry

theorem problem2 (x : ℝ) :
  (8 - x) / (5 + x) > 1 ↔ (-5 < x ∧ x ≤ 3 / 2) := by
  sorry

theorem problem3 (x : ℝ) :
  ¬ (-x^2 + 2 * x - 3 > 0) ↔ True := by
  sorry

theorem problem4 (x : ℝ) :
  x^2 - 14 * x + 50 > 0 ↔ True := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1560_156096


namespace NUMINAMATH_GPT_jose_internet_speed_l1560_156012

-- Define the given conditions
def file_size : ℕ := 160
def upload_time : ℕ := 20

-- Define the statement we need to prove
theorem jose_internet_speed : file_size / upload_time = 8 :=
by
  -- Proof should be provided here
  sorry

end NUMINAMATH_GPT_jose_internet_speed_l1560_156012


namespace NUMINAMATH_GPT_gunny_bag_can_hold_packets_l1560_156045

theorem gunny_bag_can_hold_packets :
  let ton_to_kg := 1000
  let max_capacity_tons := 13
  let pound_to_kg := 0.453592
  let ounce_to_g := 28.3495
  let kilo_to_g := 1000
  let wheat_packet_pounds := 16
  let wheat_packet_ounces := 4
  let max_capacity_kg := max_capacity_tons * ton_to_kg
  let wheat_packet_kg := wheat_packet_pounds * pound_to_kg + (wheat_packet_ounces * ounce_to_g) / kilo_to_g
  max_capacity_kg / wheat_packet_kg >= 1763 := 
by
  sorry

end NUMINAMATH_GPT_gunny_bag_can_hold_packets_l1560_156045


namespace NUMINAMATH_GPT_tan_alpha_sub_beta_l1560_156033

theorem tan_alpha_sub_beta
  (α β : ℝ)
  (h1 : Real.tan (α + Real.pi / 5) = 2)
  (h2 : Real.tan (β - 4 * Real.pi / 5) = -3) :
  Real.tan (α - β) = -1 := 
sorry

end NUMINAMATH_GPT_tan_alpha_sub_beta_l1560_156033


namespace NUMINAMATH_GPT_initial_soccer_balls_l1560_156077

theorem initial_soccer_balls {x : ℕ} (h : x + 18 = 24) : x = 6 := 
sorry

end NUMINAMATH_GPT_initial_soccer_balls_l1560_156077


namespace NUMINAMATH_GPT_current_prices_l1560_156095

theorem current_prices (initial_ram_price initial_ssd_price : ℝ) 
  (ram_increase_1 ram_decrease_1 ram_decrease_2 : ℝ) 
  (ssd_increase_1 ssd_decrease_1 ssd_decrease_2 : ℝ) 
  (initial_ram : initial_ram_price = 50) 
  (initial_ssd : initial_ssd_price = 100) 
  (ram_increase_factor : ram_increase_1 = 0.30 * initial_ram_price) 
  (ram_decrease_factor_1 : ram_decrease_1 = 0.15 * (initial_ram_price + ram_increase_1)) 
  (ram_decrease_factor_2 : ram_decrease_2 = 0.20 * ((initial_ram_price + ram_increase_1) - ram_decrease_1)) 
  (ssd_increase_factor : ssd_increase_1 = 0.10 * initial_ssd_price) 
  (ssd_decrease_factor_1 : ssd_decrease_1 = 0.05 * (initial_ssd_price + ssd_increase_1)) 
  (ssd_decrease_factor_2 : ssd_decrease_2 = 0.12 * ((initial_ssd_price + ssd_increase_1) - ssd_decrease_1)) 
  : 
  ((initial_ram_price + ram_increase_1 - ram_decrease_1 - ram_decrease_2) = 44.20) ∧ 
  ((initial_ssd_price + ssd_increase_1 - ssd_decrease_1 - ssd_decrease_2) = 91.96) := 
by
  sorry

end NUMINAMATH_GPT_current_prices_l1560_156095


namespace NUMINAMATH_GPT_find_Y_l1560_156036

-- Definition of the problem.
def arithmetic_sequence (a d n : ℕ) : ℕ := a + d * (n - 1)

-- Conditions provided in the problem.
-- Conditions of the first row
def first_row (a₁ a₄ : ℕ) : Prop :=
  a₁ = 4 ∧ a₄ = 16

-- Conditions of the last row
def last_row (a₁' a₄' : ℕ) : Prop :=
  a₁' = 10 ∧ a₄' = 40

-- Value of Y (the second element of the second row from the second column)
def center_top_element (Y : ℕ) : Prop :=
  Y = 12

-- The theorem to prove.
theorem find_Y (a₁ a₄ a₁' a₄' Y : ℕ) (h1 : first_row a₁ a₄) (h2 : last_row a₁' a₄') (h3 : center_top_element Y) : Y = 12 := 
by 
  sorry -- proof to be provided.

end NUMINAMATH_GPT_find_Y_l1560_156036


namespace NUMINAMATH_GPT_line_segment_length_is_0_7_l1560_156055

def isLineSegment (length : ℝ) (finite : Bool) : Prop :=
  finite = true ∧ length = 0.7

theorem line_segment_length_is_0_7 : isLineSegment 0.7 true :=
by
  sorry

end NUMINAMATH_GPT_line_segment_length_is_0_7_l1560_156055


namespace NUMINAMATH_GPT_min_expr_l1560_156069

theorem min_expr (a b c d : ℕ) (ha : Odd a) (hb : Odd b) (hc : Odd c) (hd : Odd d) (a_pos: 0 < a) (b_pos: 0 < b) (c_pos: 0 < c) (d_pos: 0 < d)
(h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  2 * a * b * c * d - (a * b * c + a * b * d + a * c * d + b * c * d) = 34 := 
sorry

end NUMINAMATH_GPT_min_expr_l1560_156069


namespace NUMINAMATH_GPT_meal_cost_with_tip_l1560_156037

theorem meal_cost_with_tip 
  (cost_samosas : ℕ := 3 * 2)
  (cost_pakoras : ℕ := 4 * 3)
  (cost_lassi : ℕ := 2)
  (total_cost_before_tip := cost_samosas + cost_pakoras + cost_lassi)
  (tip : ℝ := 0.25 * total_cost_before_tip) :
  (total_cost_before_tip + tip = 25) :=
sorry

end NUMINAMATH_GPT_meal_cost_with_tip_l1560_156037


namespace NUMINAMATH_GPT_tshirts_equation_l1560_156088

theorem tshirts_equation (x : ℝ) 
    (hx : x > 0)
    (march_cost : ℝ := 120000)
    (april_cost : ℝ := 187500)
    (april_increase : ℝ := 1.4)
    (cost_increase : ℝ := 5) :
    120000 / x + 5 = 187500 / (1.4 * x) :=
by 
  sorry

end NUMINAMATH_GPT_tshirts_equation_l1560_156088


namespace NUMINAMATH_GPT_price_per_working_game_eq_six_l1560_156091

-- Define the total number of video games
def total_games : Nat := 10

-- Define the number of non-working video games
def non_working_games : Nat := 8

-- Define the total income from selling working games
def total_earning : Nat := 12

-- Calculate the number of working video games
def working_games : Nat := total_games - non_working_games

-- Define the expected price per working game
def expected_price_per_game : Nat := 6

-- Theorem statement: Prove that the price per working game is $6
theorem price_per_working_game_eq_six :
  total_earning / working_games = expected_price_per_game :=
by sorry

end NUMINAMATH_GPT_price_per_working_game_eq_six_l1560_156091


namespace NUMINAMATH_GPT_find_colored_copies_l1560_156097

variable (cost_c cost_w total_copies total_cost : ℝ)
variable (colored_copies white_copies : ℝ)

def colored_copies_condition (cost_c cost_w total_copies total_cost : ℝ) :=
  ∃ (colored_copies white_copies : ℝ),
    colored_copies + white_copies = total_copies ∧
    cost_c * colored_copies + cost_w * white_copies = total_cost

theorem find_colored_copies :
  colored_copies_condition 0.10 0.05 400 22.50 → 
  ∃ (c : ℝ), c = 50 :=
by 
  sorry

end NUMINAMATH_GPT_find_colored_copies_l1560_156097


namespace NUMINAMATH_GPT_Eithan_savings_account_l1560_156098

variable (initial_amount wife_firstson_share firstson_remaining firstson_secondson_share 
          secondson_remaining secondson_thirdson_share thirdson_remaining 
          charity_donation remaining_after_charity tax_rate final_remaining : ℝ)

theorem Eithan_savings_account:
  initial_amount = 5000 →
  wife_firstson_share = initial_amount * (2/5) →
  firstson_remaining = initial_amount - wife_firstson_share →
  firstson_secondson_share = firstson_remaining * (3/10) →
  secondson_remaining = firstson_remaining - firstson_secondson_share →
  thirdson_remaining = secondson_remaining * (1-0.30) →
  charity_donation = 200 →
  remaining_after_charity = thirdson_remaining - charity_donation →
  tax_rate = 0.05 →
  final_remaining = remaining_after_charity * (1 - tax_rate) →
  final_remaining = 927.2 := 
  by
    intros
    sorry

end NUMINAMATH_GPT_Eithan_savings_account_l1560_156098


namespace NUMINAMATH_GPT_part1_part2_l1560_156021

noncomputable def f (x : ℝ) : ℝ :=
  abs (2 * x - 3) + abs (x - 5)

theorem part1 : { x : ℝ | f x ≥ 4 } = { x : ℝ | x ≥ 2 ∨ x ≤ 4 / 3 } :=
by
  sorry

theorem part2 (a : ℝ) : (∃ x : ℝ, f x < a) ↔ a > 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1560_156021


namespace NUMINAMATH_GPT_trig_identity_proof_l1560_156067

theorem trig_identity_proof :
  2 * (1 / 2) + (Real.sqrt 3 / 2) * Real.sqrt 3 = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_proof_l1560_156067


namespace NUMINAMATH_GPT_part_I_part_II_l1560_156000

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := log x - k * x

theorem part_I (k : ℝ) (hk : k = 1) :
  (∀ x, 0 < x ∧ x < 1 → 0 < f 1 x - f 1 1)
  ∧ (∀ x, 1 < x → f 1 1 > f 1 x)
  ∧ f 1 1 = 0 :=
by
  sorry

theorem part_II (k : ℝ) (h_no_zeros : ∀ x, f k x ≠ 0) :
  k > 1 / exp 1 :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1560_156000


namespace NUMINAMATH_GPT_max_points_for_top_teams_l1560_156057

-- Definitions based on the problem conditions
def points_for_win : ℕ := 3
def points_for_draw : ℕ := 1
def points_for_loss : ℕ := 0
def number_of_teams : ℕ := 8
def number_of_games_between_each_pair : ℕ := 2
def total_games : ℕ := (number_of_teams * (number_of_teams - 1) / 2) * number_of_games_between_each_pair
def total_points_in_tournament : ℕ := total_games * points_for_win
def top_teams : ℕ := 4

-- Theorem stating the correct answer
theorem max_points_for_top_teams : (total_points_in_tournament / number_of_teams = 33) :=
sorry

end NUMINAMATH_GPT_max_points_for_top_teams_l1560_156057


namespace NUMINAMATH_GPT_bus_interval_duration_l1560_156008

-- Definition of the conditions
def total_minutes : ℕ := 60
def total_buses : ℕ := 11
def intervals : ℕ := total_buses - 1

-- Theorem stating the interval between each bus departure
theorem bus_interval_duration : total_minutes / intervals = 6 := 
by
  -- The proof is omitted. 
  sorry

end NUMINAMATH_GPT_bus_interval_duration_l1560_156008


namespace NUMINAMATH_GPT_exists_i_for_inequality_l1560_156042

theorem exists_i_for_inequality (n : ℕ) (x : ℕ → ℝ) (h1 : 2 ≤ n) 
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → 0 ≤ x i ∧ x i ≤ 1) :
  ∃ i, 1 ≤ i ∧ i ≤ n - 1 ∧ x i * (1 - x (i + 1)) ≥ (1 / 4) * x 1 * (1 - x n) :=
by
  sorry

end NUMINAMATH_GPT_exists_i_for_inequality_l1560_156042


namespace NUMINAMATH_GPT_reduction_amount_is_250_l1560_156064

-- Definitions from the conditions
def original_price : ℝ := 500
def reduction_rate : ℝ := 0.5

-- The statement to be proved
theorem reduction_amount_is_250 : (reduction_rate * original_price) = 250 := by
  sorry

end NUMINAMATH_GPT_reduction_amount_is_250_l1560_156064


namespace NUMINAMATH_GPT_minute_hand_length_l1560_156028

theorem minute_hand_length 
  (arc_length : ℝ) (r : ℝ) (h : arc_length = 20 * (2 * Real.pi / 60) * r) :
  r = 1/2 :=
  sorry

end NUMINAMATH_GPT_minute_hand_length_l1560_156028


namespace NUMINAMATH_GPT_original_oil_weight_is_75_l1560_156040

def initial_oil_weight (original : ℝ) : Prop :=
  let first_remaining := original / 2
  let second_remaining := first_remaining * (4 / 5)
  second_remaining = 30

theorem original_oil_weight_is_75 : ∃ (original : ℝ), initial_oil_weight original ∧ original = 75 :=
by
  use 75
  unfold initial_oil_weight
  sorry

end NUMINAMATH_GPT_original_oil_weight_is_75_l1560_156040


namespace NUMINAMATH_GPT_solution_exists_l1560_156049

noncomputable def equation (x : ℝ) := 
  (x^2 - 5 * x + 4) / (x - 1) + (2 * x^2 + 7 * x - 4) / (2 * x - 1)

theorem solution_exists : equation 2 = 4 := by
  sorry

end NUMINAMATH_GPT_solution_exists_l1560_156049


namespace NUMINAMATH_GPT_tiling_rect_divisible_by_4_l1560_156009

theorem tiling_rect_divisible_by_4 (m n : ℕ) (h : ∃ k l : ℕ, m = 4 * k ∧ n = 4 * l) : 
  (∃ a : ℕ, m = 4 * a) ∧ (∃ b : ℕ, n = 4 * b) :=
by 
  sorry

end NUMINAMATH_GPT_tiling_rect_divisible_by_4_l1560_156009


namespace NUMINAMATH_GPT_find_t_l1560_156046

theorem find_t (t : ℝ) : 
  (∃ (m b : ℝ), (∀ x y, (y = m * x + b) → ((x = 1 ∧ y = 7) ∨ (x = 3 ∧ y = 13) ∨ (x = 5 ∧ y = 19))) ∧ (28 = 28 * m + b) ∧ (t = 28 * m + b)) → 
  t = 88 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l1560_156046


namespace NUMINAMATH_GPT_solve_for_k_in_quadratic_l1560_156058

theorem solve_for_k_in_quadratic :
  ∃ k : ℝ, (∀ x1 x2 : ℝ,
    x1 + x2 = 3 ∧
    x1 * x2 + 2 * x1 + 2 * x2 = 1 ∧
    (x1^2 - 3*x1 + k = 0) ∧ (x2^2 - 3*x2 + k = 0)) →
  k = -5 :=
sorry

end NUMINAMATH_GPT_solve_for_k_in_quadratic_l1560_156058


namespace NUMINAMATH_GPT_number_of_items_in_U_l1560_156068

theorem number_of_items_in_U (U A B : Finset ℕ)
  (hB : B.card = 41)
  (not_A_nor_B : U.card - A.card - B.card + (A ∩ B).card = 59)
  (hAB : (A ∩ B).card = 23)
  (hA : A.card = 116) :
  U.card = 193 :=
by sorry

end NUMINAMATH_GPT_number_of_items_in_U_l1560_156068


namespace NUMINAMATH_GPT_prime_p_equals_2_l1560_156050

theorem prime_p_equals_2 (p q r s : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs: Nat.Prime s)
  (h_sum : p + q + r = 2 * s) (h_order : 1 < p ∧ p < q ∧ q < r) : p = 2 :=
sorry

end NUMINAMATH_GPT_prime_p_equals_2_l1560_156050


namespace NUMINAMATH_GPT_police_emergency_number_prime_divisor_l1560_156048

theorem police_emergency_number_prime_divisor (n : ℕ) (k : ℕ) (h : n = 1000 * k + 133) : ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n :=
sorry

end NUMINAMATH_GPT_police_emergency_number_prime_divisor_l1560_156048


namespace NUMINAMATH_GPT_evaluate_expression_l1560_156010

theorem evaluate_expression (c d : ℝ) (h_c : c = 3) (h_d : d = 2) : 
  (c^2 + d + 1)^2 - (c^2 - d - 1)^2 = 80 := by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1560_156010


namespace NUMINAMATH_GPT_expand_expression_l1560_156073

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end NUMINAMATH_GPT_expand_expression_l1560_156073


namespace NUMINAMATH_GPT_sophia_read_more_pages_l1560_156060

variable (total_pages : ℝ) (finished_fraction : ℝ)
variable (pages_read : ℝ) (pages_left : ℝ) (pages_more : ℝ)

theorem sophia_read_more_pages :
  total_pages = 269.99999999999994 ∧
  finished_fraction = 2/3 ∧
  pages_read = finished_fraction * total_pages ∧
  pages_left = total_pages - pages_read →
  pages_more = pages_read - pages_left →
  pages_more = 90 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_sophia_read_more_pages_l1560_156060


namespace NUMINAMATH_GPT_absolute_value_half_angle_cosine_l1560_156007

theorem absolute_value_half_angle_cosine (x : ℝ) (h1 : Real.sin x = -5 / 13) (h2 : ∀ n : ℤ, (2 * n) * Real.pi < x ∧ x < (2 * n + 1) * Real.pi) :
  |Real.cos (x / 2)| = Real.sqrt 26 / 26 :=
sorry

end NUMINAMATH_GPT_absolute_value_half_angle_cosine_l1560_156007


namespace NUMINAMATH_GPT_monomial_sum_exponents_l1560_156059

theorem monomial_sum_exponents (m n : ℕ) (h₁ : m - 1 = 2) (h₂ : n = 2) : m^n = 9 := 
by
  sorry

end NUMINAMATH_GPT_monomial_sum_exponents_l1560_156059


namespace NUMINAMATH_GPT_fraction_videocassette_recorders_l1560_156029

variable (H : ℝ) (F : ℝ)

-- Conditions
variable (cable_TV_frac : ℝ := 1 / 5)
variable (both_frac : ℝ := 1 / 20)
variable (neither_frac : ℝ := 0.75)

-- Main theorem statement
theorem fraction_videocassette_recorders (H_pos : 0 < H) 
  (cable_tv : cable_TV_frac * H > 0)
  (both : both_frac * H > 0) 
  (neither : neither_frac * H > 0) :
  F = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_fraction_videocassette_recorders_l1560_156029


namespace NUMINAMATH_GPT_hyperbola_foci_y_axis_l1560_156076

theorem hyperbola_foci_y_axis (a b : ℝ) (h : ∀ x y : ℝ, a * x^2 + b * y^2 = 1 → (1/a < 0 ∧ 1/b > 0)) : a < 0 ∧ b > 0 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_foci_y_axis_l1560_156076


namespace NUMINAMATH_GPT_intervals_of_monotonicity_range_of_values_for_a_l1560_156056

noncomputable def f (a x : ℝ) : ℝ := x - a * Real.log x + (1 + a) / x

theorem intervals_of_monotonicity (a : ℝ) :
  (∀ x ∈ Set.Ioi 0, a ≤ -1 → deriv (f a) x > 0) ∧
  (∀ x ∈ Set.Ioc 0 (1 + a), -1 < a → deriv (f a) x < 0) ∧
  (∀ x ∈ Set.Ioi (1 + a), -1 < a → deriv (f a) x > 0) :=
sorry

theorem range_of_values_for_a (a : ℝ) (e : ℝ) (h : e = Real.exp 1) :
  (∀ x ∈ Set.Icc 1 e, f a x ≤ 0) → (a ≤ -2 ∨ a ≥ (e^2 + 1) / (e - 1)) :=
sorry

end NUMINAMATH_GPT_intervals_of_monotonicity_range_of_values_for_a_l1560_156056


namespace NUMINAMATH_GPT_inequality_d_l1560_156092

-- We define the polynomial f with integer coefficients
variable (f : ℤ → ℤ)

-- The function for f^k iteration
def iter (f: ℤ → ℤ) : ℕ → ℤ → ℤ
| 0, x => x
| (n + 1), x => f (iter f n x)

-- Definition of d(a, k) based on the problem statement
def d (a : ℤ) (k : ℕ) : ℝ := |(iter f k a : ℤ) - a|

-- Given condition that d(a, k) is positive
axiom d_pos (a : ℤ) (k : ℕ) : 0 < d f a k

-- The statement to be proved
theorem inequality_d (a : ℤ) (k : ℕ) : d f a k ≥ ↑k / 3 := by
  sorry

end NUMINAMATH_GPT_inequality_d_l1560_156092


namespace NUMINAMATH_GPT_LCM_quotient_l1560_156079

-- Define M as the least common multiple of integers from 12 to 25
def LCM_12_25 : ℕ := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 
                       (Nat.lcm 12 13) 14) 15) 16) 17) (Nat.lcm (Nat.lcm 
                       (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 18 19) 20) 21) 22) 23) 24)

-- Define N as the least common multiple of LCM_12_25, 36, 38, 40, 42, 44, 45
def N : ℕ := Nat.lcm LCM_12_25 (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 36 38) 40) 42) (Nat.lcm 44 45))

-- Prove that N / LCM_12_25 = 1
theorem LCM_quotient : N / LCM_12_25 = 1 := by
    sorry

end NUMINAMATH_GPT_LCM_quotient_l1560_156079


namespace NUMINAMATH_GPT_current_price_after_increase_and_decrease_l1560_156034

-- Define constants and conditions
def initial_price_RAM : ℝ := 50
def percent_increase : ℝ := 0.30
def percent_decrease : ℝ := 0.20

-- Define intermediate and final values based on conditions
def increased_price_RAM : ℝ := initial_price_RAM * (1 + percent_increase)
def final_price_RAM : ℝ := increased_price_RAM * (1 - percent_decrease)

-- Theorem stating the final result
theorem current_price_after_increase_and_decrease 
  (init_price : ℝ) 
  (inc : ℝ) 
  (dec : ℝ) 
  (final_price : ℝ) :
  init_price = 50 ∧ inc = 0.30 ∧ dec = 0.20 → final_price = 52 := 
  sorry

end NUMINAMATH_GPT_current_price_after_increase_and_decrease_l1560_156034


namespace NUMINAMATH_GPT_ratio_swordfish_to_pufferfish_l1560_156071

theorem ratio_swordfish_to_pufferfish (P S : ℕ) (n : ℕ) 
  (hP : P = 15)
  (hTotal : S + P = 90)
  (hRelation : S = n * P) : 
  (S : ℚ) / (P : ℚ) = 5 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_swordfish_to_pufferfish_l1560_156071


namespace NUMINAMATH_GPT_square_area_rational_l1560_156093

-- Define the condition: the side length of the square is a rational number.
def is_rational (x : ℚ) : Prop := true

-- Define the theorem to be proved: If the side length of a square is rational, then its area is rational.
theorem square_area_rational (s : ℚ) (h : is_rational s) : is_rational (s * s) := 
sorry

end NUMINAMATH_GPT_square_area_rational_l1560_156093


namespace NUMINAMATH_GPT_total_possible_rankings_l1560_156074

-- Define the players
inductive Player
| P | Q | R | S

-- Define the tournament results
inductive Result
| win | lose

-- Define Saturday's match outcomes
structure SaturdayOutcome :=
(P_vs_Q: Result)
(R_vs_S: Result)

-- Function to compute the number of possible tournament ranking sequences
noncomputable def countTournamentSequences : Nat :=
  let saturdayOutcomes: List SaturdayOutcome :=
    [ {P_vs_Q := Result.win, R_vs_S := Result.win}
    , {P_vs_Q := Result.win, R_vs_S := Result.lose}
    , {P_vs_Q := Result.lose, R_vs_S := Result.win}
    , {P_vs_Q := Result.lose, R_vs_S := Result.lose}
    ]
  let sundayPermutations (outcome : SaturdayOutcome) : Nat :=
    2 * 2  -- 2 permutations for 1st and 2nd places * 2 permutations for 3rd and 4th places per each outcome
  saturdayOutcomes.foldl (fun acc outcome => acc + sundayPermutations outcome) 0

-- Define the theorem to prove the total number of permutations
theorem total_possible_rankings : countTournamentSequences = 8 :=
by
  -- Proof steps here (proof omitted)
  sorry

end NUMINAMATH_GPT_total_possible_rankings_l1560_156074


namespace NUMINAMATH_GPT_total_time_to_fill_tank_with_leak_l1560_156052

theorem total_time_to_fill_tank_with_leak
  (C : ℝ) -- Capacity of the tank
  (rate1 : ℝ := C / 20) -- Rate of pipe 1 filling the tank
  (rate2 : ℝ := C / 30) -- Rate of pipe 2 filling the tank
  (combined_rate : ℝ := rate1 + rate2) -- Combined rate of both pipes
  (effective_rate : ℝ := (2 / 3) * combined_rate) -- Effective rate considering the leak
  : (C / effective_rate = 18) :=
by
  -- The proof would go here but is removed per the instructions.
  sorry

end NUMINAMATH_GPT_total_time_to_fill_tank_with_leak_l1560_156052


namespace NUMINAMATH_GPT_binomial_12_11_eq_12_l1560_156051

-- Define binomial coefficient function
def binomial (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Statement to prove
theorem binomial_12_11_eq_12 :
  binomial 12 11 = 12 := by
  sorry

end NUMINAMATH_GPT_binomial_12_11_eq_12_l1560_156051


namespace NUMINAMATH_GPT_cashier_can_satisfy_request_l1560_156054

theorem cashier_can_satisfy_request (k : ℕ) (h : k > 8) : ∃ m n : ℕ, k = 3 * m + 5 * n :=
sorry

end NUMINAMATH_GPT_cashier_can_satisfy_request_l1560_156054


namespace NUMINAMATH_GPT_solve_for_x_l1560_156063

-- We state the problem as a theorem.
theorem solve_for_x (y x : ℚ) : 
  (x - 60) / 3 = (4 - 3 * x) / 6 + y → x = (124 + 6 * y) / 5 :=
by
  -- The actual proof part is skipped with sorry.
  sorry

end NUMINAMATH_GPT_solve_for_x_l1560_156063


namespace NUMINAMATH_GPT_books_taken_out_on_monday_l1560_156011

-- Define total number of books initially
def total_books_init := 336

-- Define books taken out on Monday
variable (x : ℕ)

-- Define books brought back on Tuesday
def books_brought_back := 22

-- Define books present after Tuesday
def books_after_tuesday := 234

-- Theorem statement
theorem books_taken_out_on_monday :
  total_books_init - x + books_brought_back = books_after_tuesday → x = 124 :=
by sorry

end NUMINAMATH_GPT_books_taken_out_on_monday_l1560_156011


namespace NUMINAMATH_GPT_actual_time_between_two_and_three_l1560_156015

theorem actual_time_between_two_and_three (x y : ℕ) 
  (h1 : 2 ≤ x ∧ x < 3)
  (h2 : 60 * y + x = 60 * x + y - 55) : 
  x = 2 ∧ y = 5 + 5 / 11 := 
sorry

end NUMINAMATH_GPT_actual_time_between_two_and_three_l1560_156015


namespace NUMINAMATH_GPT_nancy_total_spending_l1560_156047

/-- A bead shop sells crystal beads at $9 each and metal beads at $10 each.
    Nancy buys one set of crystal beads and two sets of metal beads. -/
def cost_of_crystal_bead := 9
def cost_of_metal_bead := 10
def sets_of_crystal_beads_bought := 1
def sets_of_metal_beads_bought := 2

/-- Prove the total amount Nancy spends is $29 given the conditions. -/
theorem nancy_total_spending :
  sets_of_crystal_beads_bought * cost_of_crystal_bead +
  sets_of_metal_beads_bought * cost_of_metal_bead = 29 :=
by
  sorry

end NUMINAMATH_GPT_nancy_total_spending_l1560_156047


namespace NUMINAMATH_GPT_circle_equation_and_shortest_chord_l1560_156094

-- Definitions based on given conditions
def point_P : ℝ × ℝ := (4, -1)
def line_l1 (x y : ℝ) : Prop := x - 6 * y - 10 = 0
def line_l2 (x y : ℝ) : Prop := 5 * x - 3 * y = 0

-- The circle should be such that it intersects line l1 at point P and its center lies on line l2
theorem circle_equation_and_shortest_chord 
  (C : ℝ × ℝ) (r : ℝ) (hC_l2 : line_l2 C.1 C.2)
  (h_intersect : ∃ (k : ℝ), point_P.1 = (C.1 + k * (C.1 - point_P.1)) ∧ point_P.2 = (C.2 + k * (C.2 - point_P.2))) :
  -- Proving (1): Equation of the circle
  ((C.1 = 3) ∧ (C.2 = 5) ∧ r^2 = 37) ∧
  -- Proving (2): Length of the shortest chord through the origin is 2 * sqrt(3)
  (2 * Real.sqrt 3 = 2 * Real.sqrt (r^2 - ((C.1^2 + C.2^2) - (2 * C.1 * 0 + 2 * C.2 * 0)))) :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_and_shortest_chord_l1560_156094


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l1560_156014

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h2 : a 2 + a 5 + a 8 = 15) : a 3 + a 7 = 10 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l1560_156014


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l1560_156006

theorem system1_solution (x y : ℤ) : 
  (x - y = 3) ∧ (x = 3 * y - 1) → (x = 5) ∧ (y = 2) :=
by
  sorry

theorem system2_solution (x y : ℤ) : 
  (2 * x + 3 * y = -1) ∧ (3 * x - 2 * y = 18) → (x = 4) ∧ (y = -3) :=
by
  sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l1560_156006


namespace NUMINAMATH_GPT_binomial_standard_deviation_l1560_156041

noncomputable def standard_deviation_binomial (n : ℕ) (p : ℝ) : ℝ :=
  Real.sqrt (n * p * (1 - p))

theorem binomial_standard_deviation (n : ℕ) (p : ℝ) (hn : 0 ≤ n) (hp : 0 ≤ p) (hp1: p ≤ 1) :
  standard_deviation_binomial n p = Real.sqrt (n * p * (1 - p)) :=
by
  sorry

end NUMINAMATH_GPT_binomial_standard_deviation_l1560_156041


namespace NUMINAMATH_GPT_percentage_of_other_sales_l1560_156032

theorem percentage_of_other_sales :
  let pensPercentage := 20
  let pencilsPercentage := 15
  let notebooksPercentage := 30
  let totalPercentage := 100
  totalPercentage - (pensPercentage + pencilsPercentage + notebooksPercentage) = 35 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_other_sales_l1560_156032


namespace NUMINAMATH_GPT_beau_age_today_l1560_156090

theorem beau_age_today (sons_age : ℕ) (triplets : ∀ i j : ℕ, i ≠ j → sons_age = 16) 
                       (beau_age_three_years_ago : ℕ) 
                       (H : 3 * (sons_age - 3) = beau_age_three_years_ago) :
  beau_age_three_years_ago + 3 = 42 :=
by
  -- Normally this is the place to write the proof,
  -- but it's enough to outline the theorem statement as per the instructions.
  sorry

end NUMINAMATH_GPT_beau_age_today_l1560_156090


namespace NUMINAMATH_GPT_common_ratio_q_is_one_l1560_156022

-- Define the geometric sequence {a_n}, and the third term a_3 and sum of first three terms S_3
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = a n * a 1

variables {a : ℕ → ℝ}
variable (q : ℝ)

-- Given conditions
axiom a_3 : a 3 = 3 / 2
axiom S_3 : a 1 * (1 + q + q^2) = 9 / 2

-- We need to prove q = 1
theorem common_ratio_q_is_one (h1 : is_geometric_sequence a) : q = 1 := sorry

end NUMINAMATH_GPT_common_ratio_q_is_one_l1560_156022


namespace NUMINAMATH_GPT_bennett_brothers_count_l1560_156020

theorem bennett_brothers_count :
  ∃ B, B = 2 * 4 - 2 ∧ B = 6 :=
by
  sorry

end NUMINAMATH_GPT_bennett_brothers_count_l1560_156020


namespace NUMINAMATH_GPT_speed_of_boat_in_still_water_l1560_156024

-- Define a structure for the conditions
structure BoatConditions where
  V_b : ℝ    -- Speed of the boat in still water
  V_s : ℝ    -- Speed of the stream
  goes_along_stream : V_b + V_s = 11
  goes_against_stream : V_b - V_s = 5

-- Define the target theorem
theorem speed_of_boat_in_still_water (c : BoatConditions) : c.V_b = 8 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_boat_in_still_water_l1560_156024


namespace NUMINAMATH_GPT_seat_arrangement_l1560_156043

theorem seat_arrangement :
  ∃ (arrangement : Fin 7 → String), 
  (arrangement 6 = "Diane") ∧
  (∃ (i j : Fin 7), i < j ∧ arrangement i = "Carla" ∧ arrangement j = "Adam" ∧ j = (i + 1)) ∧
  (∃ (i j k : Fin 7), i < j ∧ j < k ∧ arrangement i = "Brian" ∧ arrangement j = "Ellie" ∧ (k - i) ≥ 3) ∧
  arrangement 3 = "Carla" := 
sorry

end NUMINAMATH_GPT_seat_arrangement_l1560_156043


namespace NUMINAMATH_GPT_charge_per_kilo_l1560_156016

variable (x : ℝ)

theorem charge_per_kilo (h : 5 * x + 10 * x + 20 * x = 70) : x = 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_charge_per_kilo_l1560_156016


namespace NUMINAMATH_GPT_ratio_of_students_to_dishes_l1560_156070

theorem ratio_of_students_to_dishes (m n : ℕ) 
  (h_students : n > 0)
  (h_dishes : ∃ dishes : Finset ℕ, dishes.card = 100)
  (h_each_student_tastes_10 : ∀ student : Finset ℕ, student.card = 10) 
  (h_pairs_taste_by_m_students : ∀ {d1 d2 : ℕ} (hd1 : d1 ∈ Finset.range 100) (hd2 : d2 ∈ Finset.range 100), m = 10) 
  : n / m = 110 := by
  sorry

end NUMINAMATH_GPT_ratio_of_students_to_dishes_l1560_156070


namespace NUMINAMATH_GPT_trapezoid_ratio_l1560_156086

theorem trapezoid_ratio (A B C D M N K : Type) 
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup M] [AddCommGroup N] [AddCommGroup K]
  (CM MD CN NA AD BC : ℝ)
  (h1 : CM / MD = 4 / 3)
  (h2 : CN / NA = 4 / 3) 
  : AD / BC = 7 / 12 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_ratio_l1560_156086


namespace NUMINAMATH_GPT_system_of_equations_is_B_l1560_156019

-- Define the given conditions and correct answer
def condition1 (x y : ℝ) : Prop := 5 * x + y = 3
def condition2 (x y : ℝ) : Prop := x + 5 * y = 2
def correctAnswer (x y : ℝ) : Prop := 5 * x + y = 3 ∧ x + 5 * y = 2

theorem system_of_equations_is_B (x y : ℝ) : condition1 x y ∧ condition2 x y ↔ correctAnswer x y := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_system_of_equations_is_B_l1560_156019


namespace NUMINAMATH_GPT_original_price_of_article_l1560_156035

theorem original_price_of_article (selling_price : ℝ) (loss_percent : ℝ) (P : ℝ) 
  (h1 : selling_price = 450)
  (h2 : loss_percent = 25)
  : selling_price = (1 - loss_percent / 100) * P → P = 600 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_article_l1560_156035


namespace NUMINAMATH_GPT_calculate_expression_l1560_156082

theorem calculate_expression : 
  ∀ (x y z : ℤ), x = 2 → y = -3 → z = 7 → (x^2 + y^2 + z^2 - 2 * x * y) = 74 :=
by
  intros x y z hx hy hz
  rw [hx, hy, hz]
  sorry

end NUMINAMATH_GPT_calculate_expression_l1560_156082


namespace NUMINAMATH_GPT_langsley_commute_time_l1560_156002

theorem langsley_commute_time (first_bus: ℕ) (first_wait: ℕ) (second_bus: ℕ) (second_wait: ℕ) (third_bus: ℕ) (total_time: ℕ)
  (h1: first_bus = 40)
  (h2: first_wait = 10)
  (h3: second_bus = 50)
  (h4: second_wait = 15)
  (h5: third_bus = 95)
  (h6: total_time = first_bus + first_wait + second_bus + second_wait + third_bus) :
  total_time = 210 := 
by 
  sorry

end NUMINAMATH_GPT_langsley_commute_time_l1560_156002


namespace NUMINAMATH_GPT_purchase_price_of_article_l1560_156065

theorem purchase_price_of_article (P M : ℝ) (h1 : M = 55) (h2 : M = 0.30 * P + 12) : P = 143.33 :=
  sorry

end NUMINAMATH_GPT_purchase_price_of_article_l1560_156065


namespace NUMINAMATH_GPT_quadratic_function_value_l1560_156038

theorem quadratic_function_value (a b c : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a - b + c = 9) :
  a + 3 * b + c = 1 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_function_value_l1560_156038


namespace NUMINAMATH_GPT_diff_between_roots_l1560_156061

theorem diff_between_roots (p : ℝ) (r s : ℝ)
  (h_eq : ∀ x : ℝ, x^2 - (p+1)*x + (p^2 + 2*p - 3)/4 = 0 → x = r ∨ x = s)
  (h_ge : r ≥ s) :
  r - s = Real.sqrt (2*p + 1 - p^2) := by
  sorry

end NUMINAMATH_GPT_diff_between_roots_l1560_156061


namespace NUMINAMATH_GPT_ron_total_tax_l1560_156013

def car_price : ℝ := 30000
def first_tier_level : ℝ := 10000
def first_tier_rate : ℝ := 0.25
def second_tier_rate : ℝ := 0.15

def first_tier_tax : ℝ := first_tier_level * first_tier_rate
def second_tier_tax : ℝ := (car_price - first_tier_level) * second_tier_rate
def total_tax : ℝ := first_tier_tax + second_tier_tax

theorem ron_total_tax : 
  total_tax = 5500 := by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_ron_total_tax_l1560_156013


namespace NUMINAMATH_GPT_find_min_value_l1560_156018

variable (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1)

theorem find_min_value :
  (1 / (2 * a + 3 * b) + 1 / (2 * b + 3 * c) + 1 / (2 * c + 3 * a)) ≥ (9 / 5) :=
sorry

end NUMINAMATH_GPT_find_min_value_l1560_156018


namespace NUMINAMATH_GPT_age_difference_l1560_156030

theorem age_difference (A B C : ℕ) (h1 : A + B > B + C) (h2 : C = A - 17) : (A + B) - (B + C) = 17 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l1560_156030


namespace NUMINAMATH_GPT_determine_asymptotes_l1560_156080

noncomputable def hyperbola_eccentricity_asymptote_relation (a b : ℝ) (e : ℝ) (k : ℝ) :=
  a > 0 ∧ b > 0 ∧ (e = Real.sqrt 2 * |k|) ∧ (k = b / a)

theorem determine_asymptotes (a b : ℝ) (h : hyperbola_eccentricity_asymptote_relation a b (Real.sqrt (a^2 + b^2) / a) (b / a)) :
  true := sorry

end NUMINAMATH_GPT_determine_asymptotes_l1560_156080


namespace NUMINAMATH_GPT_tickets_won_whack_a_mole_l1560_156001

variable (t : ℕ)

def tickets_from_skee_ball : ℕ := 9
def cost_per_candy : ℕ := 6
def number_of_candies : ℕ := 7
def total_tickets_needed : ℕ := cost_per_candy * number_of_candies

theorem tickets_won_whack_a_mole : t + tickets_from_skee_ball = total_tickets_needed → t = 33 :=
by
  intro h
  have h1 : total_tickets_needed = 42 := by sorry
  have h2 : tickets_from_skee_ball = 9 := by rfl
  rw [h2, h1] at h
  sorry

end NUMINAMATH_GPT_tickets_won_whack_a_mole_l1560_156001


namespace NUMINAMATH_GPT_fib_math_competition_l1560_156023

theorem fib_math_competition :
  ∃ (n9 n8 n7 : ℕ), 
    n9 * 4 = n8 * 7 ∧ 
    n9 * 3 = n7 * 10 ∧ 
    n9 + n8 + n7 = 131 :=
sorry

end NUMINAMATH_GPT_fib_math_competition_l1560_156023


namespace NUMINAMATH_GPT_largest_int_with_remainder_l1560_156089

theorem largest_int_with_remainder (k : ℤ) (h₁ : k < 95) (h₂ : k % 7 = 5) : k = 94 := by
sorry

end NUMINAMATH_GPT_largest_int_with_remainder_l1560_156089


namespace NUMINAMATH_GPT_bricklayer_wall_l1560_156099

/-- 
A bricklayer lays a certain number of meters of wall per day and works for a certain number of days.
Given the daily work rate and the number of days worked, this proof shows that the total meters of 
wall laid equals the product of the daily work rate and the number of days.
-/
theorem bricklayer_wall (daily_rate : ℕ) (days_worked : ℕ) (total_meters : ℕ) 
  (h1 : daily_rate = 8) (h2 : days_worked = 15) : total_meters = 120 :=
by {
  sorry
}

end NUMINAMATH_GPT_bricklayer_wall_l1560_156099


namespace NUMINAMATH_GPT_total_balloons_l1560_156081

-- Define the conditions
def joan_balloons : ℕ := 9
def sally_balloons : ℕ := 5
def jessica_balloons : ℕ := 2

-- The statement we want to prove
theorem total_balloons : joan_balloons + sally_balloons + jessica_balloons = 16 :=
by
  sorry

end NUMINAMATH_GPT_total_balloons_l1560_156081


namespace NUMINAMATH_GPT_sum_fifth_to_seventh_terms_arith_seq_l1560_156062

theorem sum_fifth_to_seventh_terms_arith_seq (a d : ℤ)
  (h1 : a + 7 * d = 16) (h2 : a + 8 * d = 22) (h3 : a + 9 * d = 28) :
  (a + 4 * d) + (a + 5 * d) + (a + 6 * d) = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_fifth_to_seventh_terms_arith_seq_l1560_156062


namespace NUMINAMATH_GPT_find_k_percent_l1560_156025

theorem find_k_percent (k : ℝ) : 0.2 * 30 = 6 → (k / 100) * 25 = 6 → k = 24 := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_k_percent_l1560_156025


namespace NUMINAMATH_GPT_cell_phone_height_l1560_156072

theorem cell_phone_height (width perimeter : ℕ) (h1 : width = 9) (h2 : perimeter = 46) : 
  ∃ length : ℕ, length = 14 ∧ perimeter = 2 * (width + length) :=
by
  sorry

end NUMINAMATH_GPT_cell_phone_height_l1560_156072


namespace NUMINAMATH_GPT_rectangle_ratio_of_semicircles_l1560_156066

theorem rectangle_ratio_of_semicircles (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h : a * b = π * b^2) : a / b = π := by
  sorry

end NUMINAMATH_GPT_rectangle_ratio_of_semicircles_l1560_156066


namespace NUMINAMATH_GPT_find_n_l1560_156026

theorem find_n : ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 7615 [MOD 15] ∧ n = 10 := by
  use 10
  repeat { sorry }

end NUMINAMATH_GPT_find_n_l1560_156026


namespace NUMINAMATH_GPT_smallest_norm_l1560_156044

noncomputable def vectorNorm (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem smallest_norm (v : ℝ × ℝ)
  (h : vectorNorm (v.1 + 4, v.2 + 2) = 10) :
  vectorNorm v >= 10 - 2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_smallest_norm_l1560_156044


namespace NUMINAMATH_GPT_arithmetic_seq_a10_l1560_156003

variable (a : ℕ → ℚ)
variable (S : ℕ → ℚ)
variable (d : ℚ := 1)

def is_arithmetic_seq (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = d

def sum_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem arithmetic_seq_a10 (h_seq : is_arithmetic_seq a d)
                          (h_sum : sum_first_n_terms a S)
                          (h_condition : S 8 = 4 * S 4) :
  a 10 = 19/2 := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_a10_l1560_156003


namespace NUMINAMATH_GPT_zebra_crossing_distance_l1560_156087

theorem zebra_crossing_distance
  (boulevard_width : ℝ)
  (distance_along_stripes : ℝ)
  (stripe_length : ℝ)
  (distance_between_stripes : ℝ) :
  boulevard_width = 60 →
  distance_along_stripes = 22 →
  stripe_length = 65 →
  distance_between_stripes = (60 * 22) / 65 →
  distance_between_stripes = 20.31 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_zebra_crossing_distance_l1560_156087


namespace NUMINAMATH_GPT_roots_equal_when_m_l1560_156031

noncomputable def equal_roots_condition (k n m : ℝ) : Prop :=
  1 + 4 * m^2 * k + 4 * m * n = 0

theorem roots_equal_when_m :
  equal_roots_condition 1 3 (-1.5 + Real.sqrt 2) ∧ 
  equal_roots_condition 1 3 (-1.5 - Real.sqrt 2) :=
by 
  sorry

end NUMINAMATH_GPT_roots_equal_when_m_l1560_156031


namespace NUMINAMATH_GPT_larger_number_of_hcf_lcm_l1560_156084

theorem larger_number_of_hcf_lcm (hcf : ℕ) (a b : ℕ) (f1 f2 : ℕ) 
  (hcf_condition : hcf = 20) 
  (factors_condition : f1 = 21 ∧ f2 = 23) 
  (lcm_condition : Nat.lcm a b = hcf * f1 * f2):
  max a b = 460 := 
  sorry

end NUMINAMATH_GPT_larger_number_of_hcf_lcm_l1560_156084


namespace NUMINAMATH_GPT_squares_difference_l1560_156053

theorem squares_difference (a b : ℕ) (h₁ : a = 601) (h₂ : b = 597) : a^2 - b^2 = 4792 := by
  rw [h₁, h₂]
  -- insert actual proof here
  sorry

end NUMINAMATH_GPT_squares_difference_l1560_156053


namespace NUMINAMATH_GPT_fireflies_joined_l1560_156027

theorem fireflies_joined (x : ℕ) : 
  let initial_fireflies := 3
  let flew_away := 2
  let remaining_fireflies := 9
  initial_fireflies + x - flew_away = remaining_fireflies → x = 8 := by
  sorry

end NUMINAMATH_GPT_fireflies_joined_l1560_156027


namespace NUMINAMATH_GPT_lcm_48_147_l1560_156085

theorem lcm_48_147 : Nat.lcm 48 147 = 2352 := sorry

end NUMINAMATH_GPT_lcm_48_147_l1560_156085


namespace NUMINAMATH_GPT_two_lines_parallel_same_plane_l1560_156075

-- Defining the types for lines and planes
variable (Line : Type) (Plane : Type)

-- Defining the relationships similar to the mathematical conditions
variable (parallel_to_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Prop)
variable (skew : Line → Line → Prop)

-- Defining the non-overlapping relationships between lines (assuming these relations are mutually exclusive)
axiom parallel_or_intersect_or_skew : ∀ (a b: Line), 
  (parallel a b ∨ intersect a b ∨ skew a b)

-- The statement we want to prove
theorem two_lines_parallel_same_plane (a b: Line) (α: Plane) :
  parallel_to_plane a α → parallel_to_plane b α → (parallel a b ∨ intersect a b ∨ skew a b) :=
by
  intro ha hb
  apply parallel_or_intersect_or_skew

end NUMINAMATH_GPT_two_lines_parallel_same_plane_l1560_156075


namespace NUMINAMATH_GPT_A_alone_days_l1560_156005

variable (r_A r_B r_C : ℝ)

-- Given conditions:
axiom cond1 : r_A + r_B = 1 / 3
axiom cond2 : r_B + r_C = 1 / 6
axiom cond3 : r_A + r_C = 4 / 15

-- Proposition stating the required proof, that A alone can do the job in 60/13 days:
theorem A_alone_days : r_A ≠ 0 → 1 / r_A = 60 / 13 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_A_alone_days_l1560_156005


namespace NUMINAMATH_GPT_max_price_reduction_l1560_156078

theorem max_price_reduction (CP SP : ℝ) (profit_margin : ℝ) (max_reduction : ℝ) :
  CP = 1000 ∧ SP = 1500 ∧ profit_margin = 0.05 → SP - max_reduction = CP * (1 + profit_margin) → max_reduction = 450 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_price_reduction_l1560_156078


namespace NUMINAMATH_GPT_speed_A_correct_l1560_156039

noncomputable def speed_A : ℝ :=
  200 / (19.99840012798976 * 60)

theorem speed_A_correct :
  speed_A = 0.16668 :=
sorry

end NUMINAMATH_GPT_speed_A_correct_l1560_156039


namespace NUMINAMATH_GPT_Linda_purchase_cost_l1560_156004

def price_peanuts : ℝ := sorry
def price_berries : ℝ := sorry
def price_coconut : ℝ := sorry
def price_dates : ℝ := sorry

theorem Linda_purchase_cost:
  ∃ (p b c d : ℝ), 
    (p + b + c + d = 30) ∧ 
    (3 * p = d) ∧
    ((p + b) / 2 = c) ∧
    (b + c = 65 / 9) :=
sorry

end NUMINAMATH_GPT_Linda_purchase_cost_l1560_156004


namespace NUMINAMATH_GPT_find_income_l1560_156017

-- Define the condition for savings
def savings_formula (income expenditure savings : ℝ) : Prop :=
  income - expenditure = savings

-- Define the ratio between income and expenditure
def ratio_condition (income expenditure : ℝ) : Prop :=
  income = 5 / 4 * expenditure

-- Given:
-- savings: Rs. 3400
-- We need to prove the income is Rs. 17000
theorem find_income (savings : ℝ) (income expenditure : ℝ) :
  savings_formula income expenditure savings →
  ratio_condition income expenditure →
  savings = 3400 →
  income = 17000 :=
sorry

end NUMINAMATH_GPT_find_income_l1560_156017


namespace NUMINAMATH_GPT_baba_yaga_powder_problem_l1560_156083

theorem baba_yaga_powder_problem (A B d : ℝ) 
  (h1 : A + B + d = 6) 
  (h2 : A + d = 3) 
  (h3 : B + d = 2) : 
  A = 4 ∧ B = 3 := 
sorry

end NUMINAMATH_GPT_baba_yaga_powder_problem_l1560_156083
