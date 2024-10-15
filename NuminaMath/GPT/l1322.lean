import Mathlib

namespace NUMINAMATH_GPT_probability_of_winning_reward_l1322_132239

-- Definitions representing the problem conditions
def red_envelopes : ℕ := 4
def card_types : ℕ := 3

-- Theorem statement: Prove the probability of winning the reward is 4/9
theorem probability_of_winning_reward : 
  (∃ (n m : ℕ), n = card_types^red_envelopes ∧ m = (Nat.choose red_envelopes 2) * (Nat.factorial 3)) → 
  (m / n = 4/9) :=
by
  sorry  -- Proof to be filled in

end NUMINAMATH_GPT_probability_of_winning_reward_l1322_132239


namespace NUMINAMATH_GPT_remainder_when_squared_l1322_132231

theorem remainder_when_squared (n : ℤ) (h : n % 5 = 3) : (n^2) % 5 = 4 := by
  sorry

end NUMINAMATH_GPT_remainder_when_squared_l1322_132231


namespace NUMINAMATH_GPT_johns_beef_order_l1322_132258

theorem johns_beef_order (B : ℕ)
  (h1 : 8 * B + 6 * (2 * B) = 14000) :
  B = 1000 :=
by
  sorry

end NUMINAMATH_GPT_johns_beef_order_l1322_132258


namespace NUMINAMATH_GPT_total_vegetables_l1322_132244

-- Define the initial conditions
def potatoes : Nat := 560
def cucumbers : Nat := potatoes - 132
def tomatoes : Nat := 3 * cucumbers
def peppers : Nat := tomatoes / 2
def carrots : Nat := cucumbers + tomatoes

-- State the theorem to prove the total number of vegetables
theorem total_vegetables :
  560 + (560 - 132) + (3 * (560 - 132)) + ((3 * (560 - 132)) / 2) + ((560 - 132) + (3 * (560 - 132))) = 4626 := by
  sorry

end NUMINAMATH_GPT_total_vegetables_l1322_132244


namespace NUMINAMATH_GPT_total_toys_l1322_132218

theorem total_toys (K A L : ℕ) (h1 : A = K + 30) (h2 : L = 2 * K) (h3 : K + A = 160) : 
    K + A + L = 290 :=
by
  sorry

end NUMINAMATH_GPT_total_toys_l1322_132218


namespace NUMINAMATH_GPT_vanessa_savings_weeks_l1322_132285

-- Define the conditions as constants
def dress_cost : ℕ := 120
def initial_savings : ℕ := 25
def weekly_allowance : ℕ := 30
def weekly_arcade_spending : ℕ := 15
def weekly_snack_spending : ℕ := 5

-- The theorem statement based on the problem
theorem vanessa_savings_weeks : 
  ∃ (n : ℕ), (n * (weekly_allowance - weekly_arcade_spending - weekly_snack_spending) + initial_savings) ≥ dress_cost ∧ 
             (n - 1) * (weekly_allowance - weekly_arcade_spending - weekly_snack_spending) + initial_savings < dress_cost := by
  sorry

end NUMINAMATH_GPT_vanessa_savings_weeks_l1322_132285


namespace NUMINAMATH_GPT_multiplication_problems_l1322_132229

theorem multiplication_problems :
  (30 * 30 = 900) ∧
  (30 * 40 = 1200) ∧
  (40 * 70 = 2800) ∧
  (50 * 70 = 3500) ∧
  (60 * 70 = 4200) ∧
  (4 * 90 = 360) :=
by sorry

end NUMINAMATH_GPT_multiplication_problems_l1322_132229


namespace NUMINAMATH_GPT_prime_odd_sum_l1322_132220

theorem prime_odd_sum (a b : ℕ) (h1 : Prime a) (h2 : Odd b) (h3 : a^2 + b = 2001) : a + b = 1999 :=
sorry

end NUMINAMATH_GPT_prime_odd_sum_l1322_132220


namespace NUMINAMATH_GPT_range_of_m_l1322_132245

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, abs (x - m) < 1 ↔ (1/3 < x ∧ x < 1/2)) ↔ (-1/2 ≤ m ∧ m ≤ 4/3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1322_132245


namespace NUMINAMATH_GPT_tan_half_angle_product_l1322_132262

theorem tan_half_angle_product (a b : ℝ) 
  (h : 7 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0) :
  ∃ x : ℝ, x = Real.tan (a / 2) * Real.tan (b / 2) ∧ (x = Real.sqrt (26 / 7) ∨ x = -Real.sqrt (26 / 7)) :=
by
  sorry

end NUMINAMATH_GPT_tan_half_angle_product_l1322_132262


namespace NUMINAMATH_GPT_fraction_given_to_son_l1322_132279

theorem fraction_given_to_son : 
  ∀ (blue_apples yellow_apples total_apples remaining_apples given_apples : ℕ),
    blue_apples = 5 →
    yellow_apples = 2 * blue_apples →
    total_apples = blue_apples + yellow_apples →
    remaining_apples = 12 →
    given_apples = total_apples - remaining_apples →
    (given_apples : ℚ) / total_apples = 1 / 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_fraction_given_to_son_l1322_132279


namespace NUMINAMATH_GPT_inequality_holds_l1322_132263

variable (a t1 t2 t3 t4 : ℝ)

theorem inequality_holds
  (a_pos : 0 < a)
  (h_a_le : a ≤ 7/9)
  (t1_pos : 0 < t1)
  (t2_pos : 0 < t2)
  (t3_pos : 0 < t3)
  (t4_pos : 0 < t4)
  (h_prod : t1 * t2 * t3 * t4 = a^4) :
  (1 / Real.sqrt (1 + t1) + 1 / Real.sqrt (1 + t2) + 1 / Real.sqrt (1 + t3) + 1 / Real.sqrt (1 + t4)) ≤ (4 / Real.sqrt (1 + a)) :=
by
  sorry 

end NUMINAMATH_GPT_inequality_holds_l1322_132263


namespace NUMINAMATH_GPT_percentage_speaking_both_langs_l1322_132219

def diplomats_total : ℕ := 100
def diplomats_french : ℕ := 22
def diplomats_not_russian : ℕ := 32
def diplomats_neither : ℕ := 20

theorem percentage_speaking_both_langs
  (h1 : 20% diplomats_total = diplomats_neither)
  (h2 : diplomats_total - diplomats_not_russian = 68)
  (h3 : diplomats_total ≠ 0) :
  (22 + 68 - 80) / diplomats_total * 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_percentage_speaking_both_langs_l1322_132219


namespace NUMINAMATH_GPT_rainfall_november_is_180_l1322_132293

-- Defining the conditions
def daily_rainfall_first_15_days := 4 -- inches per day
def days_in_first_period := 15
def total_days_in_november := 30
def multiplier_for_second_period := 2

-- Calculation based on the problem's conditions
def total_rainfall_november := 
  (daily_rainfall_first_15_days * days_in_first_period) + 
  (multiplier_for_second_period * daily_rainfall_first_15_days * (total_days_in_november - days_in_first_period))

-- Prove that the total rainfall in November is 180 inches
theorem rainfall_november_is_180 : total_rainfall_november = 180 :=
by
  -- Proof steps (to be filled in)
  sorry

end NUMINAMATH_GPT_rainfall_november_is_180_l1322_132293


namespace NUMINAMATH_GPT_estimated_probability_l1322_132267

noncomputable def needle_intersection_probability : ℝ := 0.4

structure NeedleExperimentData :=
(distance_between_lines : ℝ)
(length_of_needle : ℝ)
(num_trials_intersections : List (ℕ × ℕ))
(intersection_frequencies : List ℝ)

def experiment_data : NeedleExperimentData :=
{ distance_between_lines := 5,
  length_of_needle := 3,
  num_trials_intersections := [(50, 23), (100, 48), (200, 83), (500, 207), (1000, 404), (2000, 802)],
  intersection_frequencies := [0.460, 0.480, 0.415, 0.414, 0.404, 0.401] }

theorem estimated_probability (data : NeedleExperimentData) :
  ∀ P : ℝ, (∀ n m, (n, m) ∈ data.num_trials_intersections → abs (m / n - P) < 0.1) → P = needle_intersection_probability :=
by
  intro P hP
  sorry

end NUMINAMATH_GPT_estimated_probability_l1322_132267


namespace NUMINAMATH_GPT_fruit_seller_gain_l1322_132250

-- Define necessary variables
variables {C S : ℝ} (G : ℝ)

-- Given conditions
def selling_price_def (C : ℝ) : ℝ := 1.25 * C
def total_cost_price (C : ℝ) : ℝ := 150 * C
def total_selling_price (C : ℝ) : ℝ := 150 * (selling_price_def C)
def gain (C : ℝ) : ℝ := total_selling_price C - total_cost_price C

-- Statement to prove: number of apples' selling price gained by the fruit-seller is 30
theorem fruit_seller_gain : G = 30 ↔ gain C = G * (selling_price_def C) :=
by
  sorry

end NUMINAMATH_GPT_fruit_seller_gain_l1322_132250


namespace NUMINAMATH_GPT_postit_notes_area_l1322_132201

theorem postit_notes_area (length width adhesive_len : ℝ) (num_notes : ℕ)
  (h_length : length = 9.4) (h_width : width = 3.7) (h_adh_len : adhesive_len = 0.6) (h_num_notes : num_notes = 15) :
  (length + (length - adhesive_len) * (num_notes - 1)) * width = 490.62 :=
by
  rw [h_length, h_width, h_adh_len, h_num_notes]
  sorry

end NUMINAMATH_GPT_postit_notes_area_l1322_132201


namespace NUMINAMATH_GPT_smallest_multiple_of_6_and_9_l1322_132292

theorem smallest_multiple_of_6_and_9 : ∃ n : ℕ, n > 0 ∧ (n % 6 = 0) ∧ (n % 9 = 0) ∧ ∀ m : ℕ, m > 0 ∧ (m % 6 = 0) ∧ (m % 9 = 0) → n ≤ m :=
  by
    sorry

end NUMINAMATH_GPT_smallest_multiple_of_6_and_9_l1322_132292


namespace NUMINAMATH_GPT_blocks_needed_for_wall_l1322_132265

theorem blocks_needed_for_wall (length height : ℕ) (block_heights block_lengths : List ℕ)
  (staggered : Bool) (even_ends : Bool)
  (h_length : length = 120)
  (h_height : height = 8)
  (h_block_heights : block_heights = [1])
  (h_block_lengths : block_lengths = [1, 2, 3])
  (h_staggered : staggered = true)
  (h_even_ends : even_ends = true) :
  ∃ (n : ℕ), n = 404 := 
sorry

end NUMINAMATH_GPT_blocks_needed_for_wall_l1322_132265


namespace NUMINAMATH_GPT_convert_rectangular_to_spherical_l1322_132252

theorem convert_rectangular_to_spherical :
  ∀ (x y z : ℝ) (ρ θ φ : ℝ),
    (x, y, z) = (2, -2 * Real.sqrt 2, 2) →
    ρ = Real.sqrt (x^2 + y^2 + z^2) →
    z = ρ * Real.cos φ →
    x = ρ * Real.sin φ * Real.cos θ →
    y = ρ * Real.sin φ * Real.sin θ →
    0 < ρ ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi →
    (ρ, θ, φ) = (4, 2 * Real.pi - Real.arcsin (Real.sqrt 6 / 3), Real.pi / 3) :=
by
  intros x y z ρ θ φ H Hρ Hφ Hθ1 Hθ2 Hconditions
  sorry

end NUMINAMATH_GPT_convert_rectangular_to_spherical_l1322_132252


namespace NUMINAMATH_GPT_polar_to_cartesian_parabola_l1322_132249

theorem polar_to_cartesian_parabola (r θ : ℝ) (h : r = 1 / (1 - Real.sin θ)) :
  ∃ x y : ℝ, x^2 = 2 * y + 1 :=
by
  sorry

end NUMINAMATH_GPT_polar_to_cartesian_parabola_l1322_132249


namespace NUMINAMATH_GPT_tan_addition_identity_l1322_132271

theorem tan_addition_identity 
  (tan_30 : Real := Real.tan (Real.pi / 6))
  (tan_15 : Real := 2 - Real.sqrt 3) : 
  tan_15 + tan_30 + tan_15 * tan_30 = 1 := 
by
  have h1 : tan_30 = Real.sqrt 3 / 3 := sorry
  have h2 : tan_15 = 2 - Real.sqrt 3 := sorry
  sorry

end NUMINAMATH_GPT_tan_addition_identity_l1322_132271


namespace NUMINAMATH_GPT_find_value_added_l1322_132299

theorem find_value_added :
  ∀ (n x : ℤ), (2 * n + x = 8 * n - 4) → (n = 4) → (x = 20) :=
by
  intros n x h1 h2
  sorry

end NUMINAMATH_GPT_find_value_added_l1322_132299


namespace NUMINAMATH_GPT_sum_of_positive_integers_n_l1322_132273

theorem sum_of_positive_integers_n
  (n : ℕ) (h1: n > 0)
  (h2 : Nat.lcm n 100 = Nat.gcd n 100 + 300) :
  n = 350 :=
sorry

end NUMINAMATH_GPT_sum_of_positive_integers_n_l1322_132273


namespace NUMINAMATH_GPT_boundary_length_of_pattern_l1322_132297

theorem boundary_length_of_pattern (area : ℝ) (num_points : ℕ) 
(points_per_side : ℕ) : 
area = 144 → num_points = 4 → points_per_side = 4 →
∃ length : ℝ, length = 92.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_boundary_length_of_pattern_l1322_132297


namespace NUMINAMATH_GPT_pollen_allergy_expected_count_l1322_132209

theorem pollen_allergy_expected_count : 
  ∀ (sample_size : ℕ) (pollen_allergy_ratio : ℚ), 
  pollen_allergy_ratio = 1/4 ∧ sample_size = 400 → sample_size * pollen_allergy_ratio = 100 :=
  by 
    intros
    sorry

end NUMINAMATH_GPT_pollen_allergy_expected_count_l1322_132209


namespace NUMINAMATH_GPT_magic_square_d_e_sum_l1322_132270

theorem magic_square_d_e_sum 
  (S : ℕ)
  (a b c d e : ℕ)
  (h1 : S = 45 + d)
  (h2 : S = 51 + e) :
  d + e = 57 :=
by
  sorry

end NUMINAMATH_GPT_magic_square_d_e_sum_l1322_132270


namespace NUMINAMATH_GPT_sum_of_digits_nine_ab_l1322_132228

noncomputable def sum_digits_base_10 (n : ℕ) : ℕ :=
-- Function to compute the sum of digits of a number in base 10
sorry

def a : ℕ := 6 * ((10^1500 - 1) / 9)

def b : ℕ := 3 * ((10^1500 - 1) / 9)

def nine_ab : ℕ := 9 * a * b

theorem sum_of_digits_nine_ab :
  sum_digits_base_10 nine_ab = 13501 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_nine_ab_l1322_132228


namespace NUMINAMATH_GPT_geometric_sum_common_ratios_l1322_132286

theorem geometric_sum_common_ratios (k p r : ℝ) 
  (hp : p ≠ r) (h_seq : p ≠ 1 ∧ r ≠ 1 ∧ p ≠ 0 ∧ r ≠ 0) 
  (h : k * p^4 - k * r^4 = 4 * (k * p^2 - k * r^2)) : 
  p + r = 3 :=
by
  -- Details omitted as requested
  sorry

end NUMINAMATH_GPT_geometric_sum_common_ratios_l1322_132286


namespace NUMINAMATH_GPT_find_f_2_l1322_132221

variable {f : ℕ → ℤ}

-- Assume the condition given in the problem
axiom h : ∀ x : ℕ, f (x + 1) = x^2 - 1

-- Prove that f(2) = 0
theorem find_f_2 : f 2 = 0 := 
sorry

end NUMINAMATH_GPT_find_f_2_l1322_132221


namespace NUMINAMATH_GPT_area_of_triangle_with_given_medians_l1322_132222

noncomputable def area_of_triangle (m1 m2 m3 : ℝ) : ℝ :=
sorry

theorem area_of_triangle_with_given_medians :
    area_of_triangle 3 4 5 = 8 :=
sorry

end NUMINAMATH_GPT_area_of_triangle_with_given_medians_l1322_132222


namespace NUMINAMATH_GPT_odd_pos_4_digit_ints_div_5_no_digit_5_l1322_132238

open Nat

def is_valid_digit (d : Nat) : Prop :=
  d ≠ 5

def valid_odd_4_digit_ints_count : Nat :=
  let a := 8  -- First digit possibilities: {1, 2, 3, 4, 6, 7, 8, 9}
  let bc := 9  -- Second and third digit possibilities: {0, 1, 2, 3, 4, 6, 7, 8, 9}
  let d := 4  -- Fourth digit possibilities: {1, 3, 7, 9}
  a * bc * bc * d

theorem odd_pos_4_digit_ints_div_5_no_digit_5 : valid_odd_4_digit_ints_count = 2592 := by
  sorry

end NUMINAMATH_GPT_odd_pos_4_digit_ints_div_5_no_digit_5_l1322_132238


namespace NUMINAMATH_GPT_quadratic_inequality_l1322_132294

noncomputable def exists_real_roots (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (a - 1) * x1 + (2 * a - 5) = 0 ∧ x2^2 + (a - 1) * x2 + (2 * a - 5) = 0

noncomputable def valid_values (a : ℝ) : Prop :=
  a > 5 / 2 ∧ a < 10

theorem quadratic_inequality (a : ℝ) 
  (h1 : exists_real_roots a) 
  (h2 : ∀ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (a - 1) * x1 + (2 * a - 5) = 0 ∧ x2^2 + (a - 1) * x2 + (2 * a - 5) = 0 
  → (1 / x1 + 1 / x2 < -3 / 5)) :
  valid_values a :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_l1322_132294


namespace NUMINAMATH_GPT_coffee_price_increase_l1322_132204

theorem coffee_price_increase (price_first_quarter price_fourth_quarter : ℕ) 
  (h_first : price_first_quarter = 40) (h_fourth : price_fourth_quarter = 60) : 
  ((price_fourth_quarter - price_first_quarter) * 100) / price_first_quarter = 50 := 
by
  -- proof would proceed here
  sorry

end NUMINAMATH_GPT_coffee_price_increase_l1322_132204


namespace NUMINAMATH_GPT_sum_xyz_zero_l1322_132207

theorem sum_xyz_zero 
  (x y z : ℝ)
  (h1 : x + y = 2 * x + z)
  (h2 : x - 2 * y = 4 * z)
  (h3 : y = 6 * z) : 
  x + y + z = 0 := by
  sorry

end NUMINAMATH_GPT_sum_xyz_zero_l1322_132207


namespace NUMINAMATH_GPT_eggs_left_on_shelf_l1322_132288

-- Define the conditions as variables in the Lean statement
variables (x y z : ℝ)

-- Define the final theorem statement
theorem eggs_left_on_shelf (hx : 0 ≤ x) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) :
  x * (1 - y) - z = (x - y * x) - z :=
by
  sorry

end NUMINAMATH_GPT_eggs_left_on_shelf_l1322_132288


namespace NUMINAMATH_GPT_range_of_x_l1322_132226

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2
  else Real.log (-x) / Real.log (1 / 2)

theorem range_of_x (x : ℝ) : f x > f (-x) ↔ (x > 1) ∨ (-1 < x ∧ x < 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l1322_132226


namespace NUMINAMATH_GPT_sequence_periodic_l1322_132223

def last_digit (n : ℕ) : ℕ := n % 10

noncomputable def a_n (n : ℕ) : ℕ := last_digit (n^(n^n))

theorem sequence_periodic :
  ∃ period : ℕ, period = 20 ∧ ∀ n m : ℕ, n ≡ m [MOD period] → a_n n = a_n m :=
sorry

end NUMINAMATH_GPT_sequence_periodic_l1322_132223


namespace NUMINAMATH_GPT_library_wall_length_l1322_132276

theorem library_wall_length 
  (D B : ℕ) 
  (h1: D = B) 
  (desk_length bookshelf_length leftover_space : ℝ) 
  (h2: desk_length = 2) 
  (h3: bookshelf_length = 1.5) 
  (h4: leftover_space = 1) : 
  3.5 * D + leftover_space = 8 :=
by { sorry }

end NUMINAMATH_GPT_library_wall_length_l1322_132276


namespace NUMINAMATH_GPT_jackson_saving_l1322_132269

theorem jackson_saving (total_amount : ℝ) (months : ℕ) (paychecks_per_month : ℕ) (savings_per_paycheck : ℝ) :
  total_amount = 3000 → months = 15 → paychecks_per_month = 2 →
  savings_per_paycheck = total_amount / months / paychecks_per_month :=
by sorry

end NUMINAMATH_GPT_jackson_saving_l1322_132269


namespace NUMINAMATH_GPT_sticker_price_l1322_132296

theorem sticker_price (y : ℝ) (h1 : ∀ (p : ℝ), p = 0.8 * y - 60 → p ≤ y)
  (h2 : ∀ (q : ℝ), q = 0.7 * y → q ≤ y)
  (h3 : (0.8 * y - 60) + 20 = 0.7 * y) :
  y = 400 :=
by
  sorry

end NUMINAMATH_GPT_sticker_price_l1322_132296


namespace NUMINAMATH_GPT_TrigPowerEqualsOne_l1322_132212

theorem TrigPowerEqualsOne : ((Real.cos (160 * Real.pi / 180) + Real.sin (160 * Real.pi / 180) * Complex.I)^36 = 1) :=
by
  sorry

end NUMINAMATH_GPT_TrigPowerEqualsOne_l1322_132212


namespace NUMINAMATH_GPT_highlighter_total_l1322_132261

theorem highlighter_total 
  (pink_highlighters : ℕ)
  (yellow_highlighters : ℕ)
  (blue_highlighters : ℕ)
  (h_pink : pink_highlighters = 4)
  (h_yellow : yellow_highlighters = 2)
  (h_blue : blue_highlighters = 5) :
  pink_highlighters + yellow_highlighters + blue_highlighters = 11 :=
by
  sorry

end NUMINAMATH_GPT_highlighter_total_l1322_132261


namespace NUMINAMATH_GPT_geometric_series_common_ratio_l1322_132217

theorem geometric_series_common_ratio (a r : ℝ) (n : ℕ) 
(h1 : a = 7 / 3) 
(h2 : r = 49 / 21)
(h3 : r = 343 / 147):
  r = 7 / 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_common_ratio_l1322_132217


namespace NUMINAMATH_GPT_max_marks_l1322_132281

theorem max_marks (M : ℝ) :
  (0.33 * M = 125 + 73) → M = 600 := by
  intro h
  sorry

end NUMINAMATH_GPT_max_marks_l1322_132281


namespace NUMINAMATH_GPT_sqrt_abc_sum_l1322_132215

variable (a b c : ℝ)

theorem sqrt_abc_sum (h1 : b + c = 17) (h2 : c + a = 20) (h3 : a + b = 23) :
  Real.sqrt (a * b * c * (a + b + c)) = 10 * Real.sqrt 273 := by
  sorry

end NUMINAMATH_GPT_sqrt_abc_sum_l1322_132215


namespace NUMINAMATH_GPT_evaluate_expression_l1322_132242

theorem evaluate_expression :
  ∀ (a b c : ℚ),
  c = b + 1 →
  b = a + 5 →
  a = 3 →
  (a + 2 ≠ 0) →
  (b - 3 ≠ 0) →
  (c + 7 ≠ 0) →
  (a + 3) * (b + 1) * (c + 9) / ((a + 2) * (b - 3) * (c + 7)) = 2.43 := 
by
  intros a b c hc hb ha h1 h2 h3
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1322_132242


namespace NUMINAMATH_GPT_price_of_davids_toy_l1322_132298

theorem price_of_davids_toy :
  ∀ (n : ℕ) (avg_before : ℕ) (avg_after : ℕ) (total_toys_after : ℕ), 
    n = 5 →
    avg_before = 10 →
    avg_after = 11 →
    total_toys_after = 6 →
  (total_toys_after * avg_after - n * avg_before = 16) :=
by
  intros n avg_before avg_after total_toys_after h_n h_avg_before h_avg_after h_total_toys_after
  sorry

end NUMINAMATH_GPT_price_of_davids_toy_l1322_132298


namespace NUMINAMATH_GPT_boys_cannot_score_twice_as_girls_l1322_132251

theorem boys_cannot_score_twice_as_girls :
  ∀ (participants : Finset ℕ) (boys girls : ℕ) (points : ℕ → ℝ),
    participants.card = 6 →
    boys = 2 →
    girls = 4 →
    (∀ p, p ∈ participants → points p = 1 ∨ points p = 0.5 ∨ points p = 0) →
    (∀ (p q : ℕ), p ∈ participants → q ∈ participants → p ≠ q → points p + points q = 1) →
    ¬ (∃ (boys_points girls_points : ℝ), 
      (∀ b ∈ (Finset.range 2), boys_points = points b) ∧
      (∀ g ∈ (Finset.range 4), girls_points = points g) ∧
      boys_points = 2 * girls_points) :=
by
  sorry

end NUMINAMATH_GPT_boys_cannot_score_twice_as_girls_l1322_132251


namespace NUMINAMATH_GPT_percent_of_liquidX_in_solutionB_l1322_132203

theorem percent_of_liquidX_in_solutionB (P : ℝ) (h₁ : 0.8 / 100 = 0.008) 
(h₂ : 1.5 / 100 = 0.015) 
(h₃ : 300 * 0.008 = 2.4) 
(h₄ : 1000 * 0.015 = 15) 
(h₅ : 15 - 2.4 = 12.6) 
(h₆ : 12.6 / 700 = P) : 
P * 100 = 1.8 :=
by sorry

end NUMINAMATH_GPT_percent_of_liquidX_in_solutionB_l1322_132203


namespace NUMINAMATH_GPT_jensen_miles_city_l1322_132236

theorem jensen_miles_city (total_gallons : ℕ) (highway_miles : ℕ) (highway_mpg : ℕ)
  (city_mpg : ℕ) (highway_gallons : ℕ) (city_gallons : ℕ) (city_miles : ℕ) :
  total_gallons = 9 ∧ highway_miles = 210 ∧ highway_mpg = 35 ∧ city_mpg = 18 ∧
  highway_gallons = highway_miles / highway_mpg ∧
  city_gallons = total_gallons - highway_gallons ∧
  city_miles = city_gallons * city_mpg → city_miles = 54 :=
by
  sorry

end NUMINAMATH_GPT_jensen_miles_city_l1322_132236


namespace NUMINAMATH_GPT_ron_spends_on_chocolate_bars_l1322_132237

/-- Ron is hosting a camp for 15 scouts where each scout needs 2 s'mores.
    Each chocolate bar costs $1.50 and can be broken into 3 sections to make 3 s'mores.
    A discount of 15% applies if 10 or more chocolate bars are purchased.
    Calculate the total amount Ron will spend on chocolate bars after applying the discount if applicable. -/
theorem ron_spends_on_chocolate_bars :
  let cost_per_bar := 1.5
  let s'mores_per_bar := 3
  let scouts := 15
  let s'mores_per_scout := 2
  let total_s'mores := scouts * s'mores_per_scout
  let bars_needed := total_s'mores / s'mores_per_bar
  let discount := 0.15
  let total_cost := bars_needed * cost_per_bar
  let discount_amount := if bars_needed >= 10 then discount * total_cost else 0
  let final_cost := total_cost - discount_amount
  final_cost = 12.75 := by sorry

end NUMINAMATH_GPT_ron_spends_on_chocolate_bars_l1322_132237


namespace NUMINAMATH_GPT_expression_equivalence_l1322_132248

theorem expression_equivalence:
  let a := 10006 - 8008
  let b := 10000 - 8002
  a = b :=
by {
  sorry
}

end NUMINAMATH_GPT_expression_equivalence_l1322_132248


namespace NUMINAMATH_GPT_cricketer_initial_average_l1322_132295

def initial_bowling_average
  (runs_for_last_5_wickets : ℝ)
  (decreased_average : ℝ)
  (final_wickets : ℝ)
  (initial_wickets : ℝ)
  (initial_average : ℝ) : Prop :=
  (initial_average * initial_wickets + runs_for_last_5_wickets) / final_wickets =
    initial_average - decreased_average

theorem cricketer_initial_average :
  initial_bowling_average 26 0.4 85 80 12 :=
by
  unfold initial_bowling_average
  sorry

end NUMINAMATH_GPT_cricketer_initial_average_l1322_132295


namespace NUMINAMATH_GPT_geometric_sequence_product_l1322_132290

variable {a b c : ℝ}

theorem geometric_sequence_product (h : ∃ r : ℝ, r ≠ 0 ∧ -4 = c * r ∧ c = b * r ∧ b = a * r ∧ a = -1 * r) (hb : b < 0) : a * b * c = -8 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_product_l1322_132290


namespace NUMINAMATH_GPT_maintenance_check_days_l1322_132214

theorem maintenance_check_days (x : ℝ) (hx : x + 0.20 * x = 60) : x = 50 :=
by
  -- this is where the proof would go
  sorry

end NUMINAMATH_GPT_maintenance_check_days_l1322_132214


namespace NUMINAMATH_GPT_exterior_angle_regular_octagon_l1322_132289

-- Definition and proof statement
theorem exterior_angle_regular_octagon :
  let n := 8 -- The number of sides of the polygon (octagon)
  let interior_angle_sum := 180 * (n - 2) 
  let interior_angle := interior_angle_sum / n
  let exterior_angle := 180 - interior_angle
  exterior_angle = 45 :=
by
  let n := 8
  let interior_angle_sum := 180 * (n - 2) 
  let interior_angle := interior_angle_sum / n
  let exterior_angle := 180 - interior_angle
  sorry

end NUMINAMATH_GPT_exterior_angle_regular_octagon_l1322_132289


namespace NUMINAMATH_GPT_Alyssa_total_spent_l1322_132235

/-- Definition of fruit costs -/
def cost_grapes : ℝ := 12.08
def cost_cherries : ℝ := 9.85
def cost_mangoes : ℝ := 7.50
def cost_pineapple : ℝ := 4.25
def cost_starfruit : ℝ := 3.98

/-- Definition of tax and discount -/
def tax_rate : ℝ := 0.10
def discount : ℝ := 3.00

/-- Calculation of the total cost Alyssa spent after applying tax and discount -/
def total_spent : ℝ := 
  let total_cost_before_tax := cost_grapes + cost_cherries + cost_mangoes + cost_pineapple + cost_starfruit
  let tax := tax_rate * total_cost_before_tax
  let total_cost_with_tax := total_cost_before_tax + tax
  total_cost_with_tax - discount

/-- Statement that needs to be proven -/
theorem Alyssa_total_spent : total_spent = 38.43 := by 
  sorry

end NUMINAMATH_GPT_Alyssa_total_spent_l1322_132235


namespace NUMINAMATH_GPT_temperature_difference_correct_l1322_132280

def refrigerator_temp : ℝ := 3
def freezer_temp : ℝ := -10
def temperature_difference : ℝ := refrigerator_temp - freezer_temp

theorem temperature_difference_correct : temperature_difference = 13 := 
by
  sorry

end NUMINAMATH_GPT_temperature_difference_correct_l1322_132280


namespace NUMINAMATH_GPT_right_angled_triangle_l1322_132287

theorem right_angled_triangle (a b c : ℕ) (h₀ : a = 7) (h₁ : b = 9) (h₂ : c = 13) :
  a^2 + b^2 ≠ c^2 :=
by
  sorry

end NUMINAMATH_GPT_right_angled_triangle_l1322_132287


namespace NUMINAMATH_GPT_running_speed_l1322_132240

theorem running_speed (R : ℝ) (walking_speed : ℝ) (total_distance : ℝ) (total_time : ℝ) (half_distance : ℝ) (walking_time : ℝ) (running_time : ℝ)
  (h1 : walking_speed = 4)
  (h2 : total_distance = 16)
  (h3 : total_time = 3)
  (h4 : half_distance = total_distance / 2)
  (h5 : walking_time = half_distance / walking_speed)
  (h6 : running_time = half_distance / R)
  (h7 : walking_time + running_time = total_time) :
  R = 8 := 
sorry

end NUMINAMATH_GPT_running_speed_l1322_132240


namespace NUMINAMATH_GPT_minimum_value_of_z_l1322_132213

theorem minimum_value_of_z :
  ∀ (x y : ℝ), ∃ z : ℝ, z = 2*x^2 + 3*y^2 + 8*x - 6*y + 35 ∧ z ≥ 24 := by
  sorry

end NUMINAMATH_GPT_minimum_value_of_z_l1322_132213


namespace NUMINAMATH_GPT_part_a_value_range_part_b_value_product_l1322_132257

-- Define the polynomial 
def P (x y : ℤ) : ℤ := 2 * x^2 - 6 * x * y + 5 * y^2

-- Part (a)
theorem part_a_value_range :
  ∀ (x y : ℤ), (1 ≤ P x y) ∧ (P x y ≤ 100) → ∃ (a b : ℤ), 1 ≤ P a b ∧ P a b ≤ 100 := sorry

-- Part (b)
theorem part_b_value_product :
  ∀ (a b c d : ℤ),
    P a b = r → P c d = s → ∀ (r s : ℤ), (∃ (x y : ℤ), P x y = r) ∧ (∃ (z w : ℤ), P z w = s) → 
    ∃ (u v : ℤ), P u v = r * s := sorry

end NUMINAMATH_GPT_part_a_value_range_part_b_value_product_l1322_132257


namespace NUMINAMATH_GPT_cosine_of_angle_between_tangents_l1322_132243

-- Definitions based on the conditions given in a)
def circle_eq (x y : ℝ) : Prop := x^2 - 2 * x + y^2 - 2 * y + 1 = 0
def P : ℝ × ℝ := (3, 2)

-- The main theorem to be proved
theorem cosine_of_angle_between_tangents (x y : ℝ)
  (hx : circle_eq x y) : 
  cos_angle_between_tangents := 
  sorry

end NUMINAMATH_GPT_cosine_of_angle_between_tangents_l1322_132243


namespace NUMINAMATH_GPT_reflect_over_x_axis_l1322_132202

def coords (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

theorem reflect_over_x_axis :
  coords (-6, -9) = (-6, 9) :=
by
  sorry

end NUMINAMATH_GPT_reflect_over_x_axis_l1322_132202


namespace NUMINAMATH_GPT_fraction_zero_implies_x_is_two_l1322_132291

theorem fraction_zero_implies_x_is_two {x : ℝ} (hfrac : (2 - |x|) / (x + 2) = 0) (hdenom : x ≠ -2) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_zero_implies_x_is_two_l1322_132291


namespace NUMINAMATH_GPT_chess_club_boys_l1322_132230

theorem chess_club_boys (G B : ℕ) 
  (h1 : G + B = 30)
  (h2 : (2 / 3) * G + (3 / 4) * B = 18) : B = 24 :=
by
  sorry

end NUMINAMATH_GPT_chess_club_boys_l1322_132230


namespace NUMINAMATH_GPT_students_in_line_l1322_132278

theorem students_in_line (between : ℕ) (Yoojung Eunji : ℕ) (h1 : Yoojung = 1) (h2 : Eunji = 1) : 
  between + Yoojung + Eunji = 16 :=
  sorry

end NUMINAMATH_GPT_students_in_line_l1322_132278


namespace NUMINAMATH_GPT_henry_twice_jill_l1322_132225

-- Conditions
def Henry := 29
def Jill := 19
def sum_ages : Nat := Henry + Jill

-- Prove the statement
theorem henry_twice_jill (Y : Nat) (H J : Nat) (h_sum : H + J = 48) (h_H : H = 29) (h_J : J = 19) :
  H - Y = 2 * (J - Y) ↔ Y = 9 :=
by {
  -- Here, we would provide the proof, but we'll skip that with sorry.
  sorry
}

end NUMINAMATH_GPT_henry_twice_jill_l1322_132225


namespace NUMINAMATH_GPT_range_of_a_l1322_132275

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ≥ 1 → (x^2 + a * x + 9) ≥ 0) : a ≥ -6 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1322_132275


namespace NUMINAMATH_GPT_sequence_first_number_l1322_132283

theorem sequence_first_number (a: ℕ → ℕ) (h1: a 7 = 14) (h2: a 8 = 19) (h3: a 9 = 33) :
  (∀ n, n ≥ 2 → a (n+1) = a n + a (n-1)) → a 1 = 30 :=
by
  sorry

end NUMINAMATH_GPT_sequence_first_number_l1322_132283


namespace NUMINAMATH_GPT_height_ratio_l1322_132232

theorem height_ratio (C : ℝ) (h_o : ℝ) (V_s : ℝ) (h_s : ℝ) (r : ℝ) :
  C = 18 * π →
  h_o = 20 →
  V_s = 270 * π →
  C = 2 * π * r →
  V_s = 1 / 3 * π * r^2 * h_s →
  h_s / h_o = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_height_ratio_l1322_132232


namespace NUMINAMATH_GPT_solve_for_x_l1322_132255

namespace RationalOps

-- Define the custom operation ※ on rational numbers
def star (a b : ℚ) : ℚ := a + b

-- Define the equation involving the custom operation
def equation (x : ℚ) : Prop := star 4 (star x 3) = 1

-- State the theorem to prove the solution
theorem solve_for_x : ∃ x : ℚ, equation x ∧ x = -6 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1322_132255


namespace NUMINAMATH_GPT_count_integers_expression_negative_l1322_132241

theorem count_integers_expression_negative :
  ∃ n : ℕ, n = 4 ∧ 
  ∀ x : ℤ, x^4 - 60 * x^2 + 144 < 0 → n = 4 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_count_integers_expression_negative_l1322_132241


namespace NUMINAMATH_GPT_sum_first_n_terms_arithmetic_sequence_l1322_132208

/-- Define the arithmetic sequence with common difference d and a given term a₄. -/
def arithmetic_sequence (n : ℕ) (a₁ d : ℤ) : ℤ :=
  a₁ + (n - 1) * d

/-- Define the sum of the first n terms of an arithmetic sequence. -/
def sum_of_arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ :=
  (n : ℤ) * ((2 * a₁ + (n - 1) * d) / 2)

theorem sum_first_n_terms_arithmetic_sequence :
  ∀ n : ℕ, 
  ∀ a₁ : ℤ, 
  (∀ d, d = 2 → (∀ a₁, (a₁ + 3 * d = 8) → sum_of_arithmetic_sequence a₁ d n = (n : ℤ) * ((n : ℤ) + 1))) :=
by
  intros n a₁ d hd h₁
  sorry

end NUMINAMATH_GPT_sum_first_n_terms_arithmetic_sequence_l1322_132208


namespace NUMINAMATH_GPT_correct_operation_result_l1322_132268

variable (x : ℕ)

theorem correct_operation_result 
  (h : x / 15 = 6) : 15 * x = 1350 :=
sorry

end NUMINAMATH_GPT_correct_operation_result_l1322_132268


namespace NUMINAMATH_GPT_circle_radius_l1322_132253

theorem circle_radius (r : ℝ) (x y : ℝ) :
  x = π * r^2 ∧ y = 2 * π * r ∧ x + y = 100 * π → r = 10 := 
  by
  sorry

end NUMINAMATH_GPT_circle_radius_l1322_132253


namespace NUMINAMATH_GPT_parabola_vertex_y_coordinate_l1322_132256

theorem parabola_vertex_y_coordinate (x y : ℝ) :
  y = 5 * x^2 + 20 * x + 45 ∧ (∃ h k, y = 5 * (x + h)^2 + k ∧ k = 25) :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_y_coordinate_l1322_132256


namespace NUMINAMATH_GPT_product_xyz_42_l1322_132234

theorem product_xyz_42 (x y z : ℝ) 
  (h1 : (x - 2)^2 + (y - 3)^2 + (z - 4)^2 = 9)
  (h2 : x + y + z = 12) : x * y * z = 42 :=
by
  sorry

end NUMINAMATH_GPT_product_xyz_42_l1322_132234


namespace NUMINAMATH_GPT_remaining_trees_correct_l1322_132205

def initial_oak_trees := 57
def initial_maple_trees := 43

def full_cut_oak := 13
def full_cut_maple := 8

def partial_cut_oak := 2.5
def partial_cut_maple := 1.5

def remaining_oak_trees := initial_oak_trees - full_cut_oak
def remaining_maple_trees := initial_maple_trees - full_cut_maple

def total_remaining_trees := remaining_oak_trees + remaining_maple_trees

theorem remaining_trees_correct : remaining_oak_trees = 44 ∧ remaining_maple_trees = 35 ∧ total_remaining_trees = 79 :=
by
  sorry

end NUMINAMATH_GPT_remaining_trees_correct_l1322_132205


namespace NUMINAMATH_GPT_points_on_line_relationship_l1322_132266

theorem points_on_line_relationship :
  let m := 2 * Real.sqrt 2 + 1
  let n := 4
  m < n :=
by
  sorry

end NUMINAMATH_GPT_points_on_line_relationship_l1322_132266


namespace NUMINAMATH_GPT_box_volume_increase_l1322_132247

theorem box_volume_increase (l w h : ℝ) 
  (h1 : l * w * h = 5000)
  (h2 : l * w + w * h + h * l = 900)
  (h3 : l + w + h = 60) : 
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
by 
  sorry

end NUMINAMATH_GPT_box_volume_increase_l1322_132247


namespace NUMINAMATH_GPT_digit_difference_l1322_132200

theorem digit_difference (X Y : ℕ) (h_digits : 0 ≤ X ∧ X < 10 ∧ 0 ≤ Y ∧ Y < 10) (h_diff :  (10 * X + Y) - (10 * Y + X) = 45) : X - Y = 5 :=
sorry

end NUMINAMATH_GPT_digit_difference_l1322_132200


namespace NUMINAMATH_GPT_sqrt_domain_l1322_132259

theorem sqrt_domain (x : ℝ) : 1 - x ≥ 0 → x ≤ 1 := by
  sorry

end NUMINAMATH_GPT_sqrt_domain_l1322_132259


namespace NUMINAMATH_GPT_average_sales_l1322_132211

theorem average_sales (jan feb mar apr : ℝ) (h_jan : jan = 100) (h_feb : feb = 60) (h_mar : mar = 40) (h_apr : apr = 120) : 
  (jan + feb + mar + apr) / 4 = 80 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_sales_l1322_132211


namespace NUMINAMATH_GPT_inequality_for_real_numbers_l1322_132254

theorem inequality_for_real_numbers (x y z : ℝ) : 
  - (3 / 2) * (x^2 + y^2 + 2 * z^2) ≤ 3 * x * y + y * z + z * x ∧ 
  3 * x * y + y * z + z * x ≤ (3 + Real.sqrt 13) / 4 * (x^2 + y^2 + 2 * z^2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_for_real_numbers_l1322_132254


namespace NUMINAMATH_GPT_number_of_flute_players_l1322_132224

theorem number_of_flute_players (F T B D C H : ℕ)
  (hT : T = 3 * F)
  (hB : B = T - 8)
  (hD : D = B + 11)
  (hC : C = 2 * F)
  (hH : H = B + 3)
  (h_total : F + T + B + D + C + H = 65) :
  F = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_flute_players_l1322_132224


namespace NUMINAMATH_GPT_value_of_expression_l1322_132284

theorem value_of_expression : 10^2 + 10 + 1 = 111 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1322_132284


namespace NUMINAMATH_GPT_order_of_abc_l1322_132246

section
variables {a b c : ℝ}

def a_def : a = (1/2) * Real.log 2 := by sorry
def b_def : b = (1/4) * Real.log 16 := by sorry
def c_def : c = (1/6) * Real.log 27 := by sorry

theorem order_of_abc : a < c ∧ c < b :=
by
  have ha : a = (1/2) * Real.log 2 := by sorry
  have hb : b = (1/2) * Real.log 4 := by sorry
  have hc : c = (1/2) * Real.log 3 := by sorry
  sorry
end

end NUMINAMATH_GPT_order_of_abc_l1322_132246


namespace NUMINAMATH_GPT_unique_solution_eqn_l1322_132282

theorem unique_solution_eqn (a : ℝ) :
  (∃! x : ℝ, 3^(x^2 + 6 * a * x + 9 * a^2) = a * x^2 + 6 * a^2 * x + 9 * a^3 + a^2 - 4 * a + 4) ↔ (a = 1) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_eqn_l1322_132282


namespace NUMINAMATH_GPT_urn_problem_l1322_132274

theorem urn_problem : 
  (5 / 12 * 20 / (20 + M) + 7 / 12 * M / (20 + M) = 0.62) → M = 111 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_urn_problem_l1322_132274


namespace NUMINAMATH_GPT_tennis_racket_weight_l1322_132216

theorem tennis_racket_weight 
  (r b : ℝ)
  (h1 : 10 * r = 8 * b)
  (h2 : 4 * b = 120) :
  r = 24 :=
by
  sorry

end NUMINAMATH_GPT_tennis_racket_weight_l1322_132216


namespace NUMINAMATH_GPT_slope_of_parallel_line_l1322_132260

theorem slope_of_parallel_line (a b c : ℝ) (x y : ℝ) (h : 3 * x + 6 * y = -12):
  (∀ m : ℝ, (∀ (x y : ℝ), (3 * x + 6 * y = -12) → y = m * x + (-(12 / 6) / 6)) → m = -1/2) :=
sorry

end NUMINAMATH_GPT_slope_of_parallel_line_l1322_132260


namespace NUMINAMATH_GPT_rigged_coin_probability_l1322_132206

theorem rigged_coin_probability (p : ℝ) (h1 : p < 1 / 2) (h2 : 20 * (p ^ 3) * ((1 - p) ^ 3) = 1 / 12) :
  p = (1 - Real.sqrt 0.86) / 2 :=
by
  sorry

end NUMINAMATH_GPT_rigged_coin_probability_l1322_132206


namespace NUMINAMATH_GPT_sum_of_cubes_l1322_132210

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l1322_132210


namespace NUMINAMATH_GPT_three_digit_solutions_l1322_132233

def three_digit_number (n a x y z : ℕ) : Prop :=
  n = 100 * x + 10 * y + z ∧
  1 ≤ x ∧ x < 10 ∧ 
  0 ≤ y ∧ y < 10 ∧ 
  0 ≤ z ∧ z < 10 ∧ 
  n + (x + y + z) = 111 * a

theorem three_digit_solutions (n : ℕ) (a x y z : ℕ) :
  three_digit_number n a x y z ↔ 
  n = 105 ∨ n = 324 ∨ n = 429 ∨ n = 543 ∨ 
  n = 648 ∨ n = 762 ∨ n = 867 ∨ n = 981 :=
sorry

end NUMINAMATH_GPT_three_digit_solutions_l1322_132233


namespace NUMINAMATH_GPT_abs_neg_five_l1322_132264

theorem abs_neg_five : abs (-5) = 5 := 
by 
  sorry

end NUMINAMATH_GPT_abs_neg_five_l1322_132264


namespace NUMINAMATH_GPT_expand_binomial_l1322_132272

theorem expand_binomial (x : ℝ) : (x + 3) * (x + 8) = x^2 + 11 * x + 24 :=
by sorry

end NUMINAMATH_GPT_expand_binomial_l1322_132272


namespace NUMINAMATH_GPT_part1_part2_l1322_132227

def P : Set ℝ := {x | 4 / (x + 2) ≥ 1}
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

theorem part1 (h : m = 2) : P ∪ S m = {x | -2 < x ∧ x ≤ 3} :=
  by sorry

theorem part2 (h : ∀ x, x ∈ S m → x ∈ P) : 0 ≤ m ∧ m ≤ 1 :=
  by sorry

end NUMINAMATH_GPT_part1_part2_l1322_132227


namespace NUMINAMATH_GPT_geometric_sum_3030_l1322_132277

theorem geometric_sum_3030 {a r : ℝ}
  (h1 : a * (1 - r ^ 1010) / (1 - r) = 300)
  (h2 : a * (1 - r ^ 2020) / (1 - r) = 540) :
  a * (1 - r ^ 3030) / (1 - r) = 732 :=
sorry

end NUMINAMATH_GPT_geometric_sum_3030_l1322_132277
