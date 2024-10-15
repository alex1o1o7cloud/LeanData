import Mathlib

namespace NUMINAMATH_GPT_number_of_pencils_l1126_112648

-- Definitions based on the conditions
def ratio_pens_pencils (P L : ℕ) : Prop := P * 6 = 5 * L
def pencils_more_than_pens (P L : ℕ) : Prop := L = P + 4

-- Statement to prove the number of pencils
theorem number_of_pencils : ∃ L : ℕ, (∃ P : ℕ, ratio_pens_pencils P L ∧ pencils_more_than_pens P L) ∧ L = 24 :=
by
  sorry

end NUMINAMATH_GPT_number_of_pencils_l1126_112648


namespace NUMINAMATH_GPT_polynomial_identity_l1126_112680

theorem polynomial_identity (a_0 a_1 a_2 a_3 a_4 : ℝ) (x : ℝ) 
  (h : (2 * x + 1)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) : 
  a_0 - a_1 + a_2 - a_3 + a_4 = 1 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_identity_l1126_112680


namespace NUMINAMATH_GPT_zeke_estimate_smaller_l1126_112691

variable (x y k : ℝ)
variable (hx_pos : 0 < x)
variable (hy_pos : 0 < y)
variable (h_inequality : x > 2 * y)
variable (hk_pos : 0 < k)

theorem zeke_estimate_smaller : (x + k) - 2 * (y + k) < x - 2 * y :=
by
  sorry

end NUMINAMATH_GPT_zeke_estimate_smaller_l1126_112691


namespace NUMINAMATH_GPT_number_of_happy_configurations_is_odd_l1126_112682

def S (m n : ℕ) := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 2 * m ∧ 1 ≤ p.2 ∧ p.2 ≤ 2 * n}

def happy_configurations (m n : ℕ) : ℕ := 
  sorry -- definition of the number of happy configurations is abstracted for this statement.

theorem number_of_happy_configurations_is_odd (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  happy_configurations m n % 2 = 1 := 
sorry

end NUMINAMATH_GPT_number_of_happy_configurations_is_odd_l1126_112682


namespace NUMINAMATH_GPT_average_hidden_primes_l1126_112652

theorem average_hidden_primes
  (visible_card1 visible_card2 visible_card3 : ℕ)
  (hidden_card1 hidden_card2 hidden_card3 : ℕ)
  (h1 : visible_card1 = 68)
  (h2 : visible_card2 = 39)
  (h3 : visible_card3 = 57)
  (prime1 : Nat.Prime hidden_card1)
  (prime2 : Nat.Prime hidden_card2)
  (prime3 : Nat.Prime hidden_card3)
  (common_sum : ℕ)
  (h4 : visible_card1 + hidden_card1 = common_sum)
  (h5 : visible_card2 + hidden_card2 = common_sum)
  (h6 : visible_card3 + hidden_card3 = common_sum) :
  (hidden_card1 + hidden_card2 + hidden_card3) / 3 = 15 + 1/3 :=
sorry

end NUMINAMATH_GPT_average_hidden_primes_l1126_112652


namespace NUMINAMATH_GPT_range_a_l1126_112646

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (x - 2))

def domain_A : Set ℝ := { x | x < -1 ∨ x > 2 }

def solution_set_B (a : ℝ) : Set ℝ := { x | x < a ∨ x > a + 1 }

theorem range_a (a : ℝ)
  (h : (domain_A ∪ solution_set_B a) = solution_set_B a) :
  -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_a_l1126_112646


namespace NUMINAMATH_GPT_r_expansion_l1126_112688

theorem r_expansion (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by
  sorry

end NUMINAMATH_GPT_r_expansion_l1126_112688


namespace NUMINAMATH_GPT_total_dots_not_visible_l1126_112659

noncomputable def total_dots_on_die : ℕ := 1 + 2 + 3 + 4 + 5 + 6
noncomputable def total_dice : ℕ := 3
noncomputable def total_visible_faces : ℕ := 5

def visible_faces : List ℕ := [1, 2, 3, 3, 4]

theorem total_dots_not_visible :
  (total_dots_on_die * total_dice) - (visible_faces.sum) = 50 := by
  sorry

end NUMINAMATH_GPT_total_dots_not_visible_l1126_112659


namespace NUMINAMATH_GPT_sum_of_integers_eq_17_l1126_112605

theorem sum_of_integers_eq_17 (a b : ℕ) (h1 : a * b + a + b = 87) 
  (h2 : Nat.gcd a b = 1) (h3 : a < 15) (h4 : b < 15) (h5 : Even a ∨ Even b) :
  a + b = 17 := 
sorry

end NUMINAMATH_GPT_sum_of_integers_eq_17_l1126_112605


namespace NUMINAMATH_GPT_ramu_profit_percent_l1126_112615

noncomputable def profit_percent (purchase_price repair_cost selling_price : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit * 100) / total_cost

theorem ramu_profit_percent :
  profit_percent 42000 13000 64900 = 18 := by
  sorry

end NUMINAMATH_GPT_ramu_profit_percent_l1126_112615


namespace NUMINAMATH_GPT_t_range_inequality_l1126_112656

theorem t_range_inequality (t : ℝ) :
  (1/8) * (2 * t - t^2) ≤ -1/4 ∧ 3 - t^2 ≥ 2 ↔ -1 ≤ t ∧ t ≤ 1 - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_t_range_inequality_l1126_112656


namespace NUMINAMATH_GPT_solve_inequality_l1126_112640

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_add (m n : ℝ) : f (m + n) = f m * f n
axiom f_pos (x : ℝ) : 0 < x → 0 < f x ∧ f x < 1

theorem solve_inequality (x : ℝ) : f (x^2) * f (2 * x - 3) > 1 ↔ -3 < x ∧ x < 1 := sorry

end NUMINAMATH_GPT_solve_inequality_l1126_112640


namespace NUMINAMATH_GPT_original_length_of_wood_l1126_112675

theorem original_length_of_wood (s cl ol : ℝ) (h1 : s = 2.3) (h2 : cl = 6.6) (h3 : ol = cl + s) : 
  ol = 8.9 := 
by 
  sorry

end NUMINAMATH_GPT_original_length_of_wood_l1126_112675


namespace NUMINAMATH_GPT_projectile_time_to_meet_l1126_112617

theorem projectile_time_to_meet
  (d v1 v2 : ℝ)
  (hd : d = 1455)
  (hv1 : v1 = 470)
  (hv2 : v2 = 500) :
  (d / (v1 + v2)) * 60 = 90 := by
  sorry

end NUMINAMATH_GPT_projectile_time_to_meet_l1126_112617


namespace NUMINAMATH_GPT_quotient_zero_l1126_112603

theorem quotient_zero (D d R Q : ℕ) (hD : D = 12) (hd : d = 17) (hR : R = 8) (h : D = d * Q + R) : Q = 0 :=
by
  sorry

end NUMINAMATH_GPT_quotient_zero_l1126_112603


namespace NUMINAMATH_GPT_ram_shyam_weight_ratio_l1126_112667

theorem ram_shyam_weight_ratio
    (R S : ℝ)
    (h1 : 1.10 * R + 1.22 * S = 82.8)
    (h2 : R + S = 72) :
    R / S = 7 / 5 :=
by sorry

end NUMINAMATH_GPT_ram_shyam_weight_ratio_l1126_112667


namespace NUMINAMATH_GPT_unique_a_values_l1126_112668

theorem unique_a_values :
  ∃ a_values : Finset ℝ,
    (∀ a ∈ a_values, ∃ r s : ℤ, (r + s = -a) ∧ (r * s = 8 * a)) ∧ a_values.card = 4 :=
by
  sorry

end NUMINAMATH_GPT_unique_a_values_l1126_112668


namespace NUMINAMATH_GPT_unique_8_tuple_real_l1126_112621

theorem unique_8_tuple_real (x : Fin 8 → ℝ) :
  (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 + x 7^2 = 1 / 8 →
  ∃! (y : Fin 8 → ℝ), (1 - y 0)^2 + (y 0 - y 1)^2 + (y 1 - y 2)^2 + (y 2 - y 3)^2 + (y 3 - y 4)^2 + (y 4 - y 5)^2 + (y 5 - y 6)^2 + (y 6 - y 7)^2 + y 7^2 = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_unique_8_tuple_real_l1126_112621


namespace NUMINAMATH_GPT_not_sunny_prob_l1126_112683

theorem not_sunny_prob (P_sunny : ℚ) (h : P_sunny = 5/7) : 1 - P_sunny = 2/7 :=
by sorry

end NUMINAMATH_GPT_not_sunny_prob_l1126_112683


namespace NUMINAMATH_GPT_gabriel_pages_correct_l1126_112616

-- Given conditions
def beatrix_pages : ℕ := 704

def cristobal_pages (b : ℕ) : ℕ := 3 * b + 15

def gabriel_pages (c b : ℕ) : ℕ := 3 * (c + b)

-- Problem statement
theorem gabriel_pages_correct : gabriel_pages (cristobal_pages beatrix_pages) beatrix_pages = 8493 :=
by 
  sorry

end NUMINAMATH_GPT_gabriel_pages_correct_l1126_112616


namespace NUMINAMATH_GPT_find_three_digit_number_in_decimal_l1126_112634

theorem find_three_digit_number_in_decimal :
  ∃ (A B C : ℕ), ∀ (hA : A ≠ 0 ∧ A < 7) (hB : B ≠ 0 ∧ B < 7) (hC : C ≠ 0 ∧ C < 7) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
    (h1 : (7 * A + B) + C = 7 * C)
    (h2 : (7 * A + B) + (7 * B + A) = 7 * B + 6), 
    A * 100 + B * 10 + C = 425 :=
by
  sorry

end NUMINAMATH_GPT_find_three_digit_number_in_decimal_l1126_112634


namespace NUMINAMATH_GPT_Gwen_walking_and_elevation_gain_l1126_112609

theorem Gwen_walking_and_elevation_gain :
  ∀ (jogging_time walking_time total_time elevation_gain : ℕ)
    (jogging_feet total_feet : ℤ),
    jogging_time = 15 ∧ jogging_feet = 500 ∧ (jogging_time + walking_time = total_time) ∧
    (5 * walking_time = 3 * jogging_time) ∧ (total_time * jogging_feet = 15 * total_feet)
    → walking_time = 9 ∧ total_feet = 800 := by 
  sorry

end NUMINAMATH_GPT_Gwen_walking_and_elevation_gain_l1126_112609


namespace NUMINAMATH_GPT_value_of_a_minus_b_l1126_112697

theorem value_of_a_minus_b (a b : ℝ) (h1 : 2 * a + b = 7) (h2 : 2 * a - b = 1) : a - b = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l1126_112697


namespace NUMINAMATH_GPT_floor_of_sum_eq_l1126_112623

theorem floor_of_sum_eq (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0)
  (hxy : x^2 + y^2 = 2500) (hzw : z^2 + w^2 = 2500) (hxz : x * z = 1200) (hyw : y * w = 1200) :
  ⌊x + y + z + w⌋ = 140 := by
  sorry

end NUMINAMATH_GPT_floor_of_sum_eq_l1126_112623


namespace NUMINAMATH_GPT_area_of_sector_l1126_112601

theorem area_of_sector {R θ: ℝ} (hR: R = 2) (hθ: θ = (2 * Real.pi) / 3) :
  (1 / 2) * R^2 * θ = (4 / 3) * Real.pi :=
by
  simp [hR, hθ]
  norm_num
  linarith

end NUMINAMATH_GPT_area_of_sector_l1126_112601


namespace NUMINAMATH_GPT_michael_lap_time_l1126_112676

theorem michael_lap_time :
  ∃ T : ℝ, (∀ D : ℝ, D = 45 → (9 * T = 10 * D) → T = 50) :=
by
  sorry

end NUMINAMATH_GPT_michael_lap_time_l1126_112676


namespace NUMINAMATH_GPT_A_more_likely_than_B_l1126_112699

-- Define the conditions
variables (n : ℕ) (k : ℕ)
-- n is the total number of programs, k is the chosen number of programs
def total_programs : ℕ := 10
def selected_programs : ℕ := 3
-- Probability of person B correctly completing each program
def probability_B_correct : ℚ := 3/5
-- Person A can correctly complete 6 out of 10 programs
def person_A_correct : ℕ := 6

-- The probability of person B successfully completing the challenge
def probability_B_success : ℚ := (3 * (9/25) * (2/5)) + (27/125)

-- Define binomial coefficient function for easier combination calculations
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The probabilities for the number of correct programs for person A
def P_X_0 : ℚ := (choose 4 3 : ℕ) / (choose 10 3 : ℕ)
def P_X_1 : ℚ := (choose 6 1 * choose 4 2 : ℕ) / (choose 10 3 : ℕ)
def P_X_2 : ℚ := (choose 6 2 * choose 4 1 : ℕ) / (choose 10 3 : ℕ)
def P_X_3 : ℚ := (choose 6 3 : ℕ) / (choose 10 3 : ℕ)

-- The distribution and expectation of X for person A
def E_X : ℚ := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 + 3 * P_X_3

-- The probability of person A successfully completing the challenge
def P_A_success : ℚ := P_X_2 + P_X_3

-- Final comparisons to determine who is more likely to succeed
def compare_success : Prop := P_A_success > probability_B_success

-- Lean statement
theorem A_more_likely_than_B : compare_success := by
  sorry

end NUMINAMATH_GPT_A_more_likely_than_B_l1126_112699


namespace NUMINAMATH_GPT_remainder_when_divided_by_8_l1126_112671

theorem remainder_when_divided_by_8 (x k : ℤ) (h : x = 63 * k + 27) : x % 8 = 3 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_8_l1126_112671


namespace NUMINAMATH_GPT_number_of_snakes_l1126_112673

-- Define the variables
variable (S : ℕ) -- Number of snakes

-- Define the cost constants
def cost_per_gecko := 15
def cost_per_iguana := 5
def cost_per_snake := 10

-- Define the number of each pet
def num_geckos := 3
def num_iguanas := 2

-- Define the yearly cost
def yearly_cost := 1140

-- Calculate the total monthly cost
def monthly_cost := num_geckos * cost_per_gecko + num_iguanas * cost_per_iguana + S * cost_per_snake

-- Calculate the total yearly cost
def total_yearly_cost := 12 * monthly_cost

-- Prove the number of snakes
theorem number_of_snakes : total_yearly_cost = yearly_cost → S = 4 := by
  sorry

end NUMINAMATH_GPT_number_of_snakes_l1126_112673


namespace NUMINAMATH_GPT_smallest_n_l1126_112681

noncomputable def smallest_positive_integer (x y : ℤ) (h1 : (x + 1) % 7 = 0) (h2 : (y - 5) % 7 = 0) : ℕ :=
  if 3 % 7 = 0 then 7 else 7

theorem smallest_n (x y : ℤ) (h1 : (x + 1) % 7 = 0) (h2 : (y - 5) % 7 = 0) : smallest_positive_integer x y h1 h2 = 7 := 
  by
  admit

end NUMINAMATH_GPT_smallest_n_l1126_112681


namespace NUMINAMATH_GPT_max_rectangle_area_l1126_112661

theorem max_rectangle_area (l w : ℕ) (h1 : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
by
  have h2 : l + w = 20 := by linarith
  -- Further steps would go here but we're just stating it
  sorry

end NUMINAMATH_GPT_max_rectangle_area_l1126_112661


namespace NUMINAMATH_GPT_jackson_hermit_crabs_l1126_112658

theorem jackson_hermit_crabs (H : ℕ) (total_souvenirs : ℕ) 
  (h1 : total_souvenirs = H + 3 * H + 6 * H) 
  (h2 : total_souvenirs = 450) : H = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_jackson_hermit_crabs_l1126_112658


namespace NUMINAMATH_GPT_smallest_nat_mul_47_last_four_digits_l1126_112608

theorem smallest_nat_mul_47_last_four_digits (N : ℕ) :
  (47 * N) % 10000 = 1969 ↔ N = 8127 :=
sorry

end NUMINAMATH_GPT_smallest_nat_mul_47_last_four_digits_l1126_112608


namespace NUMINAMATH_GPT_donovan_lap_time_is_45_l1126_112638

-- Definitions based on the conditions
def circular_track_length : ℕ := 600
def michael_lap_time : ℕ := 40
def michael_laps_to_pass_donovan : ℕ := 9

-- The theorem to prove
theorem donovan_lap_time_is_45 : ∃ D : ℕ, 8 * D = michael_laps_to_pass_donovan * michael_lap_time ∧ D = 45 := by
  sorry

end NUMINAMATH_GPT_donovan_lap_time_is_45_l1126_112638


namespace NUMINAMATH_GPT_max_value_of_g_on_interval_l1126_112685

noncomputable def g (x : ℝ) : ℝ := 5 * x^2 - 2 * x^4

theorem max_value_of_g_on_interval : ∃ x : ℝ, (0 ≤ x ∧ x ≤ Real.sqrt 2) ∧ (∀ y : ℝ, (0 ≤ y ∧ y ≤ Real.sqrt 2) → g y ≤ g x) ∧ g x = 25 / 8 := by
  sorry

end NUMINAMATH_GPT_max_value_of_g_on_interval_l1126_112685


namespace NUMINAMATH_GPT_subset_singleton_zero_l1126_112647

def X : Set ℤ := {x | -2 ≤ x ∧ x ≤ 2}

theorem subset_singleton_zero : {0} ⊆ X :=
by
  sorry

end NUMINAMATH_GPT_subset_singleton_zero_l1126_112647


namespace NUMINAMATH_GPT_ratio_w_y_l1126_112657

theorem ratio_w_y (w x y z : ℝ) 
  (h1 : w / x = 5 / 4) 
  (h2 : y / z = 3 / 2) 
  (h3 : z / x = 1 / 4) 
  (h4 : w + x + y + z = 60) : 
  w / y = 10 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_w_y_l1126_112657


namespace NUMINAMATH_GPT_spoiled_apples_l1126_112644

theorem spoiled_apples (S G : ℕ) (h1 : S + G = 8) (h2 : (G * (G - 1)) / 2 = 21) : S = 1 :=
by
  sorry

end NUMINAMATH_GPT_spoiled_apples_l1126_112644


namespace NUMINAMATH_GPT_market_value_of_13_percent_stock_yielding_8_percent_l1126_112670

noncomputable def market_value_of_stock (yield rate dividend_per_share : ℝ) : ℝ :=
  (dividend_per_share / yield) * 100

theorem market_value_of_13_percent_stock_yielding_8_percent
  (yield_rate : ℝ) (dividend_per_share : ℝ) (market_value : ℝ)
  (h_yield_rate : yield_rate = 0.08)
  (h_dividend_per_share : dividend_per_share = 13) :
  market_value = 162.50 :=
by
  sorry

end NUMINAMATH_GPT_market_value_of_13_percent_stock_yielding_8_percent_l1126_112670


namespace NUMINAMATH_GPT_intersection_points_l1126_112613

theorem intersection_points (a : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 = a^2) ↔ (y = x^2 - 2 * a)) ↔ (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_GPT_intersection_points_l1126_112613


namespace NUMINAMATH_GPT_find_ratio_of_radii_l1126_112633

noncomputable def ratio_of_radii (a b : ℝ) (h1 : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2) : Prop :=
  a / b = Real.sqrt 5 / 5

theorem find_ratio_of_radii (a b : ℝ) (h1 : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2) :
  ratio_of_radii a b h1 :=
sorry

end NUMINAMATH_GPT_find_ratio_of_radii_l1126_112633


namespace NUMINAMATH_GPT_large_number_exponent_l1126_112654

theorem large_number_exponent (h : 10000 = 10 ^ 4) : 10000 ^ 50 * 10 ^ 5 = 10 ^ 205 := 
by
  sorry

end NUMINAMATH_GPT_large_number_exponent_l1126_112654


namespace NUMINAMATH_GPT_average_speed_l1126_112655

def s (t : ℝ) : ℝ := 3 + t^2

theorem average_speed {t1 t2 : ℝ} (h1 : t1 = 2) (h2: t2 = 2.1) :
  (s t2 - s t1) / (t2 - t1) = 4.1 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_l1126_112655


namespace NUMINAMATH_GPT_proof_a_squared_plus_1_l1126_112604

theorem proof_a_squared_plus_1 (a : ℤ) (h1 : 3 < a) (h2 : a < 5) : a^2 + 1 = 17 :=
  by
  sorry

end NUMINAMATH_GPT_proof_a_squared_plus_1_l1126_112604


namespace NUMINAMATH_GPT_shaded_regions_area_sum_l1126_112611

theorem shaded_regions_area_sum (side_len : ℚ) (radius : ℚ) (a b c : ℤ) :
  side_len = 16 → radius = side_len / 2 →
  a = (64 / 3) ∧ b = 32 ∧ c = 3 →
  (∃ x : ℤ, x = a + b + c ∧ x = 99) :=
by
  intros hside_len hradius h_constituents
  sorry

end NUMINAMATH_GPT_shaded_regions_area_sum_l1126_112611


namespace NUMINAMATH_GPT_least_number_with_remainder_l1126_112624

theorem least_number_with_remainder (x : ℕ) :
  (x % 6 = 4) ∧ (x % 7 = 4) ∧ (x % 9 = 4) ∧ (x % 18 = 4) ↔ x = 130 :=
by
  sorry

end NUMINAMATH_GPT_least_number_with_remainder_l1126_112624


namespace NUMINAMATH_GPT_part1_part2_l1126_112664

noncomputable def a_n (n : ℕ) : ℕ :=
  2^(n - 1)

noncomputable def b_n (n : ℕ) : ℕ :=
  2 * n

noncomputable def S_n (n : ℕ) : ℕ :=
  n^2 + n

theorem part1 (n : ℕ) : 
  S_n n = n^2 + n := 
sorry

noncomputable def C_n (n : ℕ) : ℚ :=
  (n^2 + n) / 2^(n - 1)

theorem part2 (n : ℕ) (k : ℕ) (k_gt_0 : 0 < k) : 
  (∀ n, C_n n ≤ C_n k) ↔ (k = 2 ∨ k = 3) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1126_112664


namespace NUMINAMATH_GPT_least_positive_integer_l1126_112602

theorem least_positive_integer (n : ℕ) (h1 : n > 1) 
  (h2 : n % 2 = 1) (h3 : n % 3 = 1) (h4 : n % 5 = 1) 
  (h5 : n % 7 = 1) (h6 : n % 11 = 1): 
  n = 2311 := 
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_l1126_112602


namespace NUMINAMATH_GPT_tangent_line_at_point_P_l1126_112635

-- Definitions from Conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5
def point_on_circle : Prop := circle_eq 1 2

-- Statement to Prove
theorem tangent_line_at_point_P : 
  point_on_circle → ∃ (m : ℝ) (b : ℝ), (m = -1/2) ∧ (b = 5/2) ∧ (∀ x y : ℝ, y = m * x + b ↔ x + 2 * y - 5 = 0) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_point_P_l1126_112635


namespace NUMINAMATH_GPT_perfect_square_factors_450_l1126_112600

theorem perfect_square_factors_450 :
  ∃ (n : ℕ), n = 4 ∧ ∀ (d : ℕ), d ∣ 450 → ∃ (k : ℕ), k^2 = d ↔ d = 1 ∨ d = 9 ∨ d = 25 ∨ d = 225 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_factors_450_l1126_112600


namespace NUMINAMATH_GPT_total_cost_is_96_l1126_112684

noncomputable def hair_updo_cost : ℕ := 50
noncomputable def manicure_cost : ℕ := 30
noncomputable def tip_rate : ℚ := 0.20

def total_cost_with_tip (hair_cost manicure_cost : ℕ) (tip_rate : ℚ) : ℚ :=
  let hair_tip := hair_cost * tip_rate
  let manicure_tip := manicure_cost * tip_rate
  let total_tips := hair_tip + manicure_tip
  let total_before_tips := (hair_cost : ℚ) + (manicure_cost : ℚ)
  total_before_tips + total_tips

theorem total_cost_is_96 :
  total_cost_with_tip hair_updo_cost manicure_cost tip_rate = 96 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_96_l1126_112684


namespace NUMINAMATH_GPT_john_paid_more_than_jane_by_540_l1126_112649

noncomputable def original_price : ℝ := 36.000000000000036
noncomputable def discount_percentage : ℝ := 0.10
noncomputable def tip_percentage : ℝ := 0.15

noncomputable def discounted_price : ℝ := original_price * (1 - discount_percentage)
noncomputable def john_tip : ℝ := original_price * tip_percentage
noncomputable def jane_tip : ℝ := discounted_price * tip_percentage

noncomputable def john_total_payment : ℝ := discounted_price + john_tip
noncomputable def jane_total_payment : ℝ := discounted_price + jane_tip

noncomputable def difference : ℝ := john_total_payment - jane_total_payment

theorem john_paid_more_than_jane_by_540 :
  difference = 0.5400000000000023 := sorry

end NUMINAMATH_GPT_john_paid_more_than_jane_by_540_l1126_112649


namespace NUMINAMATH_GPT_faster_train_speed_l1126_112698

theorem faster_train_speed
  (length_per_train : ℝ)
  (speed_slower_train : ℝ)
  (passing_time_secs : ℝ)
  (speed_faster_train : ℝ) :
  length_per_train = 80 / 1000 →
  speed_slower_train = 36 →
  passing_time_secs = 36 →
  speed_faster_train = 52 :=
by
  intro h_length_per_train h_speed_slower_train h_passing_time_secs
  -- Skipped steps would go here
  sorry

end NUMINAMATH_GPT_faster_train_speed_l1126_112698


namespace NUMINAMATH_GPT_box_volume_of_pyramid_l1126_112689

/-- A theorem to prove the volume of the smallest cube-shaped box that can house the given rectangular pyramid. -/
theorem box_volume_of_pyramid :
  (∀ (h l w : ℕ), h = 15 ∧ l = 8 ∧ w = 12 → (∀ (v : ℕ), v = (max h (max l w)) ^ 3 → v = 3375)) :=
by
  intros h l w h_condition v v_def
  sorry

end NUMINAMATH_GPT_box_volume_of_pyramid_l1126_112689


namespace NUMINAMATH_GPT_general_term_formula_sum_of_first_n_terms_l1126_112631

noncomputable def a (n : ℕ) : ℕ :=
(n + 2^n)^2

theorem general_term_formula :
  ∀ n : ℕ, a n = n^2 + n * 2^(n+1) + 4^n :=
sorry

noncomputable def S (n : ℕ) : ℕ :=
(n-1) * 2^(n+2) + 4 + (4^(n+1) - 4) / 3

theorem sum_of_first_n_terms :
  ∀ n : ℕ, S n = (n-1) * 2^(n+2) + 4 + (4^(n+1) - 4) / 3 :=
sorry

end NUMINAMATH_GPT_general_term_formula_sum_of_first_n_terms_l1126_112631


namespace NUMINAMATH_GPT_solve_quadratic_l1126_112662

theorem solve_quadratic (x : ℝ) (h : x^2 - 6*x + 8 = 0) : x = 2 ∨ x = 4 :=
sorry

end NUMINAMATH_GPT_solve_quadratic_l1126_112662


namespace NUMINAMATH_GPT_correct_bushes_needed_l1126_112637

def yield_per_bush := 10
def containers_per_zucchini := 3
def zucchinis_needed := 36
def bushes_needed (yield_per_bush containers_per_zucchini zucchinis_needed : ℕ) : ℕ :=
  Nat.ceil ((zucchinis_needed * containers_per_zucchini : ℕ) / yield_per_bush)

theorem correct_bushes_needed : bushes_needed yield_per_bush containers_per_zucchini zucchinis_needed = 11 := 
by
  sorry

end NUMINAMATH_GPT_correct_bushes_needed_l1126_112637


namespace NUMINAMATH_GPT_marble_203_is_green_l1126_112625

-- Define the conditions
def total_marbles : ℕ := 240
def cycle_length : ℕ := 15
def red_count : ℕ := 6
def green_count : ℕ := 5
def blue_count : ℕ := 4
def marble_pattern (n : ℕ) : String :=
  if n % cycle_length < red_count then "red"
  else if n % cycle_length < red_count + green_count then "green"
  else "blue"

-- Define the color of the 203rd marble
def marble_203 : String := marble_pattern 202

-- State the theorem
theorem marble_203_is_green : marble_203 = "green" :=
by
  sorry

end NUMINAMATH_GPT_marble_203_is_green_l1126_112625


namespace NUMINAMATH_GPT_tan_beta_minus_2alpha_l1126_112626

noncomputable def tan_alpha := 1 / 2
noncomputable def tan_beta_minus_alpha := 2 / 5
theorem tan_beta_minus_2alpha (α β : ℝ) (h1 : Real.tan α = tan_alpha) (h2 : Real.tan (β - α) = tan_beta_minus_alpha) :
  Real.tan (β - 2 * α) = -1 / 12 := 
by
  sorry

end NUMINAMATH_GPT_tan_beta_minus_2alpha_l1126_112626


namespace NUMINAMATH_GPT_smaller_side_of_rectangle_l1126_112642

theorem smaller_side_of_rectangle (r : ℝ) (h1 : r = 42) 
                                   (h2 : ∀ L W : ℝ, L / W = 6 / 5 → 2 * (L + W) = 2 * π * r) : 
                                   ∃ W : ℝ, W = (210 * π) / 11 := 
by {
    sorry
}

end NUMINAMATH_GPT_smaller_side_of_rectangle_l1126_112642


namespace NUMINAMATH_GPT_monomials_like_terms_l1126_112669

theorem monomials_like_terms (a b : ℝ) (m n : ℤ) 
  (h1 : 2 * (a^4) * (b^(-2 * m + 7)) = 3 * (a^(2 * m)) * (b^(n + 2))) :
  m + n = 3 := 
by {
  -- Our proof will be placed here
  sorry
}

end NUMINAMATH_GPT_monomials_like_terms_l1126_112669


namespace NUMINAMATH_GPT_total_money_collected_l1126_112695

theorem total_money_collected (attendees : ℕ) (reserved_price unreserved_price : ℝ) (reserved_sold unreserved_sold : ℕ)
  (h_attendees : attendees = 1096)
  (h_reserved_price : reserved_price = 25.00)
  (h_unreserved_price : unreserved_price = 20.00)
  (h_reserved_sold : reserved_sold = 246)
  (h_unreserved_sold : unreserved_sold = 246) :
  (reserved_price * reserved_sold + unreserved_price * unreserved_sold) = 11070.00 :=
by
  sorry

end NUMINAMATH_GPT_total_money_collected_l1126_112695


namespace NUMINAMATH_GPT_fraction_compare_l1126_112694

theorem fraction_compare (a b c d e : ℚ) : 
  a = 0.3333333 → 
  b = 1 / (3 * 10^6) →
  ∃ x : ℚ, 
  x = 1 / 3 ∧ 
  (x > a + d ∧ 
   x = a + b ∧
   d = b ∧
   d = -1 / (3 * 10^6)) := 
  sorry

end NUMINAMATH_GPT_fraction_compare_l1126_112694


namespace NUMINAMATH_GPT_frog_jumps_further_l1126_112663

-- Definitions according to conditions
def grasshopper_jump : ℕ := 36
def frog_jump : ℕ := 53

-- Theorem: The frog jumped 17 inches farther than the grasshopper
theorem frog_jumps_further (g_jump f_jump : ℕ) (h1 : g_jump = grasshopper_jump) (h2 : f_jump = frog_jump) :
  f_jump - g_jump = 17 :=
by
  -- Proof is skipped in this statement
  sorry

end NUMINAMATH_GPT_frog_jumps_further_l1126_112663


namespace NUMINAMATH_GPT_division_problem_l1126_112619

variables (a b c : ℤ)

theorem division_problem 
  (h1 : a ∣ b * c - 1)
  (h2 : b ∣ c * a - 1)
  (h3 : c ∣ a * b - 1) : 
  abc ∣ ab + bc + ca - 1 := 
sorry

end NUMINAMATH_GPT_division_problem_l1126_112619


namespace NUMINAMATH_GPT_walnut_trees_planted_today_l1126_112630

-- Define the number of walnut trees before planting
def walnut_trees_before_planting : ℕ := 22

-- Define the number of walnut trees after planting
def walnut_trees_after_planting : ℕ := 55

-- Define a theorem to prove the number of walnut trees planted
theorem walnut_trees_planted_today : 
  walnut_trees_after_planting - walnut_trees_before_planting = 33 :=
by
  -- The proof will be inserted here.
  sorry

end NUMINAMATH_GPT_walnut_trees_planted_today_l1126_112630


namespace NUMINAMATH_GPT_digits_sum_is_15_l1126_112636

theorem digits_sum_is_15 (f o g : ℕ) (h1 : f * 100 + o * 10 + g = 366) (h2 : 4 * (f * 100 + o * 10 + g) = 1464) (h3 : f < 10 ∧ o < 10 ∧ g < 10) :
  f + o + g = 15 :=
sorry

end NUMINAMATH_GPT_digits_sum_is_15_l1126_112636


namespace NUMINAMATH_GPT_unique_zero_of_function_l1126_112606

theorem unique_zero_of_function (a : ℝ) :
  (∃! x : ℝ, e^(abs x) + 2 * a - 1 = 0) ↔ a = 0 := 
by 
  sorry

end NUMINAMATH_GPT_unique_zero_of_function_l1126_112606


namespace NUMINAMATH_GPT_total_payroll_l1126_112629

theorem total_payroll 
  (heavy_operator_pay : ℕ) 
  (laborer_pay : ℕ) 
  (total_people : ℕ) 
  (laborers : ℕ)
  (heavy_operators : ℕ)
  (total_payroll : ℕ)
  (h1: heavy_operator_pay = 140)
  (h2: laborer_pay = 90)
  (h3: total_people = 35)
  (h4: laborers = 19)
  (h5: heavy_operators = total_people - laborers)
  (h6: total_payroll = (heavy_operators * heavy_operator_pay) + (laborers * laborer_pay)) :
  total_payroll = 3950 :=
by sorry

end NUMINAMATH_GPT_total_payroll_l1126_112629


namespace NUMINAMATH_GPT_cos_alpha_value_l1126_112620
open Real

theorem cos_alpha_value (α : ℝ) (h0 : 0 < α ∧ α < π / 2) 
  (h1 : sin (α - π / 6) = 1 / 3) : 
  cos α = (2 * sqrt 6 - 1) / 6 := 
by 
  sorry

end NUMINAMATH_GPT_cos_alpha_value_l1126_112620


namespace NUMINAMATH_GPT_num_cages_l1126_112641

-- Define the conditions as given
def parrots_per_cage : ℕ := 8
def parakeets_per_cage : ℕ := 2
def total_birds_in_store : ℕ := 40

-- Prove that the number of bird cages is 4
theorem num_cages (x : ℕ) (h : 10 * x = total_birds_in_store) : x = 4 :=
sorry

end NUMINAMATH_GPT_num_cages_l1126_112641


namespace NUMINAMATH_GPT_intersection_points_of_parabolas_l1126_112639

/-- Let P1 be the equation of the first parabola: y = 3x^2 - 8x + 2 -/
def P1 (x : ℝ) : ℝ := 3 * x^2 - 8 * x + 2

/-- Let P2 be the equation of the second parabola: y = 6x^2 + 4x + 2 -/
def P2 (x : ℝ) : ℝ := 6 * x^2 + 4 * x + 2

/-- Prove that the intersection points of P1 and P2 are (-4, 82) and (0, 2) -/
theorem intersection_points_of_parabolas : 
  {p : ℝ × ℝ | ∃ x, p = (x, P1 x) ∧ P1 x = P2 x} = 
    {(-4, 82), (0, 2)} :=
sorry

end NUMINAMATH_GPT_intersection_points_of_parabolas_l1126_112639


namespace NUMINAMATH_GPT_min_groups_with_conditions_l1126_112643

theorem min_groups_with_conditions (n a b m : ℕ) (h_n : n = 8) (h_a : a = 4) (h_b : b = 1) :
  m ≥ 2 :=
sorry

end NUMINAMATH_GPT_min_groups_with_conditions_l1126_112643


namespace NUMINAMATH_GPT_articles_production_l1126_112674

theorem articles_production (x y : ℕ) (e : ℝ) :
  (x * x * x * e / x = x^2 * e) → (y * (y + 2) * y * (e / x) = (e * y * (y^2 + 2 * y)) / x) :=
by 
  sorry

end NUMINAMATH_GPT_articles_production_l1126_112674


namespace NUMINAMATH_GPT_problem_statement_l1126_112692

theorem problem_statement (w x y z : ℕ) (h : 2^w * 3^x * 5^y * 7^z = 882) : 2 * w + 3 * x + 5 * y + 7 * z = 22 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1126_112692


namespace NUMINAMATH_GPT_product_of_decimals_l1126_112645

theorem product_of_decimals :
  0.5 * 0.8 = 0.40 :=
by
  -- Proof will go here; using sorry to skip for now
  sorry

end NUMINAMATH_GPT_product_of_decimals_l1126_112645


namespace NUMINAMATH_GPT_largest_common_multiple_of_7_8_l1126_112660

noncomputable def largest_common_multiple_of_7_8_sub_2 (n : ℕ) : ℕ :=
  if n <= 100 then n else 0

theorem largest_common_multiple_of_7_8 :
  ∃ x : ℕ, x <= 100 ∧ (x - 2) % Nat.lcm 7 8 = 0 ∧ x = 58 :=
by
  let x := 58
  use x
  have h1 : x <= 100 := by norm_num
  have h2 : (x - 2) % Nat.lcm 7 8 = 0 := by norm_num
  have h3 : x = 58 := by norm_num
  exact ⟨h1, h2, h3⟩

end NUMINAMATH_GPT_largest_common_multiple_of_7_8_l1126_112660


namespace NUMINAMATH_GPT_total_canoes_boatsRUs_l1126_112614

-- Definitions for the conditions
def initial_production := 10
def common_ratio := 3
def months := 6

-- The function to compute the total number of canoes built using the geometric sequence sum formula
noncomputable def total_canoes (a : ℕ) (r : ℕ) (n : ℕ) := a * (r^n - 1) / (r - 1)

-- Statement of the theorem
theorem total_canoes_boatsRUs : 
  total_canoes initial_production common_ratio months = 3640 :=
sorry

end NUMINAMATH_GPT_total_canoes_boatsRUs_l1126_112614


namespace NUMINAMATH_GPT_complex_number_solution_l1126_112678

theorem complex_number_solution (z : ℂ) (i : ℂ) (hi : i * i = -1) (h : i * z = 1) : z = -i :=
by
  -- Mathematical proof will be here
  sorry

end NUMINAMATH_GPT_complex_number_solution_l1126_112678


namespace NUMINAMATH_GPT_john_has_22_quarters_l1126_112610

-- Definitions based on conditions
def number_of_quarters (Q : ℕ) : ℕ := Q
def number_of_dimes (Q : ℕ) : ℕ := Q + 3
def number_of_nickels (Q : ℕ) : ℕ := Q - 6

-- Total number of coins condition
def total_number_of_coins (Q : ℕ) : Prop := 
  (number_of_quarters Q) + (number_of_dimes Q) + (number_of_nickels Q) = 63

-- Goal: Proving the number of quarters is 22
theorem john_has_22_quarters : ∃ Q : ℕ, total_number_of_coins Q ∧ Q = 22 :=
by
  -- Proof skipped 
  sorry

end NUMINAMATH_GPT_john_has_22_quarters_l1126_112610


namespace NUMINAMATH_GPT_force_on_dam_l1126_112666

noncomputable def calculate_force (ρ g a b h : ℝ) :=
  ρ * g * h^2 * (b / 2 - (b - a) / 3)

theorem force_on_dam :
  let ρ := 1000
  let g := 10
  let a := 6.0
  let b := 9.6
  let h := 4.0
  calculate_force ρ g a b h = 576000 :=
by sorry

end NUMINAMATH_GPT_force_on_dam_l1126_112666


namespace NUMINAMATH_GPT_total_employees_l1126_112677

-- Defining the number of part-time and full-time employees
def p : ℕ := 2041
def f : ℕ := 63093

-- Statement that the total number of employees is the sum of part-time and full-time employees
theorem total_employees : p + f = 65134 :=
by
  -- Use Lean's built-in arithmetic to calculate the sum
  rfl

end NUMINAMATH_GPT_total_employees_l1126_112677


namespace NUMINAMATH_GPT_chess_tournament_participants_l1126_112696

-- Define the number of grandmasters
variables (x : ℕ)

-- Define the number of masters as three times the number of grandmasters
def num_masters : ℕ := 3 * x

-- Condition on total points scored: Master's points is 1.2 times the Grandmaster's points
def points_condition (g m : ℕ) : Prop := m = 12 * g / 10

-- Proposition that the total number of participants is 12
theorem chess_tournament_participants (x_nonnegative: 0 < x) (g m : ℕ)
  (masters_points: points_condition g m) : 
  4 * x = 12 := 
sorry

end NUMINAMATH_GPT_chess_tournament_participants_l1126_112696


namespace NUMINAMATH_GPT_min_dancers_l1126_112651

theorem min_dancers (N : ℕ) (h1 : N % 4 = 0) (h2 : N % 9 = 0) (h3 : N % 10 = 0) (h4 : N > 50) : N = 180 :=
  sorry

end NUMINAMATH_GPT_min_dancers_l1126_112651


namespace NUMINAMATH_GPT_min_trips_to_fill_hole_l1126_112687

def hole_filling_trips (initial_gallons : ℕ) (required_gallons : ℕ) (capacity_2gallon : ℕ)
  (capacity_5gallon : ℕ) (capacity_8gallon : ℕ) (time_limit : ℕ) (time_per_trip : ℕ) : ℕ :=
  if initial_gallons < required_gallons then
    let remaining_gallons := required_gallons - initial_gallons
    let num_8gallon := remaining_gallons / capacity_8gallon
    let remaining_after_8gallon := remaining_gallons % capacity_8gallon
    let num_2gallon := if remaining_after_8gallon = 3 then 1 else 0
    let num_5gallon := if remaining_after_8gallon = 3 then 1 else remaining_after_8gallon / capacity_5gallon
    let total_trips := num_8gallon + num_2gallon + num_5gallon
    if total_trips <= time_limit / time_per_trip then
      total_trips
    else
      sorry -- If calculations overflow time limit
  else
    0

theorem min_trips_to_fill_hole : 
  hole_filling_trips 676 823 2 5 8 45 1 = 20 :=
by rfl

end NUMINAMATH_GPT_min_trips_to_fill_hole_l1126_112687


namespace NUMINAMATH_GPT_johns_sixth_quiz_score_l1126_112690

theorem johns_sixth_quiz_score (s1 s2 s3 s4 s5 : ℕ) (mean : ℕ) (n : ℕ) :
  s1 = 86 ∧ s2 = 91 ∧ s3 = 83 ∧ s4 = 88 ∧ s5 = 97 ∧ mean = 90 ∧ n = 6 →
  ∃ s6 : ℕ, (s1 + s2 + s3 + s4 + s5 + s6) / n = mean ∧ s6 = 95 :=
by
  intro h
  obtain ⟨hs1, hs2, hs3, hs4, hs5, hmean, hn⟩ := h
  have htotal : s1 + s2 + s3 + s4 + s5 + 95 = 540 := by sorry
  have hmean_eq : (s1 + s2 + s3 + s4 + s5 + 95) / n = mean := by sorry
  exact ⟨95, hmean_eq, rfl⟩

end NUMINAMATH_GPT_johns_sixth_quiz_score_l1126_112690


namespace NUMINAMATH_GPT_pie_eating_contest_difference_l1126_112693

-- Definition of given conditions
def num_students := 8
def emma_pies := 8
def sam_pies := 1

-- Statement to prove
theorem pie_eating_contest_difference :
  emma_pies - sam_pies = 7 :=
by
  -- Omitting the proof, as requested.
  sorry

end NUMINAMATH_GPT_pie_eating_contest_difference_l1126_112693


namespace NUMINAMATH_GPT_choir_members_minimum_l1126_112650

theorem choir_members_minimum (n : ℕ) : (∃ n, n % 8 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0 ∧ ∀ m, (m % 8 = 0 ∧ m % 9 = 0 ∧ m % 10 = 0) → n ≤ m) → n = 360 :=
by
  sorry

end NUMINAMATH_GPT_choir_members_minimum_l1126_112650


namespace NUMINAMATH_GPT_tom_searching_days_l1126_112618

variable (d : ℕ) (total_cost : ℕ)

theorem tom_searching_days :
  (∀ n, n ≤ 5 → total_cost = n * 100 + (d - n) * 60) →
  (∀ n, n > 5 → total_cost = 5 * 100 + (d - 5) * 60) →
  total_cost = 800 →
  d = 10 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_tom_searching_days_l1126_112618


namespace NUMINAMATH_GPT_range_of_a_l1126_112672

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x < - (Real.sqrt 3) / 3 ∨ x > (Real.sqrt 3) / 3 →
    a * (3 * x^2 - 1) > 0) →
  a > 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1126_112672


namespace NUMINAMATH_GPT_sara_no_ingredients_pies_l1126_112632

theorem sara_no_ingredients_pies:
  ∀ (total_pies : ℕ) (berries_pies : ℕ) (cream_pies : ℕ) (nuts_pies : ℕ) (coconut_pies : ℕ),
  total_pies = 60 →
  berries_pies = 1/3 * total_pies →
  cream_pies = 1/2 * total_pies →
  nuts_pies = 3/5 * total_pies →
  coconut_pies = 1/5 * total_pies →
  (total_pies - nuts_pies) = 24 :=
by
  intros total_pies berries_pies cream_pies nuts_pies coconut_pies ht hb hc hn hcoc
  sorry

end NUMINAMATH_GPT_sara_no_ingredients_pies_l1126_112632


namespace NUMINAMATH_GPT_complex_equilateral_triangle_expression_l1126_112665

noncomputable def omega : ℂ :=
  Complex.exp (Complex.I * 2 * Real.pi / 3)

def is_root_of_quadratic (z : ℂ) (a b : ℂ) : Prop :=
  z^2 + a * z + b = 0

theorem complex_equilateral_triangle_expression (z1 z2 a b : ℂ) (h1 : is_root_of_quadratic z1 a b) 
  (h2 : is_root_of_quadratic z2 a b) (h3 : z2 = omega * z1) : a^2 / b = 1 := by
  sorry

end NUMINAMATH_GPT_complex_equilateral_triangle_expression_l1126_112665


namespace NUMINAMATH_GPT_necessary_not_sufficient_l1126_112679

theorem necessary_not_sufficient (x : ℝ) : (x^2 ≥ 1) ↔ (x ≥ 1 ∨ x ≤ -1) ≠ (x ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_necessary_not_sufficient_l1126_112679


namespace NUMINAMATH_GPT_cannot_bisect_abs_function_l1126_112686

theorem cannot_bisect_abs_function 
  (f : ℝ → ℝ)
  (hf1 : ∀ x, f x = |x|) :
  ¬ (∃ a b, a < b ∧ f a * f b < 0) :=
by
  sorry

end NUMINAMATH_GPT_cannot_bisect_abs_function_l1126_112686


namespace NUMINAMATH_GPT_geo_seq_fifth_term_l1126_112612

theorem geo_seq_fifth_term (a : ℕ → ℝ) (q : ℝ) (h1 : q = 2) (h2 : a 3 = 3) :
  a 5 = 12 := 
sorry

end NUMINAMATH_GPT_geo_seq_fifth_term_l1126_112612


namespace NUMINAMATH_GPT_range_of_a_l1126_112628

variable (a : ℝ)
def f (x : ℝ) : ℝ := a * x^2 - 2 * a * x - 4

theorem range_of_a :
  (∀ x : ℝ, f a x < 0) → (-4 < a ∧ a ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1126_112628


namespace NUMINAMATH_GPT_geometric_series_common_ratio_l1126_112653

theorem geometric_series_common_ratio (a S r : ℝ) (h1 : a = 512) (h2 : S = 3072) 
(h3 : S = a / (1 - r)) : r = 5/6 := 
sorry

end NUMINAMATH_GPT_geometric_series_common_ratio_l1126_112653


namespace NUMINAMATH_GPT_min_value_of_f_l1126_112627

def f (x y : ℝ) : ℝ := x^3 + y^3 + x^2 * y + x * y^2 - 3 * (x^2 + y^2 + x * y) + 3 * (x + y)

theorem min_value_of_f : ∀ x y : ℝ, x ≥ 1/2 → y ≥ 1/2 → f x y ≥ 1
    := by
      intros x y hx hy
      -- Rest of the proof would go here
      sorry

end NUMINAMATH_GPT_min_value_of_f_l1126_112627


namespace NUMINAMATH_GPT_sum_of_roots_l1126_112607

theorem sum_of_roots (r s t : ℝ) (h : 3 * r * s * t - 9 * (r * s + s * t + t * r) - 28 * (r + s + t) + 12 = 0) : r + s + t = 3 :=
by sorry

end NUMINAMATH_GPT_sum_of_roots_l1126_112607


namespace NUMINAMATH_GPT_find_larger_number_l1126_112622

theorem find_larger_number (L S : ℕ) (h1 : L - S = 2415) (h2 : L = 21 * S + 15) : L = 2535 := 
by
  sorry

end NUMINAMATH_GPT_find_larger_number_l1126_112622
