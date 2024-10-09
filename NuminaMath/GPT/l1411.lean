import Mathlib

namespace gallons_of_gas_l1411_141118

-- Define the conditions
def mpg : ℕ := 19
def d1 : ℕ := 15
def d2 : ℕ := 6
def d3 : ℕ := 2
def d4 : ℕ := 4
def d5 : ℕ := 11

-- The theorem to prove
theorem gallons_of_gas : (d1 + d2 + d3 + d4 + d5) / mpg = 2 := 
by {
    sorry
}

end gallons_of_gas_l1411_141118


namespace jackson_fishes_per_day_l1411_141167

def total_fishes : ℕ := 90
def jonah_per_day : ℕ := 4
def george_per_day : ℕ := 8
def competition_days : ℕ := 5

def jackson_per_day (J : ℕ) : Prop :=
  (total_fishes - (jonah_per_day * competition_days + george_per_day * competition_days)) / competition_days = J

theorem jackson_fishes_per_day : jackson_per_day 6 :=
  by
    sorry

end jackson_fishes_per_day_l1411_141167


namespace calendar_sum_multiple_of_4_l1411_141197

theorem calendar_sum_multiple_of_4 (a : ℕ) : 
  let top_left := a - 1
  let bottom_left := a + 6
  let bottom_right := a + 7
  top_left + a + bottom_left + bottom_right = 4 * (a + 3) :=
by
  sorry

end calendar_sum_multiple_of_4_l1411_141197


namespace encoded_base5_to_base10_l1411_141157

-- Given definitions
def base5_to_int (d1 d2 d3 : ℕ) : ℕ := d1 * 25 + d2 * 5 + d3

def V := 2
def W := 0
def X := 4
def Y := 1
def Z := 3

-- Prove that the base-10 expression for the integer coded as XYZ is 108
theorem encoded_base5_to_base10 :
  base5_to_int X Y Z = 108 :=
sorry

end encoded_base5_to_base10_l1411_141157


namespace cubic_inches_in_two_cubic_feet_l1411_141183

theorem cubic_inches_in_two_cubic_feet (conv : 1 = 12) : 2 * (12 * 12 * 12) = 3456 :=
by
  sorry

end cubic_inches_in_two_cubic_feet_l1411_141183


namespace days_to_empty_tube_l1411_141134

-- Define the conditions
def gelInTube : ℕ := 128
def dailyUsage : ℕ := 4

-- Define the proof statement
theorem days_to_empty_tube : gelInTube / dailyUsage = 32 := 
by 
  sorry

end days_to_empty_tube_l1411_141134


namespace find_f_pi_over_4_l1411_141101

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem find_f_pi_over_4
  (ω φ : ℝ)
  (hω_gt_0 : ω > 0)
  (hφ_lt_pi_over_2 : |φ| < Real.pi / 2)
  (h_mono_dec : ∀ x₁ x₂, (Real.pi / 6 < x₁ ∧ x₁ < Real.pi / 3 ∧ Real.pi / 3 < x₂ ∧ x₂ < 2 * Real.pi / 3) → f x₁ ω φ > f x₂ ω φ)
  (h_values_decreasing : f (Real.pi / 6) ω φ = 1 ∧ f (2 * Real.pi / 3) ω φ = -1) : 
  f (Real.pi / 4) 2 (Real.pi / 6) = Real.sqrt 3 / 2 :=
sorry

end find_f_pi_over_4_l1411_141101


namespace identify_perfect_square_is_689_l1411_141109

-- Definitions of the conditions
def natural_numbers (n : ℕ) : Prop := True -- All natural numbers are accepted
def digits_in_result (n m : ℕ) (d : ℕ) : Prop := (n * m) % 1000 = d

-- Theorem to be proved
theorem identify_perfect_square_is_689 (n : ℕ) :
  (∀ m, natural_numbers m → digits_in_result m m 689 ∨ digits_in_result m m 759) →
  ∃ m, natural_numbers m ∧ digits_in_result m m 689 :=
sorry

end identify_perfect_square_is_689_l1411_141109


namespace least_M_bench_sections_l1411_141195

/--
A single bench section at a community event can hold either 8 adults, 12 children, or 10 teenagers. 
We are to find the smallest positive integer M such that when M bench sections are connected end to end,
an equal number of adults, children, and teenagers seated together will occupy all the bench space.
-/
theorem least_M_bench_sections
  (M : ℕ)
  (hM_pos : M > 0)
  (adults_capacity : ℕ := 8 * M)
  (children_capacity : ℕ := 12 * M)
  (teenagers_capacity : ℕ := 10 * M)
  (h_equal_capacity : adults_capacity = children_capacity ∧ children_capacity = teenagers_capacity) :
  M = 15 := 
sorry

end least_M_bench_sections_l1411_141195


namespace negation_of_universal_proposition_l1411_141132

theorem negation_of_universal_proposition :
  (¬ ∀ x > 1, (1 / 2)^x < 1 / 2) ↔ (∃ x > 1, (1 / 2)^x ≥ 1 / 2) :=
sorry

end negation_of_universal_proposition_l1411_141132


namespace find_x_l1411_141127

-- Define the vectors a, b, and c
def a : ℝ × ℝ := (-2, 0)
def b : ℝ × ℝ := (2, 1)
def c (x : ℝ) : ℝ × ℝ := (x, 1)

-- Define the collinearity condition
def collinear_with_3a_plus_b (x : ℝ) : Prop :=
  ∃ k : ℝ, c x = k • (3 • a + b)

theorem find_x :
  ∀ x : ℝ, collinear_with_3a_plus_b x → x = -4 := 
sorry

end find_x_l1411_141127


namespace number_of_triangles_in_polygon_l1411_141153

theorem number_of_triangles_in_polygon {n : ℕ} (h : n > 0) :
  let vertices := (2 * n + 1)
  ∃ triangles_containing_center : ℕ, triangles_containing_center = (n * (n + 1) * (2 * n + 1)) / 6 :=
sorry

end number_of_triangles_in_polygon_l1411_141153


namespace remainder_987654_div_8_l1411_141151

theorem remainder_987654_div_8 : 987654 % 8 = 2 := by
  sorry

end remainder_987654_div_8_l1411_141151


namespace rectangle_area_y_l1411_141148

theorem rectangle_area_y (y : ℚ) (h_pos: y > 0) 
  (h_area: ((6 : ℚ) - (-2)) * (y - 2) = 64) : y = 10 :=
by
  sorry

end rectangle_area_y_l1411_141148


namespace cube_root_neg_eight_l1411_141160

theorem cube_root_neg_eight : ∃ x : ℝ, x^3 = -8 ∧ x = -2 :=
by {
  sorry
}

end cube_root_neg_eight_l1411_141160


namespace S_40_eq_150_l1411_141193

variable {R : Type*} [Field R]

-- Define the sum function for geometric sequences.
noncomputable def geom_sum (a q : R) (n : ℕ) : R :=
  a * (1 - q^n) / (1 - q)

-- Given conditions from the problem.
axiom S_10_eq : ∀ {a q : R}, geom_sum a q 10 = 10
axiom S_30_eq : ∀ {a q : R}, geom_sum a q 30 = 70

-- The main theorem stating S40 = 150 under the given conditions.
theorem S_40_eq_150 {a q : R} (h10 : geom_sum a q 10 = 10) (h30 : geom_sum a q 30 = 70) :
  geom_sum a q 40 = 150 :=
sorry

end S_40_eq_150_l1411_141193


namespace probability_of_9_heads_in_12_l1411_141137

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l1411_141137


namespace garden_roller_diameter_l1411_141169

theorem garden_roller_diameter 
  (length : ℝ) 
  (total_area : ℝ) 
  (num_revolutions : ℕ) 
  (pi : ℝ) 
  (A : length = 2)
  (B : total_area = 37.714285714285715)
  (C : num_revolutions = 5)
  (D : pi = 22 / 7) : 
  ∃ d : ℝ, d = 1.2 :=
by
  sorry

end garden_roller_diameter_l1411_141169


namespace gcd_8251_6105_l1411_141177

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l1411_141177


namespace tan_product_equals_three_l1411_141100

noncomputable def tan_pi_div_9 := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_div_9 := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_div_9 := Real.tan (4 * Real.pi / 9)

theorem tan_product_equals_three 
  (h1 : tan_pi_div_9 ≠ 0)
  (h2 : tan_2pi_div_9 ≠ 0)
  (h3 : tan_4pi_div_9 ≠ 0)
  (h4 : tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3) :
  tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3 :=
by
  sorry

end tan_product_equals_three_l1411_141100


namespace brendas_age_l1411_141107

theorem brendas_age (A B J : ℕ) 
  (h1 : A = 4 * B) 
  (h2 : J = B + 8) 
  (h3 : A = J) 
: B = 8 / 3 := 
by 
  sorry

end brendas_age_l1411_141107


namespace number_of_students_in_club_l1411_141102

variable (y : ℕ) -- Number of girls

def total_stickers_given (y : ℕ) : ℕ := y * y + (y + 3) * (y + 3)

theorem number_of_students_in_club :
  (total_stickers_given y = 640) → (2 * y + 3 = 35) := 
by
  intro h1
  sorry

end number_of_students_in_club_l1411_141102


namespace inequality_proof_l1411_141192

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : x + y ≤ (y^2 / x) + (x^2 / y) :=
sorry

end inequality_proof_l1411_141192


namespace profit_percentage_for_unspecified_weight_l1411_141147

-- Definitions to align with the conditions
def total_sugar : ℝ := 1000
def profit_400_kg : ℝ := 0.08
def unspecified_weight : ℝ := 600
def overall_profit : ℝ := 0.14
def total_400_kg := total_sugar - unspecified_weight
def total_overall_profit := total_sugar * overall_profit
def total_400_kg_profit := total_400_kg * profit_400_kg
def total_unspecified_weight_profit (profit_percentage : ℝ) := unspecified_weight * profit_percentage

-- The theorem statement
theorem profit_percentage_for_unspecified_weight : 
  ∃ (profit_percentage : ℝ), total_400_kg_profit + total_unspecified_weight_profit profit_percentage = total_overall_profit ∧ profit_percentage = 0.18 := by
  sorry

end profit_percentage_for_unspecified_weight_l1411_141147


namespace medium_stores_in_sample_l1411_141181

theorem medium_stores_in_sample :
  let total_stores := 300
  let large_stores := 30
  let medium_stores := 75
  let small_stores := 195
  let sample_size := 20
  sample_size * (medium_stores/total_stores) = 5 :=
by
  sorry

end medium_stores_in_sample_l1411_141181


namespace cypress_tree_price_l1411_141170

def amount_per_cypress_tree (C : ℕ) : Prop :=
  let cabin_price := 129000
  let cash := 150
  let cypress_count := 20
  let pine_count := 600
  let maple_count := 24
  let pine_price := 200
  let maple_price := 300
  let leftover_cash := 350
  let total_amount_raised := cabin_price - cash + leftover_cash
  let total_pine_maple := (pine_count * pine_price) + (maple_count * maple_price)
  let total_cypress := total_amount_raised - total_pine_maple
  let cypress_sale_price := total_cypress / cypress_count
  cypress_sale_price = C

theorem cypress_tree_price : amount_per_cypress_tree 100 :=
by {
  -- Proof skipped
  sorry
}

end cypress_tree_price_l1411_141170


namespace cricket_average_increase_l1411_141162

theorem cricket_average_increase (runs_mean : ℕ) (innings : ℕ) (runs : ℕ) (new_runs : ℕ) (x : ℕ) :
  runs_mean = 35 → innings = 10 → runs = 79 → (total_runs : ℕ) = runs_mean * innings → 
  (new_total : ℕ) = total_runs + runs → (new_mean : ℕ) = new_total / (innings + 1) ∧ new_mean = runs_mean + x → x = 4 :=
by
  sorry

end cricket_average_increase_l1411_141162


namespace sum_of_ages_l1411_141178

variables (K T1 T2 : ℕ)

theorem sum_of_ages (h1 : K * T1 * T2 = 72) (h2 : T1 = T2) (h3 : T1 < K) : K + T1 + T2 = 14 :=
sorry

end sum_of_ages_l1411_141178


namespace fraction_to_decimal_l1411_141184

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 :=
by
  sorry

end fraction_to_decimal_l1411_141184


namespace greatest_possible_large_chips_l1411_141117

theorem greatest_possible_large_chips :
  ∃ l s : ℕ, ∃ p : ℕ, s + l = 61 ∧ s = l + p ∧ Nat.Prime p ∧ l = 29 :=
sorry

end greatest_possible_large_chips_l1411_141117


namespace cary_mow_weekends_l1411_141126

theorem cary_mow_weekends :
  let cost_shoes := 120
  let saved_amount := 30
  let earn_per_lawn := 5
  let lawns_per_weekend := 3
  let remaining_amount := cost_shoes - saved_amount
  let earn_per_weekend := lawns_per_weekend * earn_per_lawn
  remaining_amount / earn_per_weekend = 6 :=
by
  let cost_shoes := 120
  let saved_amount := 30
  let earn_per_lawn := 5
  let lawns_per_weekend := 3
  let remaining_amount := cost_shoes - saved_amount
  let earn_per_weekend := lawns_per_weekend * earn_per_lawn
  have needed_weekends : remaining_amount / earn_per_weekend = 6 :=
    sorry
  exact needed_weekends

end cary_mow_weekends_l1411_141126


namespace find_prices_find_min_money_spent_l1411_141150

-- Define the prices of volleyball and soccer ball
def prices (pv ps : ℕ) : Prop :=
  pv + 20 = ps ∧ 500 / ps = 400 / pv

-- Define the quantity constraint
def quantity_constraint (a : ℕ) : Prop :=
  a ≥ 25 ∧ a < 50

-- Define the minimum amount spent problem
def min_money_spent (a : ℕ) (pv ps : ℕ) : Prop :=
  prices pv ps → quantity_constraint a → 100 * a + 80 * (50 - a) = 4500

-- Prove the price of each volleyball and soccer ball
theorem find_prices : ∃ (pv ps : ℕ), prices pv ps ∧ pv = 80 ∧ ps = 100 :=
by {sorry}

-- Prove the minimum amount of money spent
theorem find_min_money_spent : ∃ (a pv ps : ℕ), min_money_spent a pv ps :=
by {sorry}

end find_prices_find_min_money_spent_l1411_141150


namespace rohan_age_is_25_l1411_141139

-- Define the current age of Rohan
def rohan_current_age (x : ℕ) : Prop :=
  x + 15 = 4 * (x - 15)

-- The goal is to prove that Rohan's current age is 25 years old
theorem rohan_age_is_25 : ∃ x : ℕ, rohan_current_age x ∧ x = 25 :=
by
  existsi (25 : ℕ)
  -- Proof is omitted since this is a statement only
  sorry

end rohan_age_is_25_l1411_141139


namespace pages_after_break_l1411_141119

-- Formalize the conditions
def total_pages : ℕ := 30
def break_percentage : ℝ := 0.70

-- Define the proof problem
theorem pages_after_break : 
  let pages_read_before_break := (break_percentage * total_pages)
  let pages_remaining := total_pages - pages_read_before_break
  pages_remaining = 9 :=
by
  sorry

end pages_after_break_l1411_141119


namespace comparison_of_a_b_c_l1411_141135

theorem comparison_of_a_b_c : 
  let a := (1/3)^(2/5)
  let b := 2^(4/3)
  let c := Real.logb 2 (1/3)
  c < a ∧ a < b :=
by
  sorry

end comparison_of_a_b_c_l1411_141135


namespace number_of_women_l1411_141180

theorem number_of_women
    (n : ℕ) -- number of men
    (d_m : ℕ) -- number of dances each man had
    (d_w : ℕ) -- number of dances each woman had
    (total_men : n = 15) -- there are 15 men
    (each_man_dances : d_m = 4) -- each man danced with 4 women
    (each_woman_dances : d_w = 3) -- each woman danced with 3 men
    (total_dances : n * d_m = w * d_w): -- total dances are the same when counted from both sides
  w = 20 := sorry -- There should be exactly 20 women.


end number_of_women_l1411_141180


namespace smallest_nonprime_with_large_prime_factors_l1411_141140

/-- 
The smallest nonprime integer greater than 1 with no prime factor less than 15
falls in the range 260 < m ≤ 270.
-/
theorem smallest_nonprime_with_large_prime_factors :
  ∃ m : ℕ, 2 < m ∧ ¬ Nat.Prime m ∧ (∀ p : ℕ, Nat.Prime p → p ∣ m → 15 ≤ p) ∧ 260 < m ∧ m ≤ 270 :=
by
  sorry

end smallest_nonprime_with_large_prime_factors_l1411_141140


namespace gain_percent_is_150_l1411_141110

theorem gain_percent_is_150 (CP SP : ℝ) (hCP : CP = 10) (hSP : SP = 25) : (SP - CP) / CP * 100 = 150 := by
  sorry

end gain_percent_is_150_l1411_141110


namespace paint_cans_needed_l1411_141141

theorem paint_cans_needed
    (num_bedrooms : ℕ)
    (num_other_rooms : ℕ)
    (total_rooms : ℕ)
    (gallons_per_room : ℕ)
    (color_paint_cans_per_gallon : ℕ)
    (white_paint_cans_per_gallon : ℕ)
    (total_paint_needed : ℕ)
    (color_paint_cans_needed : ℕ)
    (white_paint_cans_needed : ℕ)
    (total_paint_cans : ℕ)
    (h1 : num_bedrooms = 3)
    (h2 : num_other_rooms = 2 * num_bedrooms)
    (h3 : total_rooms = num_bedrooms + num_other_rooms)
    (h4 : gallons_per_room = 2)
    (h5 : total_paint_needed = total_rooms * gallons_per_room)
    (h6 : color_paint_cans_per_gallon = 1)
    (h7 : white_paint_cans_per_gallon = 3)
    (h8 : color_paint_cans_needed = num_bedrooms * gallons_per_room * color_paint_cans_per_gallon)
    (h9 : white_paint_cans_needed = (num_other_rooms * gallons_per_room) / white_paint_cans_per_gallon)
    (h10 : total_paint_cans = color_paint_cans_needed + white_paint_cans_needed) :
    total_paint_cans = 10 :=
by sorry

end paint_cans_needed_l1411_141141


namespace triple_complement_angle_l1411_141159

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l1411_141159


namespace largest_4_digit_divisible_by_88_and_prime_gt_100_l1411_141187

noncomputable def is_4_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

noncomputable def is_divisible_by (n d : ℕ) : Prop :=
  d ∣ n

noncomputable def is_prime (p : ℕ) : Prop :=
  Nat.Prime p

noncomputable def lcm (a b : ℕ) : ℕ :=
  Nat.lcm a b

theorem largest_4_digit_divisible_by_88_and_prime_gt_100 (p : ℕ) (hp : is_prime p) (h1 : 100 < p):
  ∃ n, is_4_digit n ∧ is_divisible_by n 88 ∧ is_divisible_by n p ∧
       (∀ m, is_4_digit m ∧ is_divisible_by m 88 ∧ is_divisible_by m p → m ≤ n) :=
sorry

end largest_4_digit_divisible_by_88_and_prime_gt_100_l1411_141187


namespace speed_of_train_l1411_141156

def distance : ℝ := 80
def time : ℝ := 6
def expected_speed : ℝ := 13.33

theorem speed_of_train : distance / time = expected_speed :=
by
  sorry

end speed_of_train_l1411_141156


namespace sum_of_ages_l1411_141112

variable (J L : ℝ)
variable (h1 : J = L + 8)
variable (h2 : J + 10 = 5 * (L - 5))

theorem sum_of_ages (J L : ℝ) (h1 : J = L + 8) (h2 : J + 10 = 5 * (L - 5)) : J + L = 29.5 := by
  sorry

end sum_of_ages_l1411_141112


namespace school_pupils_l1411_141105

def girls : ℕ := 868
def difference : ℕ := 281
def boys (g b : ℕ) : Prop := g = b + difference
def total_pupils (g b t : ℕ) : Prop := t = g + b

theorem school_pupils : 
  ∃ b t, boys girls b ∧ total_pupils girls b t ∧ t = 1455 :=
by
  sorry

end school_pupils_l1411_141105


namespace profit_value_l1411_141188

variable (P : ℝ) -- Total profit made by the business in that year.
variable (MaryInvestment : ℝ) -- Mary's investment
variable (MikeInvestment : ℝ) -- Mike's investment
variable (MaryExtra : ℝ) -- Extra money received by Mary

-- Conditions
axiom mary_investment : MaryInvestment = 900
axiom mike_investment : MikeInvestment = 100
axiom mary_received_more : MaryExtra = 1600
axiom profit_shared_equally : (P / 3) / 2 + (MaryInvestment / (MaryInvestment + MikeInvestment)) * (2 * P / 3) 
                           = MikeInvestment / (MaryInvestment + MikeInvestment) * (2 * P / 3) + MaryExtra

-- Statement
theorem profit_value : P = 4000 :=
by
  sorry

end profit_value_l1411_141188


namespace min_number_of_students_l1411_141133

theorem min_number_of_students 
  (n : ℕ)
  (h1 : 25 ≡ 99 [MOD n])
  (h2 : 8 ≡ 119 [MOD n]) : 
  n = 37 :=
by sorry

end min_number_of_students_l1411_141133


namespace find_gain_percent_l1411_141155

-- Definitions based on the conditions
def CP : ℕ := 900
def SP : ℕ := 1170

-- Calculation of gain
def Gain := SP - CP

-- Calculation of gain percent
def GainPercent := (Gain * 100) / CP

-- The theorem to prove the gain percent is 30%
theorem find_gain_percent : GainPercent = 30 := 
by
  sorry -- Proof to be filled in.

end find_gain_percent_l1411_141155


namespace no_three_digit_whole_number_solves_log_eq_l1411_141163

noncomputable def log_function (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem no_three_digit_whole_number_solves_log_eq :
  ¬ ∃ n : ℤ, (100 ≤ n ∧ n < 1000) ∧ log_function (3 * n) 10 + log_function (7 * n) 10 = 1 :=
by
  sorry

end no_three_digit_whole_number_solves_log_eq_l1411_141163


namespace miranda_can_stuff_10_pillows_l1411_141145

def feathers_needed_per_pillow : ℕ := 2
def goose_feathers_per_pound : ℕ := 300
def duck_feathers_per_pound : ℕ := 500
def goose_total_feathers : ℕ := 3600
def duck_total_feathers : ℕ := 4000

theorem miranda_can_stuff_10_pillows :
  (goose_total_feathers / goose_feathers_per_pound + duck_total_feathers / duck_feathers_per_pound) / feathers_needed_per_pillow = 10 :=
by
  sorry

end miranda_can_stuff_10_pillows_l1411_141145


namespace determinant_A_l1411_141179

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, -1, 5], ![0, 4, -2], ![3, 0, 1]]

theorem determinant_A : Matrix.det A = -46 := by
  sorry

end determinant_A_l1411_141179


namespace problems_per_page_l1411_141158

theorem problems_per_page (total_problems finished_problems pages_left problems_per_page : ℕ)
  (h1 : total_problems = 40)
  (h2 : finished_problems = 26)
  (h3 : pages_left = 2)
  (h4 : total_problems - finished_problems = pages_left * problems_per_page) :
  problems_per_page = 7 :=
by
  sorry

end problems_per_page_l1411_141158


namespace total_students_l1411_141198

theorem total_students (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) (hshake : (2 * m * n - m - n) = 252) : m * n = 72 :=
  sorry

end total_students_l1411_141198


namespace perfect_square_expression_l1411_141136

theorem perfect_square_expression : 
    ∀ x : ℝ, (11.98 * 11.98 + 11.98 * x + 0.02 * 0.02 = (11.98 + 0.02)^2) → (x = 0.4792) :=
by
  intros x h
  -- sorry placeholder for the proof
  sorry

end perfect_square_expression_l1411_141136


namespace problem_l1411_141106

open BigOperators

variables {p q : ℝ} {n : ℕ}

theorem problem 
  (h : p + q = 1) : 
  ∑ r in Finset.range (n / 2 + 1), (-1 : ℝ) ^ r * (Nat.choose (n - r) r) * p^r * q^r = (p ^ (n + 1) - q ^ (n + 1)) / (p - q) :=
by
  sorry

end problem_l1411_141106


namespace intersection_value_of_a_l1411_141189

theorem intersection_value_of_a (a : ℝ) (A B : Set ℝ) 
  (hA : A = {0, 1, 3})
  (hB : B = {a + 1, a^2 + 2})
  (h_inter : A ∩ B = {1}) : 
  a = 0 :=
by
  sorry

end intersection_value_of_a_l1411_141189


namespace expand_expression_l1411_141164

variable (x y z : ℕ)

theorem expand_expression (x y z: ℕ) : 
  (x + 10) * (3 * y + 5 * z + 15) = 3 * x * y + 5 * x * z + 15 * x + 30 * y + 50 * z + 150 :=
by
  sorry

end expand_expression_l1411_141164


namespace positive_integer_triples_satisfying_conditions_l1411_141161

theorem positive_integer_triples_satisfying_conditions :
  ∀ (a b c : ℕ), a^2 + b^2 + c^2 = 2005 ∧ a ≤ b ∧ b ≤ c →
  (a, b, c) = (23, 24, 30) ∨
  (a, b, c) = (12, 30, 31) ∨
  (a, b, c) = (9, 30, 32) ∨
  (a, b, c) = (4, 30, 33) ∨
  (a, b, c) = (15, 22, 36) ∨
  (a, b, c) = (9, 18, 40) ∨
  (a, b, c) = (4, 15, 42) :=
sorry

end positive_integer_triples_satisfying_conditions_l1411_141161


namespace arithmetic_seq_sum_is_110_l1411_141149

noncomputable def S₁₀ (a_1 : ℝ) : ℝ :=
  10 / 2 * (2 * a_1 + 9 * (-2))

theorem arithmetic_seq_sum_is_110 (a1 a3 a7 a9 : ℝ) 
  (h_diff3 : a3 = a1 - 4)
  (h_diff7 : a7 = a1 - 12)
  (h_diff9 : a9 = a1 - 16)
  (h_geom : (a1 - 12) ^ 2 = (a1 - 4) * (a1 - 16)) :
  S₁₀ a1 = 110 :=
by
  sorry

end arithmetic_seq_sum_is_110_l1411_141149


namespace peaches_total_l1411_141196

theorem peaches_total (n P : ℕ) (h1 : P - 6 * n = 57) (h2 : P = 9 * (n - 6) + 3) : P = 273 :=
by
  sorry

end peaches_total_l1411_141196


namespace camila_bikes_more_l1411_141199

-- Definitions based on conditions
def camila_speed : ℝ := 15
def daniel_speed_initial : ℝ := 15
def daniel_speed_after_3hours : ℝ := 10
def biking_time : ℝ := 6
def time_before_decrease : ℝ := 3
def time_after_decrease : ℝ := biking_time - time_before_decrease

def distance_camila := camila_speed * biking_time
def distance_daniel := (daniel_speed_initial * time_before_decrease) + (daniel_speed_after_3hours * time_after_decrease)

-- The statement to prove: Camila has biked 15 more miles than Daniel
theorem camila_bikes_more : distance_camila - distance_daniel = 15 := 
by
  sorry

end camila_bikes_more_l1411_141199


namespace candy_system_of_equations_l1411_141146

-- Definitions based on conditions
def candy_weight := 100
def candy_price1 := 36
def candy_price2 := 20
def mixed_candy_price := 28

theorem candy_system_of_equations (x y: ℝ):
  (x + y = candy_weight) ∧ (candy_price1 * x + candy_price2 * y = mixed_candy_price * candy_weight) :=
sorry

end candy_system_of_equations_l1411_141146


namespace vertical_asymptote_l1411_141172

theorem vertical_asymptote (x : ℝ) : 4 * x - 9 = 0 → x = 9 / 4 := by
  sorry

end vertical_asymptote_l1411_141172


namespace power_function_inequality_l1411_141191

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^a

theorem power_function_inequality (x : ℝ) (a : ℝ) : (x > 1) → (f x a < x) ↔ (a < 1) :=
by
  sorry

end power_function_inequality_l1411_141191


namespace rationalize_denominator_l1411_141125

theorem rationalize_denominator : (1 / (Real.sqrt 3 + 1)) = ((Real.sqrt 3 - 1) / 2) :=
by
  sorry

end rationalize_denominator_l1411_141125


namespace sum_of_angles_of_inscribed_quadrilateral_l1411_141143

/--
Given a quadrilateral EFGH inscribed in a circle, and the measures of ∠EGH = 50° and ∠GFE = 70°,
then the sum of the angles ∠EFG + ∠EHG is 60°.
-/
theorem sum_of_angles_of_inscribed_quadrilateral
  (E F G H : Type)
  (circumscribed : True) -- This is just a place holder for the circle condition
  (angle_EGH : ℝ) (angle_GFE : ℝ)
  (h1 : angle_EGH = 50)
  (h2 : angle_GFE = 70) :
  ∃ (angle_EFG angle_EHG : ℝ), angle_EFG + angle_EHG = 60 := sorry

end sum_of_angles_of_inscribed_quadrilateral_l1411_141143


namespace complement_union_A_B_l1411_141185

def is_element_of_set_A (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 1
def is_element_of_set_B (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 2
def is_element_of_complement_U (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k

theorem complement_union_A_B :
  {x : ℤ | ¬ (is_element_of_set_A x ∨ is_element_of_set_B x)} = {x : ℤ | is_element_of_complement_U x} :=
by
  sorry

end complement_union_A_B_l1411_141185


namespace sum_of_numbers_on_cards_l1411_141115

-- Define the natural numbers condition
variables {a b c d e f g h : ℕ}

-- The theorem statement
theorem sum_of_numbers_on_cards (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) :
  a + b + c + d + e + f + g + h = 21 :=
sorry

end sum_of_numbers_on_cards_l1411_141115


namespace power_mod_eq_nine_l1411_141131

theorem power_mod_eq_nine :
  ∃ n : ℕ, 13^6 ≡ n [MOD 11] ∧ 0 ≤ n ∧ n < 11 ∧ n = 9 :=
by
  sorry

end power_mod_eq_nine_l1411_141131


namespace min_increase_velocity_correct_l1411_141124

noncomputable def min_increase_velocity (V_A V_B V_C V_D : ℝ) (dist_AC dist_CD : ℝ) : ℝ :=
  let t_AC := dist_AC / (V_A + V_C)
  let t_AB := 30 / (V_A - V_B)
  let t_AD := (dist_AC + dist_CD) / (V_A + V_D)
  let new_velocity_A := (dist_AC + dist_CD) / t_AC - V_D
  new_velocity_A - V_A

theorem min_increase_velocity_correct :
  min_increase_velocity 80 50 70 60 300 400 = 210 :=
by
  sorry

end min_increase_velocity_correct_l1411_141124


namespace isosceles_triangle_base_length_l1411_141113

theorem isosceles_triangle_base_length
  (b : ℕ)
  (congruent_side : ℕ)
  (perimeter : ℕ)
  (h1 : congruent_side = 8)
  (h2 : perimeter = 25)
  (h3 : 2 * congruent_side + b = perimeter) :
  b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l1411_141113


namespace number_of_commonly_used_structures_is_3_l1411_141130

def commonly_used_algorithm_structures : Nat := 3
theorem number_of_commonly_used_structures_is_3 
  (structures : Nat)
  (h : structures = 1 ∨ structures = 2 ∨ structures = 3 ∨ structures = 4) :
  commonly_used_algorithm_structures = 3 :=
by
  -- Proof to be added
  sorry

end number_of_commonly_used_structures_is_3_l1411_141130


namespace fewer_seats_right_side_l1411_141129

theorem fewer_seats_right_side
  (left_seats : ℕ)
  (people_per_seat : ℕ)
  (back_seat_capacity : ℕ)
  (total_capacity : ℕ)
  (h1 : left_seats = 15)
  (h2 : people_per_seat = 3)
  (h3 : back_seat_capacity = 12)
  (h4 : total_capacity = 93)
  : left_seats - (total_capacity - (left_seats * people_per_seat + back_seat_capacity)) / people_per_seat = 3 :=
  by sorry

end fewer_seats_right_side_l1411_141129


namespace original_selling_price_l1411_141165

theorem original_selling_price (C : ℝ) (h : 1.60 * C = 2560) : 1.40 * C = 2240 :=
by
  sorry

end original_selling_price_l1411_141165


namespace cost_price_of_toy_l1411_141194

theorem cost_price_of_toy (x : ℝ) (selling_price_per_toy : ℝ) (gain : ℝ) 
  (sale_price : ℝ) (number_of_toys : ℕ) (selling_total : ℝ) (gain_condition : ℝ) :
  (selling_total = number_of_toys * selling_price_per_toy) →
  (selling_price_per_toy = x + gain) →
  (gain = gain_condition / number_of_toys) → 
  (gain_condition = 3 * x) →
  selling_total = 25200 → number_of_toys = 18 → x = 1200 :=
by
  sorry

end cost_price_of_toy_l1411_141194


namespace sin_cos_sixth_l1411_141190

theorem sin_cos_sixth (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 3) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 11 / 12 :=
sorry

end sin_cos_sixth_l1411_141190


namespace votes_cast_l1411_141121

theorem votes_cast (V : ℝ) (h1 : ∃ Vc, Vc = 0.25 * V) (h2 : ∃ Vr, Vr = 0.25 * V + 4000) : V = 8000 :=
sorry

end votes_cast_l1411_141121


namespace fraction_is_seventh_l1411_141176

-- Definition of the condition on x being greater by a certain percentage
def x_greater := 1125.0000000000002 / 100

-- Definition of x in terms of the condition
def x := (4 / 7) * (1 + x_greater)

-- Definition of the fraction f
def f := 1 / x

-- Lean theorem statement to prove the fraction is 1/7
theorem fraction_is_seventh (x_greater: ℝ) : (1 / ((4 / 7) * (1 + x_greater))) = 1 / 7 :=
by
  sorry

end fraction_is_seventh_l1411_141176


namespace combined_class_average_score_l1411_141182

theorem combined_class_average_score
  (avg_A : ℕ := 65) (avg_B : ℕ := 90) (avg_C : ℕ := 77)
  (ratio_A : ℕ := 4) (ratio_B : ℕ := 6) (ratio_C : ℕ := 5) :
  ((avg_A * ratio_A + avg_B * ratio_B + avg_C * ratio_C) / (ratio_A + ratio_B + ratio_C) = 79) :=
by 
  sorry

end combined_class_average_score_l1411_141182


namespace no_snuggly_numbers_l1411_141103

def isSnuggly (n : Nat) : Prop :=
  ∃ (a b : Nat), 
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    n = 10 * a + b ∧ 
    n = a + b^3 + 5

theorem no_snuggly_numbers : 
  ¬ ∃ n : Nat, 10 ≤ n ∧ n < 100 ∧ isSnuggly n :=
by
  sorry

end no_snuggly_numbers_l1411_141103


namespace simplify_and_evaluate_l1411_141152

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := 2 - Real.sqrt 2

theorem simplify_and_evaluate : 
  let expr := (a / (a^2 - b^2) - 1 / (a + b)) / (b / (b - a))
  expr = -1 / 2 := by
  sorry

end simplify_and_evaluate_l1411_141152


namespace polar_curve_symmetry_l1411_141173

theorem polar_curve_symmetry :
  ∀ (ρ θ : ℝ), ρ = 4 * Real.sin (θ - π / 3) → 
  ∃ k : ℤ, θ = 5 * π / 6 + k * π :=
sorry

end polar_curve_symmetry_l1411_141173


namespace helen_oranges_l1411_141111

def initial_oranges := 9
def oranges_from_ann := 29
def oranges_taken_away := 14

def final_oranges (initial : Nat) (add : Nat) (taken : Nat) : Nat :=
  initial + add - taken

theorem helen_oranges :
  final_oranges initial_oranges oranges_from_ann oranges_taken_away = 24 :=
by
  sorry

end helen_oranges_l1411_141111


namespace solution_set_of_inequality_l1411_141175

open Set

theorem solution_set_of_inequality :
  {x : ℝ | (x ≠ -2) ∧ (x ≠ -8) ∧ (2 / (x + 2) + 4 / (x + 8) ≥ 4 / 5)} =
  {x : ℝ | (-8 < x ∧ x < -2) ∨ (-2 < x ∧ x ≤ 4)} :=
by
  sorry

end solution_set_of_inequality_l1411_141175


namespace range_of_a_l1411_141171

theorem range_of_a (a : ℝ) (A : Set ℝ) (hA : ∀ x, x ∈ A ↔ a / (x - 1) < 1) (h_not_in : 2 ∉ A) : a ≥ 1 := 
sorry

end range_of_a_l1411_141171


namespace cupric_cyanide_formed_l1411_141174

-- Definition of the problem
def formonitrile : ℕ := 6
def copper_sulfate : ℕ := 3
def sulfuric_acid : ℕ := 3

-- Stoichiometry from the balanced equation
def stoichiometry (hcn mol_multiplier: ℕ): ℕ := 
  (hcn / mol_multiplier)

theorem cupric_cyanide_formed :
  stoichiometry formonitrile 2 = 3 := 
sorry

end cupric_cyanide_formed_l1411_141174


namespace convert_50_to_base_3_l1411_141142

-- Define a function to convert decimal to ternary (base-3)
def convert_to_ternary (n : ℕ) : ℕ := sorry

-- Main theorem statement
theorem convert_50_to_base_3 : convert_to_ternary 50 = 1212 :=
sorry

end convert_50_to_base_3_l1411_141142


namespace least_overlap_coffee_tea_l1411_141116

open BigOperators

-- Define the percentages in a way that's compatible in Lean
def percentage (n : ℕ) := n / 100

noncomputable def C := percentage 75
noncomputable def T := percentage 80
noncomputable def B := percentage 55

-- The theorem statement
theorem least_overlap_coffee_tea : C + T - 1 = B := sorry

end least_overlap_coffee_tea_l1411_141116


namespace find_a16_l1411_141144

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 / 2 ∧ ∀ n ≥ 1, a (n + 1) = 1 - 1 / a n

theorem find_a16 (a : ℕ → ℝ) (h : seq a) : a 16 = 1 / 2 :=
sorry

end find_a16_l1411_141144


namespace sandwiches_left_l1411_141114

theorem sandwiches_left (S G K L : ℕ) (h1 : S = 20) (h2 : G = 4) (h3 : K = 2 * G) (h4 : L = S - G - K) : L = 8 :=
sorry

end sandwiches_left_l1411_141114


namespace pq_r_sum_l1411_141168

theorem pq_r_sum (p q r : ℝ) (h1 : p^3 - 18 * p^2 + 27 * p - 72 = 0) 
                 (h2 : 27 * q^3 - 243 * q^2 + 729 * q - 972 = 0)
                 (h3 : 3 * r = 9) : p + q + r = 18 :=
by
  sorry

end pq_r_sum_l1411_141168


namespace find_speed_of_stream_l1411_141122

-- Define the given conditions
def boat_speed_still_water : ℝ := 14
def distance_downstream : ℝ := 72
def time_downstream : ℝ := 3.6

-- Define the speed of the stream (to be proven)
def speed_of_stream : ℝ := 6

-- The statement of the problem
theorem find_speed_of_stream 
  (h1 : boat_speed_still_water = 14)
  (h2 : distance_downstream = 72)
  (h3 : time_downstream = 3.6)
  (speed_of_stream_eq : boat_speed_still_water + speed_of_stream = distance_downstream / time_downstream) :
  speed_of_stream = 6 := 
by 
  sorry

end find_speed_of_stream_l1411_141122


namespace line_always_passes_fixed_point_l1411_141186

theorem line_always_passes_fixed_point:
  ∀ a x y, x = 5 → y = -3 → (a * x + (2 * a - 1) * y + a - 3 = 0) :=
by
  intros a x y h1 h2
  rw [h1, h2]
  sorry

end line_always_passes_fixed_point_l1411_141186


namespace discount_each_book_l1411_141104

-- Definition of conditions
def original_price : ℝ := 5
def num_books : ℕ := 10
def total_paid : ℝ := 45

-- Theorem statement to prove the discount
theorem discount_each_book (d : ℝ) 
  (h1 : original_price * (num_books : ℝ) - d * (num_books : ℝ) = total_paid) : 
  d = 0.5 := 
sorry

end discount_each_book_l1411_141104


namespace intersection_A_B_l1411_141138

def A : Set ℝ := { x | |x| > 1 }
def B : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | 1 < x ∧ x < 2 } :=
sorry

end intersection_A_B_l1411_141138


namespace evan_ivan_kara_total_weight_eq_432_l1411_141123

variable (weight_evan : ℕ) (weight_ivan : ℕ) (weight_kara_cat : ℕ)

-- Conditions
def evans_dog_weight : Prop := weight_evan = 63
def ivans_dog_weight : Prop := weight_evan = 7 * weight_ivan
def karas_cat_weight : Prop := weight_kara_cat = 5 * (weight_evan + weight_ivan)

-- Mathematical equivalence
def total_weight : Prop := weight_evan + weight_ivan + weight_kara_cat = 432

theorem evan_ivan_kara_total_weight_eq_432 :
  evans_dog_weight weight_evan →
  ivans_dog_weight weight_evan weight_ivan →
  karas_cat_weight weight_evan weight_ivan weight_kara_cat →
  total_weight weight_evan weight_ivan weight_kara_cat :=
by
  intros h1 h2 h3
  sorry

end evan_ivan_kara_total_weight_eq_432_l1411_141123


namespace pedestrian_walking_time_in_interval_l1411_141120

noncomputable def bus_departure_interval : ℕ := 5  -- Condition 1: Buses depart every 5 minutes
noncomputable def buses_same_direction : ℕ := 11  -- Condition 2: 11 buses passed him going the same direction
noncomputable def buses_opposite_direction : ℕ := 13  -- Condition 3: 13 buses came from opposite direction
noncomputable def bus_speed_factor : ℕ := 8  -- Condition 4: Bus speed is 8 times the pedestrian's speed
noncomputable def min_walking_time : ℚ := 57 + 1 / 7 -- Correct Answer: Minimum walking time
noncomputable def max_walking_time : ℚ := 62 + 2 / 9 -- Correct Answer: Maximum walking time

theorem pedestrian_walking_time_in_interval (t : ℚ)
  (h1 : bus_departure_interval = 5)
  (h2 : buses_same_direction = 11)
  (h3 : buses_opposite_direction = 13)
  (h4 : bus_speed_factor = 8) :
  min_walking_time ≤ t ∧ t ≤ max_walking_time :=
sorry

end pedestrian_walking_time_in_interval_l1411_141120


namespace log_bounds_sum_l1411_141128

theorem log_bounds_sum : (∀ a b : ℕ, a = 18 ∧ b = 19 → 18 < Real.log 537800 / Real.log 2 ∧ Real.log 537800 / Real.log 2 < 19 → a + b = 37) := 
sorry

end log_bounds_sum_l1411_141128


namespace tv_price_reduction_l1411_141108

theorem tv_price_reduction (x : ℝ) (Q : ℝ) (P : ℝ) (h1 : Q > 0) (h2 : P > 0) (h3 : P*(1 - x/100) * 1.85 * Q = 1.665 * P * Q) : x = 10 :=
by 
  sorry

end tv_price_reduction_l1411_141108


namespace sqrt_inequality_l1411_141154

theorem sqrt_inequality (x y z : ℝ) (hx : 1 < x) (hy : 1 < y) (hz : 1 < z)
  (h : 1 / x + 1 / y + 1 / z = 2) : 
  Real.sqrt (x + y + z) ≥ Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) := 
by
  sorry

end sqrt_inequality_l1411_141154


namespace vector_magnitude_l1411_141166

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude : 
  let AB := (-1, 2)
  let BC := (x, -5)
  let AC := (AB.1 + BC.1, AB.2 + BC.2)
  dot_product AB BC = -7 → magnitude AC = 5 :=
by sorry

end vector_magnitude_l1411_141166
