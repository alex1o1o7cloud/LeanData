import Mathlib

namespace diamond_value_l2186_218645

def diamond (a b : Int) : Int :=
  a * b^2 - b + 1

theorem diamond_value : diamond (-1) 6 = -41 := by
  sorry

end diamond_value_l2186_218645


namespace units_digit_base8_l2186_218632

theorem units_digit_base8 (a b : ℕ) (h_a : a = 123) (h_b : b = 57) :
  let product := a * b
  let units_digit := product % 8
  units_digit = 7 := by
  sorry

end units_digit_base8_l2186_218632


namespace money_weed_eating_l2186_218671

-- Define the amounts and conditions
def money_mowing : ℕ := 68
def money_per_week : ℕ := 9
def weeks : ℕ := 9
def total_money : ℕ := money_per_week * weeks

-- Define the proof that the money made weed eating is 13 dollars
theorem money_weed_eating :
  total_money - money_mowing = 13 := sorry

end money_weed_eating_l2186_218671


namespace smallest_sum_of_squares_l2186_218642

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 91) : x^2 + y^2 ≥ 109 :=
sorry

end smallest_sum_of_squares_l2186_218642


namespace cats_not_eating_cheese_or_tuna_l2186_218687

-- Define the given conditions
variables (n C T B : ℕ)

-- State the problem in Lean
theorem cats_not_eating_cheese_or_tuna 
  (h_n : n = 100)  
  (h_C : C = 25)  
  (h_T : T = 70)  
  (h_B : B = 15)
  : n - (C - B + T - B + B) = 20 := 
by {
  -- Insert proof here
  sorry
}

end cats_not_eating_cheese_or_tuna_l2186_218687


namespace parity_of_expression_l2186_218601

theorem parity_of_expression (a b c : ℕ) (h_apos : 0 < a) (h_aodd : a % 2 = 1) (h_beven : b % 2 = 0) :
  (3^a + (b+1)^2 * c) % 2 = if c % 2 = 0 then 1 else 0 :=
sorry

end parity_of_expression_l2186_218601


namespace find_number_l2186_218666

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
by
  -- Proof goes here
  sorry

end find_number_l2186_218666


namespace eval_expression_at_neg3_l2186_218606

def evaluate_expression (x : ℤ) : ℚ :=
  (5 + x * (5 + x) - 4 ^ 2 : ℤ) / (x - 4 + x ^ 3 : ℤ)

theorem eval_expression_at_neg3 :
  evaluate_expression (-3) = -17 / 20 := by
  sorry

end eval_expression_at_neg3_l2186_218606


namespace glass_bowls_sold_l2186_218694

theorem glass_bowls_sold
  (BowlsBought : ℕ) (CostPricePerBowl SellingPricePerBowl : ℝ) (PercentageGain : ℝ)
  (CostPrice := BowlsBought * CostPricePerBowl)
  (SellingPrice : ℝ := (102 : ℝ) * SellingPricePerBowl)
  (gain := (SellingPrice - CostPrice) / CostPrice * 100) :
  PercentageGain = 8.050847457627118 →
  BowlsBought = 118 →
  CostPricePerBowl = 12 →
  SellingPricePerBowl = 15 →
  PercentageGain = gain →
  102 = 102 := by
  intro h1 h2 h3 h4 h5
  sorry

end glass_bowls_sold_l2186_218694


namespace family_gathering_total_people_l2186_218682

theorem family_gathering_total_people (P : ℕ) 
  (h1 : P / 2 = 10) : 
  P = 20 := by
  sorry

end family_gathering_total_people_l2186_218682


namespace range_of_a_l2186_218689

theorem range_of_a (a b c : ℝ) 
  (h1 : a^2 - b*c - 8*a + 7 = 0) 
  (h2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 :=
sorry

end range_of_a_l2186_218689


namespace race_problem_equivalent_l2186_218608

noncomputable def race_track_distance (D_paved D_dirt D_muddy : ℝ) : Prop :=
  let v1 := 100 -- speed on paved section in km/h
  let v2 := 70  -- speed on dirt section in km/h
  let v3 := 15  -- speed on muddy section in km/h
  let initial_distance := 0.5 -- initial distance in km (since 500 meters is 0.5 km)
  
  -- Time to cover paved section
  let t_white_paved := D_paved / v1
  let t_red_paved := (D_paved - initial_distance) / v1

  -- Times to cover dirt section
  let t_white_dirt := D_dirt / v2
  let t_red_dirt := D_dirt / v2 -- same time since both start at the same time on dirt

  -- Times to cover muddy section
  let t_white_muddy := D_muddy / v3
  let t_red_muddy := D_muddy / v3 -- same time since both start at the same time on mud

  -- Distances between cars on dirt and muddy sections
  ((t_white_paved - t_red_paved) * v2 = initial_distance) ∧ 
  ((t_white_paved - t_red_paved) * v3 = initial_distance)

-- Prove the distance between the cars when both are on the dirt and muddy sections is 500 meters
theorem race_problem_equivalent (D_paved D_dirt D_muddy : ℝ) : race_track_distance D_paved D_dirt D_muddy :=
by
  -- Insert proof here, for now we use sorry
  sorry

end race_problem_equivalent_l2186_218608


namespace sum_le_square_l2186_218629

theorem sum_le_square (m n : ℕ) (h: (m * n) % (m + n) = 0) : m + n ≤ n^2 :=
by sorry

end sum_le_square_l2186_218629


namespace imaginary_unit_div_l2186_218635

open Complex

theorem imaginary_unit_div (i : ℂ) (hi : i * i = -1) : (i / (1 + i) = (1 / 2) + (1 / 2) * i) :=
by
  sorry

end imaginary_unit_div_l2186_218635


namespace train_speed_l2186_218612

theorem train_speed (train_length : ℝ) (man_speed_kmph : ℝ) (passing_time : ℝ) : 
  train_length = 160 → man_speed_kmph = 6 →
  passing_time = 6 → (train_length / passing_time + man_speed_kmph * 1000 / 3600) * 3600 / 1000 = 90 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- further proof steps are omitted
  sorry

end train_speed_l2186_218612


namespace negation_of_existence_l2186_218618

theorem negation_of_existence (p : Prop) : 
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 ≤ x + 2)) ↔ (∀ x : ℝ, x > 0 → x^2 > x + 2) :=
by
  sorry

end negation_of_existence_l2186_218618


namespace sin_double_angle_l2186_218621

theorem sin_double_angle (theta : ℝ) 
  (h : Real.sin (theta + Real.pi / 4) = 2 / 5) :
  Real.sin (2 * theta) = -17 / 25 := by
  sorry

end sin_double_angle_l2186_218621


namespace add_to_make_divisible_by_23_l2186_218609

def least_addend_for_divisibility (n k : ℕ) : ℕ :=
  let remainder := n % k
  k - remainder

theorem add_to_make_divisible_by_23 : least_addend_for_divisibility 1053 23 = 5 :=
by
  sorry

end add_to_make_divisible_by_23_l2186_218609


namespace expected_accidents_no_overtime_l2186_218643

noncomputable def accidents_with_no_overtime_hours 
    (hours1 hours2 : ℕ) (accidents1 accidents2 : ℕ) : ℕ :=
  let slope := (accidents2 - accidents1) / (hours2 - hours1)
  let intercept := accidents1 - slope * hours1
  intercept

theorem expected_accidents_no_overtime : 
    accidents_with_no_overtime_hours 1000 400 8 5 = 3 :=
by
  sorry

end expected_accidents_no_overtime_l2186_218643


namespace average_income_l2186_218654

-- Lean statement to express the given mathematical problem
theorem average_income (A B C : ℝ) 
  (h1 : (A + B) / 2 = 4050)
  (h2 : (B + C) / 2 = 5250)
  (h3 : A = 3000) :
  (A + C) / 2 = 4200 :=
by
  sorry

end average_income_l2186_218654


namespace andy_tomatoes_left_l2186_218661

theorem andy_tomatoes_left :
  let plants := 50
  let tomatoes_per_plant := 15
  let total_tomatoes := plants * tomatoes_per_plant
  let tomatoes_dried := (2 / 3) * total_tomatoes
  let tomatoes_left_after_drying := total_tomatoes - tomatoes_dried
  let tomatoes_for_marinara := (1 / 2) * tomatoes_left_after_drying
  let tomatoes_left := tomatoes_left_after_drying - tomatoes_for_marinara
  tomatoes_left = 125 := sorry

end andy_tomatoes_left_l2186_218661


namespace values_of_a2_add_b2_l2186_218644

theorem values_of_a2_add_b2 (a b : ℝ) (h1 : a^3 - 3 * a * b^2 = 11) (h2 : b^3 - 3 * a^2 * b = 2) : a^2 + b^2 = 5 := 
by
  sorry

end values_of_a2_add_b2_l2186_218644


namespace value_of_y_l2186_218633

theorem value_of_y (x y : ℤ) (h1 : x + y = 270) (h2 : x - y = 200) : y = 35 :=
by
  sorry

end value_of_y_l2186_218633


namespace problem_l2186_218658

variable {x : ℝ}

theorem problem (h : x + 1/x = 5) : x^4 + 1/x^4 = 527 :=
by
  sorry

end problem_l2186_218658


namespace circle_symmetric_line_a_value_l2186_218634

theorem circle_symmetric_line_a_value :
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0) →
  (∀ x y : ℝ, (x, y) = (-1, 2)) →
  (∀ x y : ℝ, ax + y + 1 = 0) →
  a = 3 :=
by
  sorry

end circle_symmetric_line_a_value_l2186_218634


namespace degenerate_ellipse_value_c_l2186_218603

theorem degenerate_ellipse_value_c (c : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 14 * y + c = 0) ∧
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 14 * y + c = 0 → (x+1)^2 + (y-7)^2 = 0) ↔ c = 52 :=
by
  sorry

end degenerate_ellipse_value_c_l2186_218603


namespace find_m_l2186_218699

theorem find_m (m : ℝ) : (1 : ℝ) * (-4 : ℝ) + (2 : ℝ) * m = 0 → m = 2 :=
by
  sorry

end find_m_l2186_218699


namespace chloe_at_least_85_nickels_l2186_218626

-- Define the given values
def shoe_cost : ℝ := 45.50
def ten_dollars : ℝ := 10.0
def num_ten_dollar_bills : ℕ := 4
def quarter_value : ℝ := 0.25
def num_quarters : ℕ := 5
def nickel_value : ℝ := 0.05

-- Define the statement to be proved
theorem chloe_at_least_85_nickels (n : ℕ) 
  (H1 : shoe_cost = 45.50)
  (H2 : ten_dollars = 10.0)
  (H3 : num_ten_dollar_bills = 4)
  (H4 : quarter_value = 0.25)
  (H5 : num_quarters = 5)
  (H6 : nickel_value = 0.05) :
  4 * ten_dollars + 5 * quarter_value + n * nickel_value >= shoe_cost → n >= 85 :=
by {
  sorry
}

end chloe_at_least_85_nickels_l2186_218626


namespace intersection_setA_setB_l2186_218680

def setA := {x : ℝ | |x| < 1}
def setB := {x : ℝ | x^2 - 2 * x ≤ 0}

theorem intersection_setA_setB :
  {x : ℝ | 0 ≤ x ∧ x < 1} = setA ∩ setB :=
by
  sorry

end intersection_setA_setB_l2186_218680


namespace length_of_first_square_flag_l2186_218605

theorem length_of_first_square_flag
  (x : ℝ)
  (h1x : x * 5 + 10 * 7 + 5 * 5 = 15 * 9) : 
  x = 8 :=
by
  sorry

end length_of_first_square_flag_l2186_218605


namespace greatest_two_digit_product_12_l2186_218670

-- Definition of a two-digit whole number
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Definition of the digit product condition
def digits_product (n : ℕ) (p : ℕ) : Prop := ∃ (d1 d2 : ℕ), d1 * d2 = p ∧ n = 10 * d1 + d2

-- The main theorem stating the greatest two-digit number whose digits multiply to 12 is 62
theorem greatest_two_digit_product_12 : ∀ (n : ℕ), is_two_digit (n) → digits_product (n) 12 → n <= 62 :=
by {
    sorry -- Proof of the theorem
}

end greatest_two_digit_product_12_l2186_218670


namespace additional_telephone_lines_l2186_218668

theorem additional_telephone_lines :
  let lines_six_digits := 9 * 10^5
  let lines_seven_digits := 9 * 10^6
  let additional_lines := lines_seven_digits - lines_six_digits
  additional_lines = 81 * 10^5 :=
by
  sorry

end additional_telephone_lines_l2186_218668


namespace recreation_proof_l2186_218652

noncomputable def recreation_percentage_last_week (W : ℝ) (P : ℝ) :=
  let last_week_spent := (P/100) * W
  let this_week_wages := (70/100) * W
  let this_week_spent := (20/100) * this_week_wages
  this_week_spent = (70/100) * last_week_spent

theorem recreation_proof :
  ∀ (W : ℝ), recreation_percentage_last_week W 20 :=
by
  intros
  sorry

end recreation_proof_l2186_218652


namespace greatest_ln_2_l2186_218675

theorem greatest_ln_2 (x1 x2 x3 x4 : ℝ) (h1 : x1 = (Real.log 2) ^ 2) (h2 : x2 = Real.log (Real.log 2)) (h3 : x3 = Real.log (Real.sqrt 2)) (h4 : x4 = Real.log 2) 
  (h5 : Real.log 2 < 1) : 
  x4 = max x1 (max x2 (max x3 x4)) := by 
  sorry

end greatest_ln_2_l2186_218675


namespace least_number_divisible_l2186_218613

-- Define the numbers as given in the conditions
def given_number : ℕ := 3072
def divisor1 : ℕ := 57
def divisor2 : ℕ := 29
def least_number_to_add : ℕ := 234

-- Define the LCM
noncomputable def lcm_57_29 : ℕ := Nat.lcm divisor1 divisor2

-- Prove that adding least_number_to_add to given_number makes it divisible by both divisors
theorem least_number_divisible :
  (given_number + least_number_to_add) % divisor1 = 0 ∧ 
  (given_number + least_number_to_add) % divisor2 = 0 := 
by
  -- Proof should be provided here
  sorry

end least_number_divisible_l2186_218613


namespace range_of_b_over_a_l2186_218691

noncomputable def f (a b x : ℝ) : ℝ := (a * x - b / x - 2 * a) * Real.exp x

noncomputable def f' (a b x : ℝ) : ℝ := (b / x^2 + a * x - b / x - a) * Real.exp x

theorem range_of_b_over_a (a b : ℝ) (h₀ : a > 0) (h₁ : ∃ x : ℝ, 1 < x ∧ f a b x + f' a b x = 0) : 
  -1 < b / a := sorry

end range_of_b_over_a_l2186_218691


namespace selfish_subsets_equals_fibonacci_l2186_218628

noncomputable def fibonacci : ℕ → ℕ
| 0           => 0
| 1           => 1
| (n + 2)     => fibonacci (n + 1) + fibonacci n

noncomputable def selfish_subsets_count (n : ℕ) : ℕ := 
sorry -- This will be replaced with the correct recursive function

theorem selfish_subsets_equals_fibonacci (n : ℕ) : 
  selfish_subsets_count n = fibonacci n :=
sorry

end selfish_subsets_equals_fibonacci_l2186_218628


namespace average_of_w_x_z_l2186_218665

theorem average_of_w_x_z (w x z y a : ℝ) (h1 : 2 / w + 2 / x + 2 / z = 2 / y)
  (h2 : w * x * z = y) (h3 : w + x + z = a) : (w + x + z) / 3 = a / 3 :=
by sorry

end average_of_w_x_z_l2186_218665


namespace vlad_taller_than_sister_l2186_218697

-- Definitions based on the conditions
def vlad_feet : ℕ := 6
def vlad_inches : ℕ := 3
def sister_feet : ℕ := 2
def sister_inches : ℕ := 10
def inches_per_foot : ℕ := 12

-- Derived values for heights in inches
def vlad_height_in_inches : ℕ := (vlad_feet * inches_per_foot) + vlad_inches
def sister_height_in_inches : ℕ := (sister_feet * inches_per_foot) + sister_inches

-- Lean 4 statement for the proof problem
theorem vlad_taller_than_sister : vlad_height_in_inches - sister_height_in_inches = 41 := 
by 
  sorry

end vlad_taller_than_sister_l2186_218697


namespace two_pow_2014_mod_seven_l2186_218676

theorem two_pow_2014_mod_seven : 
  ∃ r : ℕ, 2 ^ 2014 ≡ r [MOD 7] → r = 2 :=
sorry

end two_pow_2014_mod_seven_l2186_218676


namespace betty_height_correct_l2186_218673

-- Definitions for the conditions
def dog_height : ℕ := 24
def carter_height : ℕ := 2 * dog_height
def betty_height_inches : ℕ := carter_height - 12
def betty_height_feet : ℕ := betty_height_inches / 12

-- Theorem that we need to prove
theorem betty_height_correct : betty_height_feet = 3 :=
by
  sorry

end betty_height_correct_l2186_218673


namespace gina_snake_mice_eaten_in_decade_l2186_218657

-- Define the constants and conditions
def weeks_per_year : ℕ := 52
def years_per_decade : ℕ := 10
def weeks_per_decade : ℕ := years_per_decade * weeks_per_year
def mouse_eating_period : ℕ := 4

-- The problem to prove
theorem gina_snake_mice_eaten_in_decade : (weeks_per_decade / mouse_eating_period) = 130 := 
by
  -- The proof would typically go here, but we skip it
  sorry

end gina_snake_mice_eaten_in_decade_l2186_218657


namespace ax_product_zero_l2186_218627

theorem ax_product_zero 
  {a x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ x₁₀ x₁₁ x₁₂ x₁₃ : ℤ} 
  (h1 : a = (1 + x₁) * (1 + x₂) * (1 + x₃) * (1 + x₄) * (1 + x₅) * (1 + x₆) * (1 + x₇) *
           (1 + x₈) * (1 + x₉) * (1 + x₁₀) * (1 + x₁₁) * (1 + x₁₂) * (1 + x₁₃))
  (h2 : a = (1 - x₁) * (1 - x₂) * (1 - x₃) * (1 - x₄) * (1 - x₅) * (1 - x₆) * (1 - x₇) *
           (1 - x₈) * (1 - x₉) * (1 - x₁₀) * (1 - x₁₁) * (1 - x₁₂) * (1 - x₁₃)) :
  a * x₁ * x₂ * x₃ * x₄ * x₅ * x₆ * x₇ * x₈ * x₉ * x₁₀ * x₁₁ * x₁₂ * x₁₃ = 0 := 
sorry

end ax_product_zero_l2186_218627


namespace train_pass_bridge_in_approx_26_64_sec_l2186_218698

noncomputable def L_train : ℝ := 240 -- Length of the train in meters
noncomputable def L_bridge : ℝ := 130 -- Length of the bridge in meters
noncomputable def Speed_train_kmh : ℝ := 50 -- Speed of the train in km/h
noncomputable def Speed_train_ms : ℝ := (Speed_train_kmh * 1000) / 3600 -- Speed of the train in m/s
noncomputable def Total_distance : ℝ := L_train + L_bridge -- Total distance to be covered by the train
noncomputable def Time : ℝ := Total_distance / Speed_train_ms -- Time to pass the bridge

theorem train_pass_bridge_in_approx_26_64_sec : |Time - 26.64| < 0.01 := by
  sorry

end train_pass_bridge_in_approx_26_64_sec_l2186_218698


namespace corner_contains_same_color_cells_l2186_218637

theorem corner_contains_same_color_cells (colors : Finset (Fin 120)) :
  ∀ (coloring : Fin 2017 × Fin 2017 → Fin 120),
  ∃ (corner : Fin 2017 × Fin 2017 → Prop), 
    (∃ cell1 cell2, corner cell1 ∧ corner cell2 ∧ coloring cell1 = coloring cell2) := 
by 
  sorry

end corner_contains_same_color_cells_l2186_218637


namespace fare_range_l2186_218678

noncomputable def fare (x : ℝ) : ℝ :=
  if x <= 3 then 8 else 8 + 1.5 * (x - 3)

theorem fare_range (x : ℝ) (hx : fare x = 16) : 8 ≤ x ∧ x < 9 :=
by
  sorry

end fare_range_l2186_218678


namespace find_n_l2186_218640

theorem find_n (n : ℕ) : (16 : ℝ)^(1/4) = 2^n ↔ n = 1 := by
  sorry

end find_n_l2186_218640


namespace price_increase_percentage_l2186_218631

theorem price_increase_percentage (original_price new_price : ℝ) (h₁ : original_price = 300) (h₂ : new_price = 360) : 
  (new_price - original_price) / original_price * 100 = 20 := 
by
  sorry

end price_increase_percentage_l2186_218631


namespace measure_angle_ABC_l2186_218647

theorem measure_angle_ABC (x : ℝ) (h1 : ∃ θ, θ = 180 - x ∧ x / 2 = (180 - x) / 3) : x = 72 :=
by
  sorry

end measure_angle_ABC_l2186_218647


namespace golden_ratio_problem_l2186_218638

noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

theorem golden_ratio_problem (m : ℝ) (x : ℝ) :
  (1000 ≤ m) → (1000 ≤ x) → (x ≤ m) →
  ((m - 1000) / (x - 1000) = phi ∧ (x - 1000) / (m - x) = phi) →
  (m = 2000 ∨ m = 2618) :=
by
  sorry

end golden_ratio_problem_l2186_218638


namespace correct_sum_l2186_218667

theorem correct_sum (a b c n : ℕ) (h_m_pos : 100 * a + 10 * b + c > 0) (h_n_pos : n > 0)
    (h_err_sum : 100 * a + 10 * c + b + n = 128) : 100 * a + 10 * b + c + n = 128 := 
by
  sorry

end correct_sum_l2186_218667


namespace binary_11101_to_decimal_l2186_218622

theorem binary_11101_to_decimal : 
  (1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 29) := by
  sorry

end binary_11101_to_decimal_l2186_218622


namespace condition_a_neither_necessary_nor_sufficient_for_b_l2186_218662

theorem condition_a_neither_necessary_nor_sufficient_for_b {x y : ℝ} (h : ¬(x = 1 ∧ y = 2)) (k : ¬(x + y = 3)) : ¬((x ≠ 1 ∧ y ≠ 2) ↔ (x + y ≠ 3)) :=
by
  sorry

end condition_a_neither_necessary_nor_sufficient_for_b_l2186_218662


namespace train_pass_time_eq_4_seconds_l2186_218630

-- Define the length of the train in meters
def train_length : ℕ := 40

-- Define the speed of the train in km/h
def train_speed_kmph : ℕ := 36

-- Conversion factor: 1 kmph = 1000 meters / 3600 seconds
def conversion_factor : ℚ := 1000 / 3600

-- Convert the train's speed from km/h to m/s
def train_speed_mps : ℚ := train_speed_kmph * conversion_factor

-- Calculate the time to pass the telegraph post
def time_to_pass_post : ℚ := train_length / train_speed_mps

-- The goal: prove the actual time is 4 seconds
theorem train_pass_time_eq_4_seconds : time_to_pass_post = 4 := by
  sorry

end train_pass_time_eq_4_seconds_l2186_218630


namespace next_number_in_sequence_is_131_l2186_218651

/-- Define the sequence increments between subsequent numbers -/
def sequencePattern : List ℕ := [1, 2, 2, 4, 2, 4, 2, 4, 6, 2]

-- Function to apply a sequence of increments starting from an initial value
def computeNext (initial : ℕ) (increments : List ℕ) : ℕ :=
  increments.foldl (λ acc inc => acc + inc) initial

-- Function to get the sequence's nth element 
def sequenceNthElement (n : ℕ) : ℕ :=
  (computeNext 12 (sequencePattern.take n))

-- Proof that the next number in the sequence is 131 
theorem next_number_in_sequence_is_131 :
  sequenceNthElement 10 = 131 :=
  by
  -- Proof omitted
  sorry

end next_number_in_sequence_is_131_l2186_218651


namespace skating_average_l2186_218686

variable (minutesPerDay1 minutesPerDay2 : Nat)
variable (days1 days2 totalDays requiredAverage : Nat)

theorem skating_average :
  minutesPerDay1 = 80 →
  days1 = 6 →
  minutesPerDay2 = 100 →
  days2 = 2 →
  totalDays = 9 →
  requiredAverage = 95 →
  (minutesPerDay1 * days1 + minutesPerDay2 * days2 + x) / totalDays = requiredAverage →
  x = 175 :=
by
  intro h1 h2 h3 h4 h5 h6 h7
  sorry

end skating_average_l2186_218686


namespace no_solution_inequality_l2186_218619

theorem no_solution_inequality (a b x : ℝ) (h : |a - b| > 2) : ¬(|x - a| + |x - b| ≤ 2) :=
sorry

end no_solution_inequality_l2186_218619


namespace jesse_remaining_pages_l2186_218684

theorem jesse_remaining_pages (pages_read : ℕ)
  (h1 : pages_read = 83)
  (h2 : pages_read = (1 / 3 : ℝ) * total_pages)
  : pages_remaining = 166 :=
  by 
    -- Here we would build the proof, skipped with sorry
    sorry

end jesse_remaining_pages_l2186_218684


namespace probability_of_not_red_l2186_218692

-- Definitions based on conditions
def total_number_of_jelly_beans : ℕ := 7 + 9 + 10 + 12 + 5
def number_of_non_red_jelly_beans : ℕ := 9 + 10 + 12 + 5

-- Proving the probability
theorem probability_of_not_red : 
  (number_of_non_red_jelly_beans : ℚ) / total_number_of_jelly_beans = 36 / 43 :=
by sorry

end probability_of_not_red_l2186_218692


namespace soda_cost_l2186_218677

-- Definitions of the given conditions
def initial_amount : ℝ := 40
def cost_pizza : ℝ := 2.75
def cost_jeans : ℝ := 11.50
def quarters_left : ℝ := 97
def value_per_quarter : ℝ := 0.25

-- Calculate amount left in dollars
def amount_left : ℝ := quarters_left * value_per_quarter

-- Statement we want to prove: the cost of the soda
theorem soda_cost :
  initial_amount - amount_left - (cost_pizza + cost_jeans) = 1.5 :=
by
  sorry

end soda_cost_l2186_218677


namespace tangent_and_normal_are_correct_at_point_l2186_218669

def point_on_curve (x y : ℝ) : Prop :=
  x^2 - 2*x*y + 3*y^2 - 2*y - 16 = 0

def tangent_line (x y : ℝ) : Prop :=
  2*x - 7*y + 19 = 0

def normal_line (x y : ℝ) : Prop :=
  7*x + 2*y - 13 = 0

theorem tangent_and_normal_are_correct_at_point
  (hx : point_on_curve 1 3) :
  tangent_line 1 3 ∧ normal_line 1 3 :=
by
  sorry

end tangent_and_normal_are_correct_at_point_l2186_218669


namespace quadratic_inequality_solution_l2186_218615

theorem quadratic_inequality_solution :
  ∀ x : ℝ, (3 * x^2 - 5 * x - 2 < 0) ↔ (-1/3 < x ∧ x < 2) :=
by
  sorry

end quadratic_inequality_solution_l2186_218615


namespace camp_weights_l2186_218636

theorem camp_weights (m_e_w : ℕ) (m_e_w1 : ℕ) (c_w : ℕ) (m_e_w2 : ℕ) (d : ℕ)
  (h1 : m_e_w = 30) 
  (h2 : m_e_w1 = 28) 
  (h3 : c_w = 56)
  (h4 : m_e_w = m_e_w1 + d)
  (h5 : m_e_w1 = m_e_w2 + d)
  (h6 : c_w = m_e_w + m_e_w1 + d) :
  m_e_w = 28 ∧ m_e_w2 = 26 := 
by {
    sorry
}

end camp_weights_l2186_218636


namespace eq_satisfies_exactly_four_points_l2186_218655

theorem eq_satisfies_exactly_four_points : ∀ (x y : ℝ), 
  (x^2 - 4)^2 + (y^2 - 4)^2 = 0 ↔ 
  (x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2) ∨ (x = 2 ∧ y = -2) ∨ (x = -2 ∧ y = -2) := 
by
  sorry

end eq_satisfies_exactly_four_points_l2186_218655


namespace fraction_of_population_married_l2186_218607

theorem fraction_of_population_married
  (M W N : ℕ)
  (h1 : (2 / 3 : ℚ) * M = N)
  (h2 : (3 / 5 : ℚ) * W = N)
  : ((2 * N) : ℚ) / (M + W) = 12 / 19 := 
by
  sorry

end fraction_of_population_married_l2186_218607


namespace polynomial_transformation_l2186_218625

theorem polynomial_transformation (x y : ℂ) (h : y = x + 1/x) : x^4 + x^3 - 4*x^2 + x + 1 = 0 ↔ x^2 * (y^2 + y - 6) = 0 :=
by
  sorry

end polynomial_transformation_l2186_218625


namespace integers_sum_eighteen_l2186_218600

theorem integers_sum_eighteen (a b : ℕ) (h₀ : a ≠ b) (h₁ : a < 20) (h₂ : b < 20) (h₃ : Nat.gcd a b = 1) 
(h₄ : a * b + a + b = 95) : a + b = 18 :=
by
  sorry

end integers_sum_eighteen_l2186_218600


namespace problem1_problem2_l2186_218693

variable (α : ℝ)

-- First problem statement
theorem problem1 (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.sin α + 3 * Real.cos α) = 6 / 13 :=
by 
  sorry

-- Second problem statement
theorem problem2 (h : Real.tan α = 2) :
  3 * (Real.sin α)^2 + 3 * Real.sin α * Real.cos α - 2 * (Real.cos α)^2 = 16 / 5 :=
by 
  sorry

end problem1_problem2_l2186_218693


namespace complete_square_form_l2186_218611

theorem complete_square_form (a b x : ℝ) : 
  ∃ (p : ℝ) (q : ℝ), 
  (p = x ∧ q = 1 ∧ (x^2 + 2*x + 1 = (p + q)^2)) ∧ 
  (¬ ∃ (p q : ℝ), a^2 + 4 = (a + p) * (a + q)) ∧
  (¬ ∃ (p q : ℝ), a^2 + a*b + b^2 = (a + p) * (a + q)) ∧
  (¬ ∃ (p q : ℝ), a^2 + 4*a*b + b^2 = (a + p) * (a + q)) :=
  sorry

end complete_square_form_l2186_218611


namespace solve_for_t_l2186_218646

theorem solve_for_t (p t : ℝ) (h1 : 5 = p * 3^t) (h2 : 45 = p * 9^t) : t = 2 :=
by
  sorry

end solve_for_t_l2186_218646


namespace remainder_5_pow_100_div_18_l2186_218614

theorem remainder_5_pow_100_div_18 : (5 ^ 100) % 18 = 13 := 
  sorry

end remainder_5_pow_100_div_18_l2186_218614


namespace total_tbs_of_coffee_l2186_218695

theorem total_tbs_of_coffee (guests : ℕ) (weak_drinkers : ℕ) (medium_drinkers : ℕ) (strong_drinkers : ℕ) 
                           (cups_per_weak_drinker : ℕ) (cups_per_medium_drinker : ℕ) (cups_per_strong_drinker : ℕ) 
                           (tbsp_per_cup_weak : ℕ) (tbsp_per_cup_medium : ℝ) (tbsp_per_cup_strong : ℕ) :
  guests = 18 ∧ 
  weak_drinkers = 6 ∧ 
  medium_drinkers = 6 ∧ 
  strong_drinkers = 6 ∧ 
  cups_per_weak_drinker = 2 ∧ 
  cups_per_medium_drinker = 3 ∧ 
  cups_per_strong_drinker = 1 ∧ 
  tbsp_per_cup_weak = 1 ∧ 
  tbsp_per_cup_medium = 1.5 ∧ 
  tbsp_per_cup_strong = 2 →
  (weak_drinkers * cups_per_weak_drinker * tbsp_per_cup_weak + 
   medium_drinkers * cups_per_medium_drinker * tbsp_per_cup_medium + 
   strong_drinkers * cups_per_strong_drinker * tbsp_per_cup_strong) = 51 :=
by
  sorry

end total_tbs_of_coffee_l2186_218695


namespace average_annual_growth_rate_l2186_218650

variable (a b : ℝ)

theorem average_annual_growth_rate :
  ∃ x : ℝ, (1 + x)^2 = (1 + a) * (1 + b) ∧ x = Real.sqrt ((1 + a) * (1 + b)) - 1 := by
  sorry

end average_annual_growth_rate_l2186_218650


namespace horner_v4_at_2_l2186_218679

def horner (a : List Int) (x : Int) : Int :=
  a.foldr (fun ai acc => ai + x * acc) 0

noncomputable def poly_coeffs : List Int := [1, -12, 60, -160, 240, -192, 64]

theorem horner_v4_at_2 : horner poly_coeffs 2 = 80 := by
  sorry

end horner_v4_at_2_l2186_218679


namespace hanoi_moves_minimal_l2186_218604

theorem hanoi_moves_minimal (n : ℕ) : ∃ m, 
  (∀ move : ℕ, move = 2^n - 1 → move = m) := 
by
  sorry

end hanoi_moves_minimal_l2186_218604


namespace average_ab_l2186_218664

theorem average_ab {a b : ℝ} (h : (3 + 5 + 7 + a + b) / 5 = 15) : (a + b) / 2 = 30 :=
by
  sorry

end average_ab_l2186_218664


namespace final_state_of_marbles_after_operations_l2186_218656

theorem final_state_of_marbles_after_operations :
  ∃ (b w : ℕ), b + w = 2 ∧ w = 2 ∧ (∀ n : ℕ, n % 2 = 0 → n = 100 - k * 2) :=
sorry

end final_state_of_marbles_after_operations_l2186_218656


namespace total_sum_of_ages_is_correct_l2186_218683

-- Definition of conditions
def ageOfYoungestChild : Nat := 4
def intervals : Nat := 3

-- Total sum calculation
def sumOfAges (ageOfYoungestChild intervals : Nat) :=
  let Y := ageOfYoungestChild
  Y + (Y + intervals) + (Y + 2 * intervals) + (Y + 3 * intervals) + (Y + 4 * intervals)

theorem total_sum_of_ages_is_correct : sumOfAges 4 3 = 50 :=
by
  sorry

end total_sum_of_ages_is_correct_l2186_218683


namespace remainder_2n_div_9_l2186_218620

theorem remainder_2n_div_9 (n : ℤ) (h : n % 18 = 10) : (2 * n) % 9 = 2 := 
sorry

end remainder_2n_div_9_l2186_218620


namespace magic_square_sum_l2186_218659

-- Given conditions
def magic_square (S : ℕ) (a b c d e : ℕ) :=
  (30 + b + 27 = S) ∧
  (30 + 33 + a = S) ∧
  (33 + c + d = S) ∧
  (a + 18 + e = S) ∧
  (30 + c + e = S)

-- Prove that the sum a + d is 38 given the sums of the 3x3 magic square are equivalent
theorem magic_square_sum (a b c d e S : ℕ) (h : magic_square S a b c d e) : a + d = 38 :=
  sorry

end magic_square_sum_l2186_218659


namespace opposite_numbers_reciprocal_values_l2186_218688

theorem opposite_numbers_reciprocal_values (a b m n : ℝ) (h₁ : a + b = 0) (h₂ : m * n = 1) : 5 * a + 5 * b - m * n = -1 :=
by sorry

end opposite_numbers_reciprocal_values_l2186_218688


namespace max_value_x_y3_z4_l2186_218685

theorem max_value_x_y3_z4 (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 2) :
  x + y^3 + z^4 ≤ 2 :=
by
  sorry

end max_value_x_y3_z4_l2186_218685


namespace dogs_in_shelter_l2186_218653

theorem dogs_in_shelter (D C : ℕ) (h1 : D * 7 = 15 * C) (h2 : D * 11 = 15 * (C + 8)) :
  D = 30 :=
sorry

end dogs_in_shelter_l2186_218653


namespace divisors_of_30240_l2186_218649

theorem divisors_of_30240 : 
  ∃ s : Finset ℕ, (s = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (∀ d ∈ s, (30240 % d = 0)) ∧ (s.card = 9) :=
by
  sorry

end divisors_of_30240_l2186_218649


namespace opposite_of_neg_2023_l2186_218602

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l2186_218602


namespace average_people_per_hour_l2186_218674

theorem average_people_per_hour (total_people : ℕ) (days : ℕ) (hours_per_day : ℕ) (total_hours : ℕ) (average_per_hour : ℕ) :
  total_people = 3000 ∧ days = 5 ∧ hours_per_day = 24 ∧ total_hours = days * hours_per_day ∧ average_per_hour = total_people / total_hours → 
  average_per_hour = 25 :=
by
  sorry

end average_people_per_hour_l2186_218674


namespace bicentric_quad_lemma_l2186_218617

-- Define the properties and radii of the bicentric quadrilateral
variables (KLMN : Type) (r ρ h : ℝ)

-- Assuming quadrilateral KLMN is bicentric with given radii
def is_bicentric (KLMN : Type) := true

-- State the theorem we wish to prove
theorem bicentric_quad_lemma (br : is_bicentric KLMN) : 
  (1 / (ρ + h) ^ 2) + (1 / (ρ - h) ^ 2) = (1 / r ^ 2) :=
sorry

end bicentric_quad_lemma_l2186_218617


namespace isosceles_triangle_altitude_l2186_218660

open Real

theorem isosceles_triangle_altitude (DE DF DG EG GF EF : ℝ) (h1 : DE = 5) (h2 : DF = 5) (h3 : EG = 2 * GF)
(h4 : DG = sqrt (DE^2 - GF^2)) (h5 : EF = EG + GF) (h6 : EF = 3 * GF) : EF = 5 :=
by
  -- Proof would go here
  sorry

end isosceles_triangle_altitude_l2186_218660


namespace Bennett_sales_l2186_218696

-- Define the variables for the number of screens sold in each month.
variables (J F M : ℕ)

-- State the given conditions.
theorem Bennett_sales (h1: F = 2 * J) (h2: F = M / 4) (h3: M = 8800) :
  J + F + M = 12100 := by
sorry

end Bennett_sales_l2186_218696


namespace hyperbola_equation_l2186_218681

theorem hyperbola_equation {x y : ℝ} (h1 : x ^ 2 / 2 - y ^ 2 = 1) 
  (h2 : x = -2) (h3 : y = 2) : y ^ 2 / 2 - x ^ 2 / 4 = 1 :=
by sorry

end hyperbola_equation_l2186_218681


namespace inequality_solution_l2186_218616

theorem inequality_solution (x : ℝ) (h : 1 / (x - 2) < 4) : x < 2 ∨ x > 9 / 4 :=
sorry

end inequality_solution_l2186_218616


namespace total_coronavirus_cases_l2186_218663

theorem total_coronavirus_cases (ny_cases ca_cases tx_cases : ℕ)
    (h_ny : ny_cases = 2000)
    (h_ca : ca_cases = ny_cases / 2)
    (h_tx : ca_cases = tx_cases + 400) :
    ny_cases + ca_cases + tx_cases = 3600 := by
  sorry

end total_coronavirus_cases_l2186_218663


namespace value_of_x_squared_plus_reciprocal_squared_l2186_218672

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (hx : 0 < x) (h : x + 1/x = Real.sqrt 2020) : x^2 + 1/x^2 = 2018 :=
sorry

end value_of_x_squared_plus_reciprocal_squared_l2186_218672


namespace polynomial_divisibility_condition_l2186_218648

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := x^5 - x^4 + x^3 - p * x^2 + q * x - 6

theorem polynomial_divisibility_condition (p q : ℝ) :
  (f (-1) p q = 0) ∧ (f 2 p q = 0) → 
  (p = 0) ∧ (q = -9) := by
  sorry

end polynomial_divisibility_condition_l2186_218648


namespace find_lost_bowls_l2186_218610

def bowls_problem (L : ℕ) : Prop :=
  let total_bowls := 638
  let broken_bowls := 15
  let payment := 1825
  let fee := 100
  let safe_bowl_payment := 3
  let lost_broken_bowl_cost := 4
  100 + 3 * (total_bowls - L - broken_bowls) - 4 * (L + broken_bowls) = payment

theorem find_lost_bowls : ∃ L : ℕ, bowls_problem L ∧ L = 26 :=
  by
  sorry

end find_lost_bowls_l2186_218610


namespace john_gallons_of_gas_l2186_218690

theorem john_gallons_of_gas
  (rental_cost : ℝ)
  (gas_cost_per_gallon : ℝ)
  (mile_cost : ℝ)
  (miles_driven : ℝ)
  (total_cost : ℝ)
  (rental_cost_val : rental_cost = 150)
  (gas_cost_per_gallon_val : gas_cost_per_gallon = 3.50)
  (mile_cost_val : mile_cost = 0.50)
  (miles_driven_val : miles_driven = 320)
  (total_cost_val : total_cost = 338) :
  ∃ gallons_of_gas : ℝ, gallons_of_gas = 8 :=
by
  sorry

end john_gallons_of_gas_l2186_218690


namespace find_f4_l2186_218624

variable (a b : ℝ)
variable (f : ℝ → ℝ)
variable (h1 : f 1 = 5)
variable (h2 : f 2 = 8)
variable (h3 : f 3 = 11)
variable (h4 : ∀ x, f x = a * x + b)

theorem find_f4 : f 4 = 14 := by
  sorry

end find_f4_l2186_218624


namespace sixty_percent_of_total_is_960_l2186_218623

-- Definitions from the conditions
def boys : ℕ := 600
def difference : ℕ := 400
def girls : ℕ := boys + difference
def total : ℕ := boys + girls
def sixty_percent_of_total : ℕ := total * 60 / 100

-- The theorem to prove
theorem sixty_percent_of_total_is_960 :
  sixty_percent_of_total = 960 := 
  sorry

end sixty_percent_of_total_is_960_l2186_218623


namespace part_1_part_2_l2186_218641

def p (a x : ℝ) : Prop :=
a * x - 2 ≤ 0 ∧ a * x + 1 > 0

def q (x : ℝ) : Prop :=
x^2 - x - 2 < 0

theorem part_1 (a : ℝ) :
  (∃ x : ℝ, (1/2 < x ∧ x < 3) ∧ p a x) → 
  (-2 < a ∧ a < 4) :=
sorry

theorem part_2 (a : ℝ) :
  (∀ x, p a x → q x) ∧ 
  (∃ x, q x ∧ ¬p a x) → 
  (-1/2 ≤ a ∧ a ≤ 1) :=
sorry

end part_1_part_2_l2186_218641


namespace sum_of_squares_divisible_by_7_implies_product_divisible_by_49_l2186_218639

theorem sum_of_squares_divisible_by_7_implies_product_divisible_by_49 (a b : ℕ) 
  (h : (a * a + b * b) % 7 = 0) : (a * b) % 49 = 0 :=
sorry

end sum_of_squares_divisible_by_7_implies_product_divisible_by_49_l2186_218639
