import Mathlib

namespace NUMINAMATH_GPT_three_boys_in_shop_at_same_time_l257_25798

-- Definitions for the problem conditions
def boys : Type := Fin 7  -- Representing the 7 boys
def visits : Type := Fin 3  -- Each boy makes 3 visits

-- A structure representing a visit by a boy
structure Visit := (boy : boys) (visit_num : visits)

-- Meeting condition: Every pair of boys meets at the shop
def meets_at_shop (v1 v2 : Visit) : Prop :=
  v1.boy ≠ v2.boy  -- Ensure it's not the same boy (since we assume each pair meets)

-- The theorem to be proven
theorem three_boys_in_shop_at_same_time :
  ∃ (v1 v2 v3 : Visit), v1.boy ≠ v2.boy ∧ v2.boy ≠ v3.boy ∧ v1.boy ≠ v3.boy :=
sorry

end NUMINAMATH_GPT_three_boys_in_shop_at_same_time_l257_25798


namespace NUMINAMATH_GPT_margaret_time_is_10_minutes_l257_25797

variable (time_billy_first_5_laps : ℕ)
variable (time_billy_next_3_laps : ℕ)
variable (time_billy_next_lap : ℕ)
variable (time_billy_final_lap : ℕ)
variable (time_difference : ℕ)

def billy_total_time := time_billy_first_5_laps + time_billy_next_3_laps + time_billy_next_lap + time_billy_final_lap

def margaret_total_time := billy_total_time + time_difference

theorem margaret_time_is_10_minutes :
  time_billy_first_5_laps = 120 ∧
  time_billy_next_3_laps = 240 ∧
  time_billy_next_lap = 60 ∧
  time_billy_final_lap = 150 ∧
  time_difference = 30 →
  margaret_total_time = 600 :=
by 
  sorry

end NUMINAMATH_GPT_margaret_time_is_10_minutes_l257_25797


namespace NUMINAMATH_GPT_find_A_d_minus_B_d_l257_25702

-- Definitions of the conditions
def is_digit_in_base (x : ℕ) (d : ℕ) : Prop := x < d

def ab_aa_sum_to_172 (A B d : ℕ) : Prop :=
  is_digit_in_base A d ∧ is_digit_in_base B d ∧ d > 7 ∧ (d * A + B) + (d * A + A) = d^2 + 7 * d + 2

-- The final theorem statement
theorem find_A_d_minus_B_d (A B d : ℕ) (h : ab_aa_sum_to_172 A B d) : A - B = 5 :=
by sorry

end NUMINAMATH_GPT_find_A_d_minus_B_d_l257_25702


namespace NUMINAMATH_GPT_sum_first_60_natural_numbers_l257_25767

theorem sum_first_60_natural_numbers : (60 * (60 + 1)) / 2 = 1830 := by
  sorry

end NUMINAMATH_GPT_sum_first_60_natural_numbers_l257_25767


namespace NUMINAMATH_GPT_exists_x_y_for_2021_pow_n_l257_25752

theorem exists_x_y_for_2021_pow_n (n : ℕ) :
  (∃ x y : ℤ, 2021 ^ n = x ^ 4 - 4 * y ^ 4) ↔ ∃ m : ℕ, n = 4 * m := 
sorry

end NUMINAMATH_GPT_exists_x_y_for_2021_pow_n_l257_25752


namespace NUMINAMATH_GPT_percentage_increase_l257_25765

variables (a b x m : ℝ) (p : ℝ)
variables (h1 : a / b = 4 / 5)
variables (h2 : x = a + (p / 100) * a)
variables (h3 : m = b - 0.6 * b)
variables (h4 : m / x = 0.4)

theorem percentage_increase (a_pos : 0 < a) (b_pos : 0 < b) : p = 25 :=
by sorry

end NUMINAMATH_GPT_percentage_increase_l257_25765


namespace NUMINAMATH_GPT_range_of_a_l257_25787

theorem range_of_a {a : ℝ} : (∀ x : ℝ, (x^2 + 2 * (a + 1) * x + a^2 - 1 = 0) → (x = 0 ∨ x = -4)) → (a = 1 ∨ a ≤ -1) := 
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l257_25787


namespace NUMINAMATH_GPT_second_train_length_is_120_l257_25721

noncomputable def length_of_second_train
  (speed_train1_kmph : ℝ) 
  (speed_train2_kmph : ℝ) 
  (crossing_time : ℝ) 
  (length_train1_m : ℝ) : ℝ :=
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600
  let speed_train2_mps := speed_train2_kmph * 1000 / 3600
  let relative_speed := speed_train1_mps + speed_train2_mps
  let distance := relative_speed * crossing_time
  distance - length_train1_m

theorem second_train_length_is_120 :
  length_of_second_train 60 40 6.119510439164867 50 = 120 :=
by
  -- Here's where the proof would go
  sorry

end NUMINAMATH_GPT_second_train_length_is_120_l257_25721


namespace NUMINAMATH_GPT_total_amount_spent_l257_25718

-- Definitions based on the conditions
def games_this_month := 11
def cost_per_ticket_this_month := 25
def total_cost_this_month := games_this_month * cost_per_ticket_this_month

def games_last_month := 17
def cost_per_ticket_last_month := 30
def total_cost_last_month := games_last_month * cost_per_ticket_last_month

def games_next_month := 16
def cost_per_ticket_next_month := 35
def total_cost_next_month := games_next_month * cost_per_ticket_next_month

-- Lean statement for the proof problem
theorem total_amount_spent :
  total_cost_this_month + total_cost_last_month + total_cost_next_month = 1345 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_amount_spent_l257_25718


namespace NUMINAMATH_GPT_find_additional_discount_percentage_l257_25743

noncomputable def additional_discount_percentage(msrp : ℝ) (max_regular_discount : ℝ) (lowest_price : ℝ) : ℝ :=
  let regular_discount_price := msrp * (1 - max_regular_discount)
  let additional_discount := (regular_discount_price - lowest_price) / regular_discount_price
  additional_discount * 100

theorem find_additional_discount_percentage :
  additional_discount_percentage 40 0.3 22.4 = 20 :=
by
  unfold additional_discount_percentage
  simp
  sorry

end NUMINAMATH_GPT_find_additional_discount_percentage_l257_25743


namespace NUMINAMATH_GPT_num_three_digit_perfect_cubes_divisible_by_16_l257_25746

-- define what it means for an integer to be a three-digit number
def is_three_digit (n : ℤ) : Prop := 100 ≤ n ∧ n ≤ 999

-- define what it means for an integer to be a perfect cube
def is_perfect_cube (n : ℤ) : Prop := ∃ m : ℤ, m^3 = n

-- define what it means for an integer to be divisible by 16
def is_divisible_by_sixteen (n : ℤ) : Prop := n % 16 = 0

-- define the main theorem that combines these conditions
theorem num_three_digit_perfect_cubes_divisible_by_16 : 
  ∃ n, n = 2 := sorry

end NUMINAMATH_GPT_num_three_digit_perfect_cubes_divisible_by_16_l257_25746


namespace NUMINAMATH_GPT_exponentiation_properties_l257_25708

theorem exponentiation_properties
  (a : ℝ) (m n : ℕ) (hm : a^m = 9) (hn : a^n = 3) : a^(m - n) = 3 :=
by
  sorry

end NUMINAMATH_GPT_exponentiation_properties_l257_25708


namespace NUMINAMATH_GPT_same_function_representation_l257_25722

theorem same_function_representation : 
  ∀ (f g : ℝ → ℝ), 
    (∀ x, f x = x^2 - 2*x - 1) ∧ (∀ m, g m = m^2 - 2*m - 1) →
    (f = g) :=
by
  sorry

end NUMINAMATH_GPT_same_function_representation_l257_25722


namespace NUMINAMATH_GPT_quadratic_equation_nonzero_coefficient_l257_25774

theorem quadratic_equation_nonzero_coefficient (m : ℝ) : 
  m - 1 ≠ 0 ↔ m ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_nonzero_coefficient_l257_25774


namespace NUMINAMATH_GPT_students_not_making_cut_l257_25724

theorem students_not_making_cut
  (girls boys called_back : ℕ) 
  (h1 : girls = 39) 
  (h2 : boys = 4) 
  (h3 : called_back = 26) :
  (girls + boys) - called_back = 17 := 
by sorry

end NUMINAMATH_GPT_students_not_making_cut_l257_25724


namespace NUMINAMATH_GPT_t_shirts_per_package_l257_25788

theorem t_shirts_per_package (total_tshirts : ℕ) (packages : ℕ) (tshirts_per_package : ℕ) :
  total_tshirts = 70 → packages = 14 → tshirts_per_package = total_tshirts / packages → tshirts_per_package = 5 :=
by
  sorry

end NUMINAMATH_GPT_t_shirts_per_package_l257_25788


namespace NUMINAMATH_GPT_winning_percentage_is_70_l257_25753

def percentage_of_votes (P : ℝ) : Prop :=
  ∃ (P : ℝ), (7 * P - 7 * (100 - P) = 280 ∧ 0 ≤ P ∧ P ≤ 100)

theorem winning_percentage_is_70 :
  percentage_of_votes 70 :=
by
  sorry

end NUMINAMATH_GPT_winning_percentage_is_70_l257_25753


namespace NUMINAMATH_GPT_sum_of_squares_of_distances_is_constant_l257_25705

variable {r1 r2 : ℝ}
variable {x y : ℝ}

theorem sum_of_squares_of_distances_is_constant
  (h1 : r1 < r2)
  (h2 : x^2 + y^2 = r1^2) :
  let PA := (x - r2)^2 + y^2
  let PB := (x + r2)^2 + y^2
  PA + PB = 2 * r1^2 + 2 * r2^2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_distances_is_constant_l257_25705


namespace NUMINAMATH_GPT_fraction_of_desks_full_l257_25729

-- Define the conditions
def restroom_students : ℕ := 2
def absent_students : ℕ := (3 * restroom_students) - 1
def total_students : ℕ := 23
def desks_per_row : ℕ := 6
def number_of_rows : ℕ := 4
def total_desks : ℕ := desks_per_row * number_of_rows
def students_in_classroom : ℕ := total_students - absent_students - restroom_students

-- Prove the fraction of desks that are full
theorem fraction_of_desks_full : (students_in_classroom : ℚ) / (total_desks : ℚ) = 2 / 3 :=
by
    sorry

end NUMINAMATH_GPT_fraction_of_desks_full_l257_25729


namespace NUMINAMATH_GPT_percent_millet_mix_correct_l257_25725

-- Define the necessary percentages
def percent_BrandA_in_mix : ℝ := 0.60
def percent_BrandB_in_mix : ℝ := 0.40
def percent_millet_in_BrandA : ℝ := 0.60
def percent_millet_in_BrandB : ℝ := 0.65

-- Define the overall percentage of millet in the mix
def percent_millet_in_mix : ℝ :=
  percent_BrandA_in_mix * percent_millet_in_BrandA +
  percent_BrandB_in_mix * percent_millet_in_BrandB

-- State the theorem
theorem percent_millet_mix_correct :
  percent_millet_in_mix = 0.62 :=
  by
    -- Here, we would provide the proof, but we use sorry as instructed.
    sorry

end NUMINAMATH_GPT_percent_millet_mix_correct_l257_25725


namespace NUMINAMATH_GPT_ones_digit_of_sum_is_0_l257_25772

-- Define the integer n
def n : ℕ := 2012

-- Define the ones digit function
def ones_digit (x : ℕ) : ℕ := x % 10

-- Define the power function mod 10
def power_mod_10 (d a : ℕ) : ℕ := (d^a) % 10

-- Define the sequence sum for ones digits
def seq_sum_mod_10 (m : ℕ) : ℕ :=
  Finset.sum (Finset.range m) (λ k => power_mod_10 (k+1) n)

-- Define the final sum mod 10 considering the repeating cycle and sum
def total_ones_digit_sum (a b : ℕ) : ℕ :=
  let cycle_sum := Finset.sum (Finset.range 10) (λ k => power_mod_10 (k+1) n)
  let s := cycle_sum * (a / 10) + Finset.sum (Finset.range b) (λ k => power_mod_10 (k+1) n)
  s % 10

-- Prove that the ones digit of the sum is 0
theorem ones_digit_of_sum_is_0 : total_ones_digit_sum n (n % 10) = 0 :=
sorry

end NUMINAMATH_GPT_ones_digit_of_sum_is_0_l257_25772


namespace NUMINAMATH_GPT_amount_of_cocoa_powder_given_by_mayor_l257_25755

def total_cocoa_powder_needed : ℕ := 306
def cocoa_powder_still_needed : ℕ := 47

def cocoa_powder_given_by_mayor : ℕ :=
  total_cocoa_powder_needed - cocoa_powder_still_needed

theorem amount_of_cocoa_powder_given_by_mayor :
  cocoa_powder_given_by_mayor = 259 := by
  sorry

end NUMINAMATH_GPT_amount_of_cocoa_powder_given_by_mayor_l257_25755


namespace NUMINAMATH_GPT_false_statement_E_l257_25751

theorem false_statement_E
  (A B C : Type)
  (a b c : ℝ)
  (ha_gt_hb : a > b)
  (hb_gt_hc : b > c)
  (AB BC : ℝ)
  (hAB : AB = a - b → True)
  (hBC : BC = b + c → True)
  (hABC : AB + BC > a + b + c → True)
  (hAC : AB + BC > a - c → True) : False := sorry

end NUMINAMATH_GPT_false_statement_E_l257_25751


namespace NUMINAMATH_GPT_parallelogram_angle_l257_25756

theorem parallelogram_angle (a b : ℝ) (h1 : a + b = 180) (h2 : a = b + 50) : b = 65 :=
by
  -- Proof would go here, but we're adding a placeholder
  sorry

end NUMINAMATH_GPT_parallelogram_angle_l257_25756


namespace NUMINAMATH_GPT_replace_batteries_in_December_16_years_later_l257_25728

theorem replace_batteries_in_December_16_years_later :
  ∀ (n : ℕ), n = 30 → ∃ (years : ℕ) (months : ℕ), years = 16 ∧ months = 11 :=
by
  sorry

end NUMINAMATH_GPT_replace_batteries_in_December_16_years_later_l257_25728


namespace NUMINAMATH_GPT_investment_C_l257_25773

-- Definitions of the given conditions
def investment_A : ℝ := 6300
def investment_B : ℝ := 4200
def total_profit : ℝ := 12700
def profit_A : ℝ := 3810

-- Defining the total investment, including C's investment
noncomputable def investment_total_including_C (C : ℝ) : ℝ := investment_A + investment_B + C

-- Proving the correct investment for C under the given conditions
theorem investment_C (C : ℝ) :
  (investment_A / investment_total_including_C C) = (profit_A / total_profit) → 
  C = 10500 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_investment_C_l257_25773


namespace NUMINAMATH_GPT_vector_BC_is_correct_l257_25738

-- Given points B(1,2) and C(4,5)
def point_B := (1, 2)
def point_C := (4, 5)

-- Define the vector BC
def vector_BC (B C : ℕ × ℕ) : ℕ × ℕ :=
  (C.1 - B.1, C.2 - B.2)

-- Prove that the vector BC is (3, 3)
theorem vector_BC_is_correct : vector_BC point_B point_C = (3, 3) :=
  sorry

end NUMINAMATH_GPT_vector_BC_is_correct_l257_25738


namespace NUMINAMATH_GPT_flower_shop_sold_bouquets_l257_25761

theorem flower_shop_sold_bouquets (roses_per_bouquet : ℕ) (daisies_per_bouquet : ℕ) 
  (rose_bouquets_sold : ℕ) (daisy_bouquets_sold : ℕ) (total_flowers_sold : ℕ)
  (h1 : roses_per_bouquet = 12) (h2 : rose_bouquets_sold = 10) 
  (h3 : daisy_bouquets_sold = 10) (h4 : total_flowers_sold = 190) : 
  (rose_bouquets_sold + daisy_bouquets_sold) = 20 :=
by sorry

end NUMINAMATH_GPT_flower_shop_sold_bouquets_l257_25761


namespace NUMINAMATH_GPT_current_babysitter_hourly_rate_l257_25778

-- Define variables
def new_babysitter_hourly_rate := 12
def extra_charge_per_scream := 3
def hours_hired := 6
def number_of_screams := 2
def cost_difference := 18

-- Define the total cost calculations
def new_babysitter_total_cost :=
  new_babysitter_hourly_rate * hours_hired + extra_charge_per_scream * number_of_screams

def current_babysitter_total_cost :=
  new_babysitter_total_cost + cost_difference

theorem current_babysitter_hourly_rate :
  current_babysitter_total_cost / hours_hired = 16 := by
  sorry

end NUMINAMATH_GPT_current_babysitter_hourly_rate_l257_25778


namespace NUMINAMATH_GPT_strawberry_rows_l257_25794

theorem strawberry_rows (yield_per_row total_harvest : ℕ) (h1 : yield_per_row = 268) (h2 : total_harvest = 1876) :
  total_harvest / yield_per_row = 7 := 
by 
  sorry

end NUMINAMATH_GPT_strawberry_rows_l257_25794


namespace NUMINAMATH_GPT_inequality_proof_l257_25789

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) : 
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000 / 9 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l257_25789


namespace NUMINAMATH_GPT_f_divisible_by_64_l257_25735

theorem f_divisible_by_64 (n : ℕ) (h : n > 0) : 64 ∣ (3^(2*n + 2) - 8*n - 9) :=
sorry

end NUMINAMATH_GPT_f_divisible_by_64_l257_25735


namespace NUMINAMATH_GPT_inscribed_circle_distance_l257_25715

-- description of the geometry problem
theorem inscribed_circle_distance (r : ℝ) (AB : ℝ):
  r = 4 →
  AB = 4 →
  ∃ d : ℝ, d = 6.4 :=
by
  intros hr hab
  -- skipping proof steps
  let a := 2*r
  let PQ := 2 * r * (Real.sqrt 3 / 2)
  use PQ
  sorry

end NUMINAMATH_GPT_inscribed_circle_distance_l257_25715


namespace NUMINAMATH_GPT_cylinder_volume_increase_l257_25771

theorem cylinder_volume_increase (r h : ℝ) (V : ℝ) : 
  V = π * r^2 * h → 
  (3 * h) * (2 * r)^2 * π = 12 * V := by
    sorry

end NUMINAMATH_GPT_cylinder_volume_increase_l257_25771


namespace NUMINAMATH_GPT_max_value_of_fraction_l257_25700

theorem max_value_of_fraction (a b : ℝ) (ha : a > 0) (hb : b > 1) (h_discriminant : a^2 = 4 * (b - 1)) :
  a = 2 → b = 2 → (3 * a + 2 * b) / (a + b) = 5 / 2 :=
by
  intro ha_eq
  intro hb_eq
  sorry

end NUMINAMATH_GPT_max_value_of_fraction_l257_25700


namespace NUMINAMATH_GPT_fraction_ratio_l257_25745

theorem fraction_ratio (x : ℚ) : 
  (x : ℚ) / (2/6) = (3/4) / (1/2) -> (x = 1/2) :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_ratio_l257_25745


namespace NUMINAMATH_GPT_binary_to_base4_representation_l257_25770

def binary_to_base4 (n : ℕ) : ℕ :=
  -- Assuming implementation that converts binary number n to its base 4 representation 
  sorry

theorem binary_to_base4_representation :
  binary_to_base4 0b10110110010 = 23122 :=
by sorry

end NUMINAMATH_GPT_binary_to_base4_representation_l257_25770


namespace NUMINAMATH_GPT_dime_probability_l257_25757

theorem dime_probability (dime_value quarter_value : ℝ) (dime_worth quarter_worth total_coins: ℕ) :
  dime_value = 0.10 ∧
  quarter_value = 0.25 ∧
  dime_worth = 10 ∧
  quarter_worth = 4 ∧
  total_coins = 14 →
  (dime_worth / total_coins : ℝ) = 5 / 7 :=
by
  sorry

end NUMINAMATH_GPT_dime_probability_l257_25757


namespace NUMINAMATH_GPT_initial_numbers_is_five_l257_25727

theorem initial_numbers_is_five : 
  ∀ (n S : ℕ), 
    (12 * n = S) →
    (10 * (n - 1) = S - 20) → 
    n = 5 := 
by sorry

end NUMINAMATH_GPT_initial_numbers_is_five_l257_25727


namespace NUMINAMATH_GPT_sin_b_in_triangle_l257_25710

theorem sin_b_in_triangle (a b : ℝ) (sin_A sin_B : ℝ) (h₁ : a = 2) (h₂ : b = 1) (h₃ : sin_A = 1 / 3) 
  (h₄ : sin_B = (b * sin_A) / a) : sin_B = 1 / 6 :=
by
  have h₅ : sin_B = 1 / 6 := by 
    sorry
  exact h₅

end NUMINAMATH_GPT_sin_b_in_triangle_l257_25710


namespace NUMINAMATH_GPT_f_1_eq_0_range_x_l257_25711

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined_on_Rstar : ∀ x : ℝ, ¬ (x = 0) → f x = sorry
axiom f_4_eq_1 : f 4 = 1
axiom f_mult : ∀ (x₁ x₂ : ℝ), ¬ (x₁ = 0) → ¬ (x₂ = 0) → f (x₁ * x₂) = f x₁ + f x₂
axiom f_increasing : ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f x₁ < f x₂

theorem f_1_eq_0 : f 1 = 0 := sorry

theorem range_x (x : ℝ) : f (3 * x + 1) + f (2 * x - 6) ≤ 3 → 3 < x ∧ x ≤ 5 := sorry

end NUMINAMATH_GPT_f_1_eq_0_range_x_l257_25711


namespace NUMINAMATH_GPT_temperature_on_wednesday_l257_25726

theorem temperature_on_wednesday
  (T_sunday   : ℕ)
  (T_monday   : ℕ)
  (T_tuesday  : ℕ)
  (T_thursday : ℕ)
  (T_friday   : ℕ)
  (T_saturday : ℕ)
  (average_temperature : ℕ)
  (h_sunday   : T_sunday = 40)
  (h_monday   : T_monday = 50)
  (h_tuesday  : T_tuesday = 65)
  (h_thursday : T_thursday = 82)
  (h_friday   : T_friday = 72)
  (h_saturday : T_saturday = 26)
  (h_avg_temp : (T_sunday + T_monday + T_tuesday + W + T_thursday + T_friday + T_saturday) / 7 = average_temperature)
  (h_avg_val  : average_temperature = 53) :
  W = 36 :=
by { sorry }

end NUMINAMATH_GPT_temperature_on_wednesday_l257_25726


namespace NUMINAMATH_GPT_sum_first_20_terms_l257_25719

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the conditions stated in the problem
variables {a : ℕ → ℤ}
variables (h_arith : is_arithmetic_sequence a)
variables (h_sum_first_three : a 1 + a 2 + a 3 = -24)
variables (h_sum_18_to_20 : a 18 + a 19 + a 20 = 78)

-- State the theorem to prove
theorem sum_first_20_terms : (Finset.range 20).sum a = 180 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_20_terms_l257_25719


namespace NUMINAMATH_GPT_triangle_ABC_properties_l257_25734

theorem triangle_ABC_properties 
  (a b c : ℝ) (A B C : ℝ)
  (h1 : a * Real.sin C = Real.sqrt 3 * c * Real.cos A)
  (h2 : a = Real.sqrt 13)
  (h3 : c = 3)
  (h_angle_range : A > 0 ∧ A < Real.pi) : 
  A = Real.pi / 3 ∧ (1 / 2) * b * c * Real.sin A = 3 * Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_triangle_ABC_properties_l257_25734


namespace NUMINAMATH_GPT_negation_of_proposition_l257_25758

theorem negation_of_proposition:
  (∀ x : ℝ, x ≥ 0 → x - 2 > 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x - 2 ≤ 0) := 
sorry

end NUMINAMATH_GPT_negation_of_proposition_l257_25758


namespace NUMINAMATH_GPT_cistern_emptied_fraction_l257_25732

variables (minutes : ℕ) (fractionA fractionB fractionC : ℚ)

def pipeA_rate := 1 / 2 / 12
def pipeB_rate := 1 / 3 / 15
def pipeC_rate := 1 / 4 / 20

def time_active := 5

def emptiedA := pipeA_rate * time_active
def emptiedB := pipeB_rate * time_active
def emptiedC := pipeC_rate * time_active

def total_emptied := emptiedA + emptiedB + emptiedC

theorem cistern_emptied_fraction :
  total_emptied = 55 / 144 := by
  sorry

end NUMINAMATH_GPT_cistern_emptied_fraction_l257_25732


namespace NUMINAMATH_GPT_angle_complement_half_supplement_is_zero_l257_25707

theorem angle_complement_half_supplement_is_zero (x : ℝ) 
  (h_complement: x - 90 = (1 / 2) * (x - 180)) : x = 0 := 
sorry

end NUMINAMATH_GPT_angle_complement_half_supplement_is_zero_l257_25707


namespace NUMINAMATH_GPT_domain_of_f_l257_25736

noncomputable def f (x: ℝ): ℝ := 1 / Real.sqrt (x - 2)

theorem domain_of_f:
  {x: ℝ | 2 < x} = {x: ℝ | f x = 1 / Real.sqrt (x - 2)} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l257_25736


namespace NUMINAMATH_GPT_baguettes_sold_third_batch_l257_25716

-- Definitions of the conditions
def daily_batches : ℕ := 3
def baguettes_per_batch : ℕ := 48
def baguettes_sold_first_batch : ℕ := 37
def baguettes_sold_second_batch : ℕ := 52
def baguettes_left : ℕ := 6

theorem baguettes_sold_third_batch : 
  daily_batches * baguettes_per_batch - (baguettes_sold_first_batch + baguettes_sold_second_batch + baguettes_left) = 49 :=
by sorry

end NUMINAMATH_GPT_baguettes_sold_third_batch_l257_25716


namespace NUMINAMATH_GPT_bugs_initial_count_l257_25786

theorem bugs_initial_count (B : ℝ) 
  (h_spray : ∀ (b : ℝ), b * 0.8 = b * (4 / 5)) 
  (h_spiders : ∀ (s : ℝ), s * 7 = 12 * 7) 
  (h_initial_spray_spiders : ∀ (b : ℝ), b * 0.8 - (12 * 7) = 236) 
  (h_final_bugs : 320 / 0.8 = 400) : 
  B = 400 :=
sorry

end NUMINAMATH_GPT_bugs_initial_count_l257_25786


namespace NUMINAMATH_GPT_trade_and_unification_effects_l257_25777

theorem trade_and_unification_effects :
  let country_A_corn := 8
  let country_B_eggplants := 18
  let country_B_corn := 12
  let country_A_eggplants := 10
  
  -- Part (a): Absolute and comparative advantages
  (country_B_corn > country_A_corn) ∧ (country_B_eggplants > country_A_eggplants) ∧
  let opportunity_cost_A_eggplants := country_A_corn / country_A_eggplants
  let opportunity_cost_A_corn := country_A_eggplants / country_A_corn
  let opportunity_cost_B_eggplants := country_B_corn / country_B_eggplants
  let opportunity_cost_B_corn := country_B_eggplants / country_B_corn
  (opportunity_cost_B_eggplants < opportunity_cost_A_eggplants) ∧ (opportunity_cost_A_corn < opportunity_cost_B_corn) ∧

  -- Part (b): Volumes produced and consumed with trade
  let price := 1
  let income_A := country_A_corn * price
  let income_B := country_B_eggplants * price
  let consumption_A_eggplants := income_A / price / 2
  let consumption_A_corn := country_A_corn / 2
  let consumption_B_corn := income_B / price / 2
  let consumption_B_eggplants := country_B_eggplants / 2
  (consumption_A_eggplants = 4) ∧ (consumption_A_corn = 4) ∧
  (consumption_B_corn = 9) ∧ (consumption_B_eggplants = 9) ∧

  -- Part (c): Volumes after unification without trade
  let unified_eggplants := 18 - (1.5 * 4)
  let unified_corn := 8 + 4
  let total_unified_eggplants := unified_eggplants
  let total_unified_corn := unified_corn
  (total_unified_eggplants = 12) ∧ (total_unified_corn = 12) ->
  
  total_unified_eggplants = 12 ∧ total_unified_corn = 12 ∧
  (total_unified_eggplants < (consumption_A_eggplants + consumption_B_eggplants)) ∧
  (total_unified_corn < (consumption_A_corn + consumption_B_corn))
:= by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_trade_and_unification_effects_l257_25777


namespace NUMINAMATH_GPT_solve_trig_eq_l257_25795

open Real -- Open real number structure

theorem solve_trig_eq (x : ℝ) :
  (sin x)^2 + (sin (2 * x))^2 + (sin (3 * x))^2 = 2 ↔ 
  (∃ n : ℤ, x = π / 4 + (π * n) / 2)
  ∨ (∃ n : ℤ, x = π / 2 + π * n)
  ∨ (∃ n : ℤ, x = π / 6 + π * n ∨ x = -π / 6 + π * n) := by sorry

end NUMINAMATH_GPT_solve_trig_eq_l257_25795


namespace NUMINAMATH_GPT_solve_system_l257_25747

theorem solve_system (x y : ℚ) 
  (h1 : 3 * (x - 1) = y + 6) 
  (h2 : x / 2 + y / 3 = 2) : 
  x = 10 / 3 ∧ y = 1 := 
by 
  sorry

end NUMINAMATH_GPT_solve_system_l257_25747


namespace NUMINAMATH_GPT_ivan_spent_fraction_l257_25749

theorem ivan_spent_fraction (f : ℝ) (h1 : 10 - 10 * f - 5 = 3) : f = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ivan_spent_fraction_l257_25749


namespace NUMINAMATH_GPT_hyperbola_standard_equation_l257_25733

theorem hyperbola_standard_equation :
  (∃ c : ℝ, c = Real.sqrt 5) →
  (∃ a b : ℝ, b / a = 2 ∧ a ^ 2 + b ^ 2 = 5) →
  (∃ a b : ℝ, a = 1 ∧ b = 2 ∧ (x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_standard_equation_l257_25733


namespace NUMINAMATH_GPT_hyperbola_real_axis_length_l257_25784

theorem hyperbola_real_axis_length (x y : ℝ) :
  x^2 - y^2 / 9 = 1 → 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_real_axis_length_l257_25784


namespace NUMINAMATH_GPT_correct_statements_l257_25713

def studentsPopulation : Nat := 70000
def sampleSize : Nat := 1000
def isSamplePopulation (s : Nat) (p : Nat) : Prop := s < p
def averageSampleEqualsPopulation (sampleAvg populationAvg : ℕ) : Prop := sampleAvg = populationAvg
def isPopulation (p : Nat) : Prop := p = studentsPopulation

theorem correct_statements (p s : ℕ) (h1 : isSamplePopulation s p) (h2 : isPopulation p) 
  (h4 : s = sampleSize) : 
  (isSamplePopulation s p ∧ ¬averageSampleEqualsPopulation 1 1 ∧ isPopulation p ∧ s = sampleSize) := 
by
  sorry

end NUMINAMATH_GPT_correct_statements_l257_25713


namespace NUMINAMATH_GPT_video_files_count_l257_25739

-- Definitions for the given conditions
def total_files : ℝ := 48.0
def music_files : ℝ := 4.0
def picture_files : ℝ := 23.0

-- The proposition to prove
theorem video_files_count : total_files - (music_files + picture_files) = 21.0 :=
by
  sorry

end NUMINAMATH_GPT_video_files_count_l257_25739


namespace NUMINAMATH_GPT_encyclopedia_pages_count_l257_25709

theorem encyclopedia_pages_count (digits_used : ℕ) (h : digits_used = 6869) : ∃ pages : ℕ, pages = 1994 :=
by 
  sorry

end NUMINAMATH_GPT_encyclopedia_pages_count_l257_25709


namespace NUMINAMATH_GPT_maximum_value_of_d_l257_25759

theorem maximum_value_of_d (a b c d : ℝ) 
  (h₁ : a + b + c + d = 10)
  (h₂ : ab + ac + ad + bc + bd + cd = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_d_l257_25759


namespace NUMINAMATH_GPT_typing_speed_ratio_l257_25741

-- Defining the conditions for the problem
def typing_speeds (T M : ℝ) : Prop :=
  (T + M = 12) ∧ (T + 1.25 * M = 14)

-- Stating the theorem with conditions and the expected result
theorem typing_speed_ratio (T M : ℝ) (h : typing_speeds T M) : M / T = 2 :=
by
  cases h
  sorry

end NUMINAMATH_GPT_typing_speed_ratio_l257_25741


namespace NUMINAMATH_GPT_train_length_is_150_meters_l257_25762

def train_speed_kmph : ℝ := 68
def man_speed_kmph : ℝ := 8
def passing_time_sec : ℝ := 8.999280057595392

noncomputable def length_of_train : ℝ :=
  let relative_speed_kmph := train_speed_kmph - man_speed_kmph
  let relative_speed_mps := (relative_speed_kmph * 1000) / 3600
  relative_speed_mps * passing_time_sec

theorem train_length_is_150_meters (train_speed_kmph man_speed_kmph passing_time_sec : ℝ) :
  train_speed_kmph = 68 → man_speed_kmph = 8 → passing_time_sec = 8.999280057595392 →
  length_of_train = 150 :=
by
  intros h1 h2 h3
  simp [length_of_train, h1, h2, h3]
  sorry

end NUMINAMATH_GPT_train_length_is_150_meters_l257_25762


namespace NUMINAMATH_GPT_second_prime_is_23_l257_25704

-- Define the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def x := 69
def p : ℕ := 3
def q : ℕ := 23

-- State the theorem
theorem second_prime_is_23 (h1 : is_prime p) (h2 : 2 < p ∧ p < 6) (h3 : is_prime q) (h4 : x = p * q) : q = 23 := 
by 
  sorry

end NUMINAMATH_GPT_second_prime_is_23_l257_25704


namespace NUMINAMATH_GPT_cost_of_apples_l257_25793

theorem cost_of_apples 
  (total_cost : ℕ)
  (cost_bananas : ℕ)
  (cost_bread : ℕ)
  (cost_milk : ℕ)
  (cost_apples : ℕ)
  (h1 : total_cost = 42)
  (h2 : cost_bananas = 12)
  (h3 : cost_bread = 9)
  (h4 : cost_milk = 7)
  (h5 : total_cost = cost_bananas + cost_bread + cost_milk + cost_apples) :
  cost_apples = 14 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_apples_l257_25793


namespace NUMINAMATH_GPT_total_cost_is_18_l257_25754

-- Definitions based on the conditions
def cost_soda : ℕ := 1
def cost_3_sodas := 3 * cost_soda
def cost_soup := cost_3_sodas
def cost_2_soups := 2 * cost_soup
def cost_sandwich := 3 * cost_soup
def total_cost := cost_3_sodas + cost_2_soups + cost_sandwich

-- The proof statement
theorem total_cost_is_18 : total_cost = 18 := by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_total_cost_is_18_l257_25754


namespace NUMINAMATH_GPT_remainder_when_divided_by_30_l257_25712

theorem remainder_when_divided_by_30 (n k R m : ℤ) (h1 : 0 ≤ R ∧ R < 30) (h2 : 2 * n % 15 = 2) (h3 : n = 30 * k + R) : R = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_30_l257_25712


namespace NUMINAMATH_GPT_hyperbola_focal_length_l257_25748

/--
In the Cartesian coordinate system \( xOy \),
let the focal length of the hyperbola \( \frac{x^{2}}{2m^{2}} - \frac{y^{2}}{3m} = 1 \) be 6.
Prove that the set of all real numbers \( m \) that satisfy this condition is {3/2}.
-/
theorem hyperbola_focal_length (m : ℝ) (h1 : 2 * m^2 > 0) (h2 : 3 * m > 0) (h3 : 2 * m^2 + 3 * m = 9) :
  m = 3 / 2 :=
sorry

end NUMINAMATH_GPT_hyperbola_focal_length_l257_25748


namespace NUMINAMATH_GPT_factorial_expression_evaluation_l257_25790

theorem factorial_expression_evaluation : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + 5 * Nat.factorial 5 = 5760 := 
by 
  sorry

end NUMINAMATH_GPT_factorial_expression_evaluation_l257_25790


namespace NUMINAMATH_GPT_solve_congruence_l257_25744

theorem solve_congruence : ∃ (a m : ℕ), 10 * x + 3 ≡ 7 [MOD 18] ∧ x ≡ a [MOD m] ∧ a < m ∧ m ≥ 2 ∧ a + m = 13 := 
sorry

end NUMINAMATH_GPT_solve_congruence_l257_25744


namespace NUMINAMATH_GPT_feathers_to_cars_ratio_l257_25779

theorem feathers_to_cars_ratio (initial_feathers : ℕ) (final_feathers : ℕ) (cars_dodged : ℕ)
  (h₁ : initial_feathers = 5263) (h₂ : final_feathers = 5217) (h₃ : cars_dodged = 23) :
  (initial_feathers - final_feathers) / cars_dodged = 2 :=
by
  sorry

end NUMINAMATH_GPT_feathers_to_cars_ratio_l257_25779


namespace NUMINAMATH_GPT_reality_show_duration_l257_25785

variable (x : ℕ)

theorem reality_show_duration :
  (5 * x + 10 = 150) → (x = 28) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_reality_show_duration_l257_25785


namespace NUMINAMATH_GPT_contrapositive_of_equality_square_l257_25703

theorem contrapositive_of_equality_square (a b : ℝ) (h : a^2 ≠ b^2) : a ≠ b := 
by 
  sorry

end NUMINAMATH_GPT_contrapositive_of_equality_square_l257_25703


namespace NUMINAMATH_GPT_find_divisor_l257_25760

theorem find_divisor (X : ℕ) (h12 : 12 ∣ (1020 - 12)) (h24 : 24 ∣ (1020 - 12)) (h48 : 48 ∣ (1020 - 12)) (h56 : 56 ∣ (1020 - 12)) :
  X = 63 :=
sorry

end NUMINAMATH_GPT_find_divisor_l257_25760


namespace NUMINAMATH_GPT_polynomial_roots_l257_25742

theorem polynomial_roots :
  Polynomial.roots (3 * X^4 + 11 * X^3 - 28 * X^2 + 10 * X) = {0, 1/3, 2, -5} :=
sorry

end NUMINAMATH_GPT_polynomial_roots_l257_25742


namespace NUMINAMATH_GPT_g_1993_at_4_l257_25717

def g (x : ℚ) : ℚ := (2 + x) / (2 - 4 * x)

def g_n : ℕ → ℚ → ℚ
  | 0, x     => x
  | (n+1), x => g (g_n n x)

theorem g_1993_at_4 : g_n 1993 4 = 11 / 20 :=
by
  sorry

end NUMINAMATH_GPT_g_1993_at_4_l257_25717


namespace NUMINAMATH_GPT_fluorescent_bulbs_switched_on_percentage_l257_25714

theorem fluorescent_bulbs_switched_on_percentage (I F : ℕ) (x : ℝ) (Inc_on F_on total_on Inc_on_ratio : ℝ) 
  (h1 : Inc_on = 0.3 * I) 
  (h2 : total_on = 0.7 * (I + F)) 
  (h3 : Inc_on_ratio = 0.08571428571428571) 
  (h4 : Inc_on_ratio = Inc_on / total_on) 
  (h5 : total_on = Inc_on + F_on) 
  (h6 : F_on = x * F) :
  x = 0.9 :=
sorry

end NUMINAMATH_GPT_fluorescent_bulbs_switched_on_percentage_l257_25714


namespace NUMINAMATH_GPT_price_difference_l257_25799

-- Definitions of conditions
def market_price : ℝ := 15400
def initial_sales_tax_rate : ℝ := 0.076
def new_sales_tax_rate : ℝ := 0.0667
def discount_rate : ℝ := 0.05
def handling_fee : ℝ := 200

-- Calculation of original sales tax
def original_sales_tax_amount : ℝ := market_price * initial_sales_tax_rate
-- Calculation of price after discount
def discount_amount : ℝ := market_price * discount_rate
def price_after_discount : ℝ := market_price - discount_amount
-- Calculation of new sales tax
def new_sales_tax_amount : ℝ := price_after_discount * new_sales_tax_rate
-- Calculation of total price with new sales tax and handling fee
def total_price_new : ℝ := price_after_discount + new_sales_tax_amount + handling_fee
-- Calculation of original total price with handling fee
def original_total_price : ℝ := market_price + original_sales_tax_amount + handling_fee

-- Expected difference in total cost
def expected_difference : ℝ := 964.60

-- Lean 4 statement to prove the difference
theorem price_difference :
  original_total_price - total_price_new = expected_difference :=
by
  sorry

end NUMINAMATH_GPT_price_difference_l257_25799


namespace NUMINAMATH_GPT_longest_diagonal_length_l257_25750

theorem longest_diagonal_length (A : ℝ) (d1 d2 : ℝ) (h1 : A = 150) (h2 : d1 / d2 = 4 / 3) : d1 = 20 :=
by
  -- Skipping the proof here
  sorry

end NUMINAMATH_GPT_longest_diagonal_length_l257_25750


namespace NUMINAMATH_GPT_integer_values_abc_l257_25769

theorem integer_values_abc (a b c : ℤ) :
  1 < a ∧ a < b ∧ b < c ∧ (a - 1) * (b - 1) * (c - 1) ∣ (a * b * c - 1) →
  (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15) :=
by
  sorry

end NUMINAMATH_GPT_integer_values_abc_l257_25769


namespace NUMINAMATH_GPT_min_value_problem_l257_25720

theorem min_value_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 4) :
  (x + 1) * (2 * y + 1) / (x * y) ≥ 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_problem_l257_25720


namespace NUMINAMATH_GPT_coloring_count_in_3x3_grid_l257_25791

theorem coloring_count_in_3x3_grid (n m : ℕ) (h1 : n = 3) (h2 : m = 3) : 
  ∃ count : ℕ, count = 15 ∧ ∀ (cells : Finset (Fin n × Fin m)),
  (cells.card = 3 ∧ ∀ (c1 c2 : Fin n × Fin m), c1 ∈ cells → c2 ∈ cells → c1 ≠ c2 → 
  (c1.fst ≠ c2.fst ∧ c1.snd ≠ c2.snd)) → cells.card ∣ count :=
sorry

end NUMINAMATH_GPT_coloring_count_in_3x3_grid_l257_25791


namespace NUMINAMATH_GPT_incircle_excircle_relation_l257_25796

variables {α : Type*} [LinearOrderedField α]

-- Defining the area expressions and radii
def area_inradius (a b c r : α) : α := (a + b + c) * r / 2
def area_exradius1 (a b c r1 : α) : α := (b + c - a) * r1 / 2
def area_exradius2 (a b c r2 : α) : α := (a + c - b) * r2 / 2
def area_exradius3 (a b c r3 : α) : α := (a + b - c) * r3 / 2

theorem incircle_excircle_relation (a b c r r1 r2 r3 Q : α) 
  (h₁ : Q = area_inradius a b c r)
  (h₂ : Q = area_exradius1 a b c r1)
  (h₃ : Q = area_exradius2 a b c r2)
  (h₄ : Q = area_exradius3 a b c r3) :
  1 / r = 1 / r1 + 1 / r2 + 1 / r3 :=
by 
  sorry

end NUMINAMATH_GPT_incircle_excircle_relation_l257_25796


namespace NUMINAMATH_GPT_heather_total_oranges_l257_25740

-- Define the initial conditions
def initial_oranges : ℝ := 60.0
def additional_oranges : ℝ := 35.0

-- Define the total number of oranges
def total_oranges : ℝ := initial_oranges + additional_oranges

-- State the theorem that needs to be proven
theorem heather_total_oranges : total_oranges = 95.0 := 
by
  sorry

end NUMINAMATH_GPT_heather_total_oranges_l257_25740


namespace NUMINAMATH_GPT_dart_game_solution_l257_25723

theorem dart_game_solution (x y z : ℕ) (h_x : 8 * x + 9 * y + 10 * z = 100) (h_y : x + y + z > 11) :
  (x = 10 ∧ y = 0 ∧ z = 2) ∨ (x = 9 ∧ y = 2 ∧ z = 1) ∨ (x = 8 ∧ y = 4 ∧ z = 0) :=
by
  sorry

end NUMINAMATH_GPT_dart_game_solution_l257_25723


namespace NUMINAMATH_GPT_square_plot_area_l257_25768

theorem square_plot_area (cost_per_foot total_cost : ℕ) (hcost_per_foot : cost_per_foot = 60) (htotal_cost : total_cost = 4080) :
  ∃ (A : ℕ), A = 289 :=
by
  have h : 4 * 60 * 17 = 4080 := by rfl
  have s : 17 = 4080 / (4 * 60) := by sorry
  use 17 ^ 2
  have hsquare : 17 ^ 2 = 289 := by rfl
  exact hsquare

end NUMINAMATH_GPT_square_plot_area_l257_25768


namespace NUMINAMATH_GPT_arithmetic_geometric_seq_sum_5_l257_25737

-- Define the arithmetic-geometric sequence a_n
def a (n : ℕ) : ℤ := sorry

-- Define the sum S_n of the first n terms of the sequence a_n
def S (n : ℕ) : ℤ := sorry

-- Condition: a_1 = 1
axiom a1 : a 1 = 1

-- Condition: a_{n+2} + a_{n+1} - 2 * a_{n} = 0 for all n ∈ ℕ_+
axiom recurrence (n : ℕ) : a (n + 2) + a (n + 1) - 2 * a n = 0

-- Prove that S_5 = 11
theorem arithmetic_geometric_seq_sum_5 : S 5 = 11 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_seq_sum_5_l257_25737


namespace NUMINAMATH_GPT_scramble_words_count_l257_25701

-- Definitions based on the conditions
def alphabet_size : Nat := 25
def alphabet_size_no_B : Nat := 24

noncomputable def num_words_with_B : Nat :=
  let total_without_restriction := 25^1 + 25^2 + 25^3 + 25^4 + 25^5
  let total_without_B := 24^1 + 24^2 + 24^3 + 24^4 + 24^5
  total_without_restriction - total_without_B

-- Lean statement to prove the result
theorem scramble_words_count : num_words_with_B = 1692701 :=
by
  sorry

end NUMINAMATH_GPT_scramble_words_count_l257_25701


namespace NUMINAMATH_GPT_product_eq_one_l257_25780

theorem product_eq_one (a b c : ℝ) (h1 : a^2 + 2 = b^4) (h2 : b^2 + 2 = c^4) (h3 : c^2 + 2 = a^4) : 
  (a^2 - 1) * (b^2 - 1) * (c^2 - 1) = 1 :=
sorry

end NUMINAMATH_GPT_product_eq_one_l257_25780


namespace NUMINAMATH_GPT_ones_digit_sum_l257_25730

theorem ones_digit_sum : 
  (1 + 2 ^ 2023 + 3 ^ 2023 + 4 ^ 2023 + 5 : ℕ) % 10 = 5 := 
by 
  sorry

end NUMINAMATH_GPT_ones_digit_sum_l257_25730


namespace NUMINAMATH_GPT_population_growth_l257_25731

theorem population_growth :
  let scale_factor1 := 1 + 10 / 100
  let scale_factor2 := 1 + 20 / 100
  let k := 2 * 20
  let scale_factor3 := 1 + k / 100
  let combined_scale := scale_factor1 * scale_factor2 * scale_factor3
  (combined_scale - 1) * 100 = 84.8 :=
by
  sorry

end NUMINAMATH_GPT_population_growth_l257_25731


namespace NUMINAMATH_GPT_vertex_in_fourth_quadrant_l257_25776

theorem vertex_in_fourth_quadrant (m : ℝ) (h : m < 0) : 
  (0 < -m) ∧ (-1 < 0) :=
by
  sorry

end NUMINAMATH_GPT_vertex_in_fourth_quadrant_l257_25776


namespace NUMINAMATH_GPT_peter_and_susan_dollars_l257_25783

theorem peter_and_susan_dollars :
  (2 / 5 : Real) + (1 / 4 : Real) = 0.65 := 
by
  sorry

end NUMINAMATH_GPT_peter_and_susan_dollars_l257_25783


namespace NUMINAMATH_GPT_f_x1_plus_f_x2_always_greater_than_zero_l257_25792

theorem f_x1_plus_f_x2_always_greater_than_zero
  {f : ℝ → ℝ}
  (h1 : ∀ x, f (-x) = -f (x + 2))
  (h2 : ∀ x > 1, ∀ y > 1, x < y → f y < f x)
  (h3 : ∃ x₁ x₂ : ℝ, 1 + x₁ * x₂ < x₁ + x₂ ∧ x₁ + x₂ < 2) :
  ∀ x₁ x₂ : ℝ, (1 + x₁ * x₂ < x₁ + x₂ ∧ x₁ + x₂ < 2) → f x₁ + f x₂ > 0 := by
  sorry

end NUMINAMATH_GPT_f_x1_plus_f_x2_always_greater_than_zero_l257_25792


namespace NUMINAMATH_GPT_unique_integer_sum_squares_l257_25764

theorem unique_integer_sum_squares (n : ℤ) (h : ∃ d1 d2 d3 d4 : ℕ, d1 * d2 * d3 * d4 = n ∧ n = d1*d1 + d2*d2 + d3*d3 + d4*d4) : n = 42 := 
sorry

end NUMINAMATH_GPT_unique_integer_sum_squares_l257_25764


namespace NUMINAMATH_GPT_candy_difference_l257_25763

theorem candy_difference (frankie_candies : ℕ) (max_candies : ℕ) (h1 : frankie_candies = 74) (h2 : max_candies = 92) : max_candies - frankie_candies = 18 := by
  sorry

end NUMINAMATH_GPT_candy_difference_l257_25763


namespace NUMINAMATH_GPT_Bran_remaining_payment_l257_25775

theorem Bran_remaining_payment :
  let tuition_fee : ℝ := 90
  let job_income_per_month : ℝ := 15
  let scholarship_percentage : ℝ := 0.30
  let months : ℕ := 3
  let scholarship_amount : ℝ := tuition_fee * scholarship_percentage
  let remaining_after_scholarship : ℝ := tuition_fee - scholarship_amount
  let total_job_income : ℝ := job_income_per_month * months
  let amount_to_pay : ℝ := remaining_after_scholarship - total_job_income
  amount_to_pay = 18 := sorry

end NUMINAMATH_GPT_Bran_remaining_payment_l257_25775


namespace NUMINAMATH_GPT_rectangle_width_l257_25782

-- Define the conditions
def length := 6
def area_triangle := 60
def area_ratio := 2/5

-- The theorem: proving that the width of the rectangle is 4 cm
theorem rectangle_width (w : ℝ) (A_triangle : ℝ) (len : ℝ) 
  (ratio : ℝ) (h1 : A_triangle = 60) (h2 : len = 6) (h3 : ratio = 2 / 5) 
  (h4 : (len * w) / A_triangle = ratio) : 
  w = 4 := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_width_l257_25782


namespace NUMINAMATH_GPT_aang_caught_7_fish_l257_25766

theorem aang_caught_7_fish (A : ℕ) (h_avg : (A + 5 + 12) / 3 = 8) : A = 7 :=
by
  sorry

end NUMINAMATH_GPT_aang_caught_7_fish_l257_25766


namespace NUMINAMATH_GPT_distance_C_to_D_l257_25706

noncomputable def side_length_smaller_square (perimeter : ℝ) : ℝ := perimeter / 4
noncomputable def side_length_larger_square (area : ℝ) : ℝ := Real.sqrt area

theorem distance_C_to_D 
  (perimeter_smaller : ℝ) (area_larger : ℝ) (h1 : perimeter_smaller = 8) (h2 : area_larger = 36) :
  let s_smaller := side_length_smaller_square perimeter_smaller
  let s_larger := side_length_larger_square area_larger 
  let leg1 := s_larger 
  let leg2 := s_larger - 2 * s_smaller 
  Real.sqrt (leg1 ^ 2 + leg2 ^ 2) = 2 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_distance_C_to_D_l257_25706


namespace NUMINAMATH_GPT_divides_p_minus_one_l257_25781

theorem divides_p_minus_one {p a b : ℕ} {n : ℕ} 
  (hp : p ≥ 3) 
  (prime_p : Nat.Prime p )
  (gcd_ab : Nat.gcd a b = 1)
  (hdiv : p ∣ (a ^ (2 ^ n) + b ^ (2 ^ n))) : 
  2 ^ (n + 1) ∣ p - 1 := 
sorry

end NUMINAMATH_GPT_divides_p_minus_one_l257_25781
