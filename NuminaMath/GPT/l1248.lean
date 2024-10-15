import Mathlib

namespace NUMINAMATH_GPT_determine_moles_Al2O3_formed_l1248_124828

noncomputable def initial_moles_Al : ℝ := 10
noncomputable def initial_moles_Fe2O3 : ℝ := 6
noncomputable def balanced_eq (moles_Al moles_Fe2O3 moles_Al2O3 moles_Fe : ℝ) : Prop :=
  2 * moles_Al + moles_Fe2O3 = moles_Al2O3 + 2 * moles_Fe

theorem determine_moles_Al2O3_formed :
  ∃ moles_Al2O3 : ℝ, balanced_eq 10 6 moles_Al2O3 (moles_Al2O3 * 2) ∧ moles_Al2O3 = 5 := 
  by 
  sorry

end NUMINAMATH_GPT_determine_moles_Al2O3_formed_l1248_124828


namespace NUMINAMATH_GPT_circle_to_ellipse_scaling_l1248_124823

theorem circle_to_ellipse_scaling :
  ∀ (x' y' : ℝ), (4 * x')^2 + y'^2 = 16 → x'^2 / 16 + y'^2 / 4 = 1 :=
by
  intro x' y'
  intro h
  sorry

end NUMINAMATH_GPT_circle_to_ellipse_scaling_l1248_124823


namespace NUMINAMATH_GPT_A_E_not_third_l1248_124881

-- Define the runners and their respective positions.
inductive Runner
| A : Runner
| B : Runner
| C : Runner
| D : Runner
| E : Runner
open Runner

variable (position : Runner → Nat)

-- Conditions
axiom A_beats_B : position A < position B
axiom C_beats_D : position C < position D
axiom B_beats_E : position B < position E
axiom D_after_A_before_B : position A < position D ∧ position D < position B

-- Prove that A and E cannot be in third place.
theorem A_E_not_third : position A ≠ 3 ∧ position E ≠ 3 :=
sorry

end NUMINAMATH_GPT_A_E_not_third_l1248_124881


namespace NUMINAMATH_GPT_packages_delivered_by_third_butcher_l1248_124818

theorem packages_delivered_by_third_butcher 
  (x y z : ℕ) 
  (h1 : x = 10) 
  (h2 : y = 7) 
  (h3 : 4 * x + 4 * y + 4 * z = 100) : 
  z = 8 :=
by { sorry }

end NUMINAMATH_GPT_packages_delivered_by_third_butcher_l1248_124818


namespace NUMINAMATH_GPT_minimum_value_l1248_124864

open Real

theorem minimum_value (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 / (y - 2) + y^2 / (x - 2)) ≥ 12 :=
sorry

end NUMINAMATH_GPT_minimum_value_l1248_124864


namespace NUMINAMATH_GPT_leftover_stickers_l1248_124877

-- Definitions for each person's stickers
def ninaStickers : ℕ := 53
def oliverStickers : ℕ := 68
def pattyStickers : ℕ := 29

-- The number of stickers in a package
def packageSize : ℕ := 18

-- The total number of stickers
def totalStickers : ℕ := ninaStickers + oliverStickers + pattyStickers

-- Proof that the number of leftover stickers is 6 when all stickers are divided into packages of 18
theorem leftover_stickers : totalStickers % packageSize = 6 := by
  sorry

end NUMINAMATH_GPT_leftover_stickers_l1248_124877


namespace NUMINAMATH_GPT_polynomial_coeff_sum_abs_l1248_124829

theorem polynomial_coeff_sum_abs (a a_1 a_2 a_3 a_4 a_5 : ℤ) :
    (2 * x - 1)^5 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
    |a_1| + |a_2| + |a_3| + |a_4| + |a_5| = 242 := by 
  sorry

end NUMINAMATH_GPT_polynomial_coeff_sum_abs_l1248_124829


namespace NUMINAMATH_GPT_ounces_per_bowl_l1248_124888

theorem ounces_per_bowl (oz_per_gallon : ℕ) (gallons : ℕ) (bowls_per_minute : ℕ) (minutes : ℕ) (total_ounces : ℕ) (total_bowls : ℕ) (oz_per_bowl : ℕ) : 
  oz_per_gallon = 128 → 
  gallons = 6 →
  bowls_per_minute = 5 →
  minutes = 15 →
  total_ounces = oz_per_gallon * gallons →
  total_bowls = bowls_per_minute * minutes →
  oz_per_bowl = total_ounces / total_bowls →
  round (oz_per_bowl : ℚ) = 10 :=
by
  sorry

end NUMINAMATH_GPT_ounces_per_bowl_l1248_124888


namespace NUMINAMATH_GPT_factor_expression_l1248_124884

theorem factor_expression (c : ℝ) : 270 * c^2 + 45 * c - 15 = 15 * c * (18 * c + 2) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1248_124884


namespace NUMINAMATH_GPT_problem1_problem2_l1248_124814

open Real

-- Proof problem for the first expression
theorem problem1 : 
  (-2^2 * (1 / 4) + 4 / (4/9) + (-1) ^ 2023 = 7) :=
by 
  sorry

-- Proof problem for the second expression
theorem problem2 : 
  (-1 ^ 4 + abs (2 - (-3)^2) + (1/2) / (-3/2) = 17/3) :=
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1248_124814


namespace NUMINAMATH_GPT_y_value_on_line_l1248_124872

theorem y_value_on_line (x y : ℝ) (k : ℝ → ℝ)
  (h1 : k 0 = 0)
  (h2 : ∀ x, k x = (1/5) * x)
  (hx1 : k x = 1)
  (hx2 : k 5 = y) :
  y = 1 :=
sorry

end NUMINAMATH_GPT_y_value_on_line_l1248_124872


namespace NUMINAMATH_GPT_correct_exponent_calculation_l1248_124806

theorem correct_exponent_calculation (a : ℝ) : 
  (a^5 * a^2 = a^7) :=
by
  sorry

end NUMINAMATH_GPT_correct_exponent_calculation_l1248_124806


namespace NUMINAMATH_GPT_John_lost_3_ebook_readers_l1248_124866

-- Definitions based on the conditions
def A : Nat := 50  -- Anna bought 50 eBook readers
def J : Nat := A - 15  -- John bought 15 less than Anna
def total : Nat := 82  -- Total eBook readers now

-- The number of eBook readers John has after the loss:
def J_after_loss : Nat := total - A

-- The number of eBook readers John lost:
def John_loss : Nat := J - J_after_loss

theorem John_lost_3_ebook_readers : John_loss = 3 :=
by
  sorry

end NUMINAMATH_GPT_John_lost_3_ebook_readers_l1248_124866


namespace NUMINAMATH_GPT_sum_of_a_for_quadratic_has_one_solution_l1248_124815

noncomputable def discriminant (a : ℝ) : ℝ := (a + 12)^2 - 4 * 3 * 16

theorem sum_of_a_for_quadratic_has_one_solution : 
  (∀ a : ℝ, discriminant a = 0) → 
  (-12 + 8 * Real.sqrt 3) + (-12 - 8 * Real.sqrt 3) = -24 :=
by
  intros h
  simp [discriminant] at h
  sorry

end NUMINAMATH_GPT_sum_of_a_for_quadratic_has_one_solution_l1248_124815


namespace NUMINAMATH_GPT_smallest_a_l1248_124825

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

theorem smallest_a (a : ℕ) (h1 : is_factor 112 (a * 43 * 62 * 1311)) (h2 : is_factor 33 (a * 43 * 62 * 1311)) : a = 1848 :=
by
  sorry

end NUMINAMATH_GPT_smallest_a_l1248_124825


namespace NUMINAMATH_GPT_woman_away_time_l1248_124854

noncomputable def angle_hour_hand (n : ℝ) : ℝ := 150 + n / 2
noncomputable def angle_minute_hand (n : ℝ) : ℝ := 6 * n

theorem woman_away_time : 
  (∀ n : ℝ, abs (angle_hour_hand n - angle_minute_hand n) = 120) → 
  abs ((540 / 11 : ℝ) - (60 / 11 : ℝ)) = 43.636 :=
by sorry

end NUMINAMATH_GPT_woman_away_time_l1248_124854


namespace NUMINAMATH_GPT_lcm_of_6_8_10_l1248_124852

theorem lcm_of_6_8_10 : Nat.lcm (Nat.lcm 6 8) 10 = 120 := 
  by sorry

end NUMINAMATH_GPT_lcm_of_6_8_10_l1248_124852


namespace NUMINAMATH_GPT_least_positive_integer_remainder_l1248_124838

theorem least_positive_integer_remainder :
  ∃ n : ℕ, (n > 0) ∧ (n % 5 = 1) ∧ (n % 4 = 2) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 5 = 1) ∧ (m % 4 = 2) → n ≤ m) :=
sorry

end NUMINAMATH_GPT_least_positive_integer_remainder_l1248_124838


namespace NUMINAMATH_GPT_union_of_A_B_l1248_124874

def A : Set ℝ := {x | |x - 3| < 2}
def B : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}

theorem union_of_A_B : A ∪ B = {x | -1 ≤ x ∧ x < 5} :=
by
  sorry

end NUMINAMATH_GPT_union_of_A_B_l1248_124874


namespace NUMINAMATH_GPT_find_k_l1248_124856

theorem find_k (k : ℝ) : (1 - 1.5 * k = (k - 2.5) / 3) → k = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_k_l1248_124856


namespace NUMINAMATH_GPT_average_letters_per_day_l1248_124897

theorem average_letters_per_day 
  (letters_tuesday : ℕ)
  (letters_wednesday : ℕ)
  (days : ℕ := 2) 
  (letters_total : ℕ := letters_tuesday + letters_wednesday) :
  letters_tuesday = 7 → letters_wednesday = 3 → letters_total / days = 5 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_average_letters_per_day_l1248_124897


namespace NUMINAMATH_GPT_count_of_valid_four_digit_numbers_l1248_124878

def is_four_digit_number (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9

def digits_sum_to_twelve (a b c d : ℕ) : Prop :=
  a + b + c + d = 12

def divisible_by_eleven (a b c d : ℕ) : Prop :=
  (a + c - (b + d)) % 11 = 0

theorem count_of_valid_four_digit_numbers : ∃ n : ℕ, n = 20 ∧
  (∀ a b c d : ℕ, is_four_digit_number a b c d →
  digits_sum_to_twelve a b c d →
  divisible_by_eleven a b c d →
  true) :=
sorry

end NUMINAMATH_GPT_count_of_valid_four_digit_numbers_l1248_124878


namespace NUMINAMATH_GPT_trapezium_area_correct_l1248_124835

-- Define the lengths of the parallel sides and the distance between them
def a := 24  -- length of the first parallel side in cm
def b := 14  -- length of the second parallel side in cm
def h := 18  -- distance between the parallel sides in cm

-- Define the area calculation function for the trapezium
def trapezium_area (a b h : ℕ) : ℕ :=
  1 / 2 * (a + b) * h

-- The theorem to prove that the area of the given trapezium is 342 square centimeters
theorem trapezium_area_correct : trapezium_area a b h = 342 :=
  sorry

end NUMINAMATH_GPT_trapezium_area_correct_l1248_124835


namespace NUMINAMATH_GPT_students_study_both_l1248_124834

-- Define variables and conditions
variable (total_students G B G_and_B : ℕ)
variable (G_percent B_percent : ℝ)
variable (total_students_eq : total_students = 300)
variable (G_percent_eq : G_percent = 0.8)
variable (B_percent_eq : B_percent = 0.5)
variable (G_eq : G = G_percent * total_students)
variable (B_eq : B = B_percent * total_students)
variable (students_eq : total_students = G + B - G_and_B)

-- Theorem statement
theorem students_study_both :
  G_and_B = 90 :=
by
  sorry

end NUMINAMATH_GPT_students_study_both_l1248_124834


namespace NUMINAMATH_GPT_sandwich_cost_l1248_124889

theorem sandwich_cost (total_cost soda_cost sandwich_count soda_count : ℝ) :
  total_cost = 8.38 → soda_cost = 0.87 → sandwich_count = 2 → soda_count = 4 → 
  (∀ S, sandwich_count * S + soda_count * soda_cost = total_cost → S = 2.45) :=
by
  intros h_total h_soda h_sandwich_count h_soda_count S h_eqn
  sorry

end NUMINAMATH_GPT_sandwich_cost_l1248_124889


namespace NUMINAMATH_GPT_common_root_l1248_124882

variable (m x : ℝ)
variable (h₁ : m * x - 1000 = 1021)
variable (h₂ : 1021 * x = m - 1000 * x)

theorem common_root (hx : m * x - 1000 = 1021 ∧ 1021 * x = m - 1000 * x) : m = 2021 ∨ m = -2021 := sorry

end NUMINAMATH_GPT_common_root_l1248_124882


namespace NUMINAMATH_GPT_sampling_method_is_systematic_l1248_124891

def conveyor_belt_sampling (interval: ℕ) (product_picking: ℕ → ℕ) : Prop :=
  ∀ (n: ℕ), product_picking n = n * interval

theorem sampling_method_is_systematic
  (interval: ℕ)
  (product_picking: ℕ → ℕ)
  (h: conveyor_belt_sampling interval product_picking) :
  interval = 30 → product_picking = systematic_sampling := 
sorry

end NUMINAMATH_GPT_sampling_method_is_systematic_l1248_124891


namespace NUMINAMATH_GPT_correct_operation_B_l1248_124837

variable (a : ℝ)

theorem correct_operation_B :
  2 * a^2 * a^4 = 2 * a^6 :=
by sorry

end NUMINAMATH_GPT_correct_operation_B_l1248_124837


namespace NUMINAMATH_GPT_alvin_marble_count_correct_l1248_124801

variable (initial_marble_count lost_marble_count won_marble_count final_marble_count : ℕ)

def calculate_final_marble_count (initial : ℕ) (lost : ℕ) (won : ℕ) : ℕ :=
  initial - lost + won

theorem alvin_marble_count_correct :
  initial_marble_count = 57 →
  lost_marble_count = 18 →
  won_marble_count = 25 →
  final_marble_count = calculate_final_marble_count initial_marble_count lost_marble_count won_marble_count →
  final_marble_count = 64 :=
by
  intros h_initial h_lost h_won h_calculate
  rw [h_initial, h_lost, h_won] at h_calculate
  exact h_calculate

end NUMINAMATH_GPT_alvin_marble_count_correct_l1248_124801


namespace NUMINAMATH_GPT_percentage_increase_20_l1248_124804

noncomputable def oldCompanyEarnings : ℝ := 3 * 12 * 5000
noncomputable def totalEarnings : ℝ := 426000
noncomputable def newCompanyMonths : ℕ := 36 + 5
noncomputable def newCompanyEarnings : ℝ := totalEarnings - oldCompanyEarnings
noncomputable def newCompanyMonthlyEarnings : ℝ := newCompanyEarnings / newCompanyMonths
noncomputable def oldCompanyMonthlyEarnings : ℝ := 5000

theorem percentage_increase_20 :
  (newCompanyMonthlyEarnings - oldCompanyMonthlyEarnings) / oldCompanyMonthlyEarnings * 100 = 20 :=
by sorry

end NUMINAMATH_GPT_percentage_increase_20_l1248_124804


namespace NUMINAMATH_GPT_number_of_int_pairs_l1248_124839

theorem number_of_int_pairs (x y : ℤ) (h : x^2 + 2 * y^2 < 25) : 
  ∃ S : Finset (ℤ × ℤ), S.card = 55 ∧ ∀ (a : ℤ × ℤ), a ∈ S ↔ a.1^2 + 2 * a.2^2 < 25 :=
sorry

end NUMINAMATH_GPT_number_of_int_pairs_l1248_124839


namespace NUMINAMATH_GPT_find_distance_to_school_l1248_124820

variable (v d : ℝ)
variable (h_rush_hour : d = v * (1 / 2))
variable (h_no_traffic : d = (v + 20) * (1 / 4))

theorem find_distance_to_school (h_rush_hour : d = v * (1 / 2)) (h_no_traffic : d = (v + 20) * (1 / 4)) : d = 10 := by
  sorry

end NUMINAMATH_GPT_find_distance_to_school_l1248_124820


namespace NUMINAMATH_GPT_negation_of_proposition_l1248_124810

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*x > 0) ↔ (∀ x : ℝ, x^2 + 2*x ≤ 0) :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l1248_124810


namespace NUMINAMATH_GPT_max_candy_remainder_l1248_124870

theorem max_candy_remainder (x : ℕ) : x % 11 < 11 ∧ (∀ r : ℕ, r < 11 → x % 11 ≤ r) → x % 11 = 10 := 
sorry

end NUMINAMATH_GPT_max_candy_remainder_l1248_124870


namespace NUMINAMATH_GPT_shopkeeper_milk_sold_l1248_124842

theorem shopkeeper_milk_sold :
  let morning_packets := 150
  let morning_250 := 60
  let morning_300 := 40
  let morning_350 := morning_packets - morning_250 - morning_300
  
  let evening_packets := 100
  let evening_400 := evening_packets * 50 / 100
  let evening_500 := evening_packets * 25 / 100
  let evening_450 := evening_packets * 25 / 100

  let morning_milk := morning_250 * 250 + morning_300 * 300 + morning_350 * 350
  let evening_milk := evening_400 * 400 + evening_500 * 500 + evening_450 * 450
  let total_milk := morning_milk + evening_milk

  let remaining_milk := 42000
  let sold_milk := total_milk - remaining_milk

  let ounces_per_mil := 1 / 30
  let sold_milk_ounces := sold_milk * ounces_per_mil

  sold_milk_ounces = 1541.67 := by sorry

end NUMINAMATH_GPT_shopkeeper_milk_sold_l1248_124842


namespace NUMINAMATH_GPT_exp_log_pb_eq_log_ba_l1248_124832

noncomputable def log_b (b a : ℝ) := Real.log a / Real.log b

theorem exp_log_pb_eq_log_ba (a b p : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : p = log_b b (log_b b a) / log_b b a) :
  a^p = log_b b a :=
by
  sorry

end NUMINAMATH_GPT_exp_log_pb_eq_log_ba_l1248_124832


namespace NUMINAMATH_GPT_regression_line_passes_through_center_l1248_124817

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := 1.5 * x - 15

-- Define the condition of the sample center point
def sample_center (x_bar y_bar : ℝ) : Prop :=
  y_bar = regression_eq x_bar

-- The proof goal
theorem regression_line_passes_through_center (x_bar y_bar : ℝ) (h : sample_center x_bar y_bar) :
  y_bar = 1.5 * x_bar - 15 :=
by
  -- Using the given condition as hypothesis
  exact h

end NUMINAMATH_GPT_regression_line_passes_through_center_l1248_124817


namespace NUMINAMATH_GPT_royWeight_l1248_124844

-- Define the problem conditions
def johnWeight : ℕ := 81
def johnHeavierBy : ℕ := 77

-- Define the main proof problem
theorem royWeight : (johnWeight - johnHeavierBy) = 4 := by
  sorry

end NUMINAMATH_GPT_royWeight_l1248_124844


namespace NUMINAMATH_GPT_ex_sq_sum_l1248_124869

theorem ex_sq_sum (x y : ℝ) (h1 : (x + y)^2 = 9) (h2 : x * y = -1) : x^2 + y^2 = 11 :=
by
  sorry

end NUMINAMATH_GPT_ex_sq_sum_l1248_124869


namespace NUMINAMATH_GPT_modulus_of_complex_number_l1248_124800

noncomputable def z := Complex

theorem modulus_of_complex_number (z : Complex) (h : z * (1 + Complex.I) = 2) :
  Complex.abs z = Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_modulus_of_complex_number_l1248_124800


namespace NUMINAMATH_GPT_quadratic_root_relationship_l1248_124861

noncomputable def roots_of_quadratic (a b c: ℚ) (h_nonzero: a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h_root_relation: ∀ (s₁ s₂ : ℚ), s₁ + s₂ = -c ∧ s₁ * s₂ = a → (3 * s₁) + (3 * s₂) = -a ∧ (3 * s₁) * (3 * s₂) = b) : Prop :=
  b / c = 27

theorem quadratic_root_relationship (a b c : ℚ) (h_nonzero: a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_root_relation: ∀ (s₁ s₂ : ℚ), s₁ + s₂ = -c ∧ s₁ * s₂ = a → (3 * s₁) + (3 * s₂) = -a ∧ (3 * s₁) * (3 * s₂) = b) : 
  roots_of_quadratic a b c h_nonzero h_root_relation := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_root_relationship_l1248_124861


namespace NUMINAMATH_GPT_more_newborn_elephants_than_baby_hippos_l1248_124899

-- Define the given conditions
def initial_elephants := 20
def initial_hippos := 35
def female_frac := 5 / 7
def births_per_female_hippo := 5
def total_animals_after_birth := 315

-- Calculate the required values
def female_hippos := female_frac * initial_hippos
def baby_hippos := female_hippos * births_per_female_hippo
def total_animals_before_birth := initial_elephants + initial_hippos
def total_newborns := total_animals_after_birth - total_animals_before_birth
def newborn_elephants := total_newborns - baby_hippos

-- Define the proof statement
theorem more_newborn_elephants_than_baby_hippos :
  (newborn_elephants - baby_hippos) = 10 :=
by
  sorry

end NUMINAMATH_GPT_more_newborn_elephants_than_baby_hippos_l1248_124899


namespace NUMINAMATH_GPT_positive_integer_perfect_square_l1248_124896

theorem positive_integer_perfect_square (n : ℕ) (h1: n > 0) (h2 : ∃ k : ℕ, n^2 - 19 * n - 99 = k^2) : n = 199 :=
sorry

end NUMINAMATH_GPT_positive_integer_perfect_square_l1248_124896


namespace NUMINAMATH_GPT_sally_initial_cards_l1248_124873

variable (initial_cards : ℕ)

-- Define the conditions
def cards_given := 41
def cards_lost := 20
def cards_now := 48

-- Define the proof problem
theorem sally_initial_cards :
  initial_cards + cards_given - cards_lost = cards_now → initial_cards = 27 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sally_initial_cards_l1248_124873


namespace NUMINAMATH_GPT_cost_of_downloading_360_songs_in_2005_is_144_dollars_l1248_124860

theorem cost_of_downloading_360_songs_in_2005_is_144_dollars :
  (∀ (c_2004 c_2005 : ℕ), (∀ c : ℕ, c_2005 = c ∧ c_2004 = c + 32) →
  200 * c_2004 = 360 * c_2005 → 360 * c_2005 / 100 = 144) :=
  by sorry

end NUMINAMATH_GPT_cost_of_downloading_360_songs_in_2005_is_144_dollars_l1248_124860


namespace NUMINAMATH_GPT_derivative_of_f_l1248_124858

def f (x : ℝ) : ℝ := 2 * x + 3

theorem derivative_of_f :
  ∀ x : ℝ, (deriv f x) = 2 :=
by 
  sorry

end NUMINAMATH_GPT_derivative_of_f_l1248_124858


namespace NUMINAMATH_GPT_inequality_proof_l1248_124867

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 1/3) : 
  (1 - a) * (1 - b) ≤ 25/36 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1248_124867


namespace NUMINAMATH_GPT_sum_q_p_is_minus_12_l1248_124862

noncomputable def p (x : ℝ) : ℝ := x^2 - 3 * x + 2

noncomputable def q (x : ℝ) : ℝ := -x^2

theorem sum_q_p_is_minus_12 :
  (q (p 0) + q (p 1) + q (p 2) + q (p 3) + q (p 4)) = -12 :=
by
  sorry

end NUMINAMATH_GPT_sum_q_p_is_minus_12_l1248_124862


namespace NUMINAMATH_GPT_sqrt32_plus_4sqrt_half_minus_sqrt18_l1248_124805

theorem sqrt32_plus_4sqrt_half_minus_sqrt18 :
  (Real.sqrt 32 + 4 * Real.sqrt (1/2) - Real.sqrt 18) = 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_sqrt32_plus_4sqrt_half_minus_sqrt18_l1248_124805


namespace NUMINAMATH_GPT_circle_radius_three_points_on_line_l1248_124821

theorem circle_radius_three_points_on_line :
  ∀ R : ℝ,
  (∀ x y : ℝ, (x - 1)^2 + (y + 1)^2 = R^2 → (4 * x + 3 * y = 11) → (dist (x, y) (1, -1) = 1)) →
  R = 3
:= sorry

end NUMINAMATH_GPT_circle_radius_three_points_on_line_l1248_124821


namespace NUMINAMATH_GPT_total_nickels_l1248_124849

-- Definition of the number of original nickels Mary had
def original_nickels := 7

-- Definition of the number of nickels her dad gave her
def added_nickels := 5

-- Prove that the total number of nickels Mary has now is 12
theorem total_nickels : original_nickels + added_nickels = 12 := by
  sorry

end NUMINAMATH_GPT_total_nickels_l1248_124849


namespace NUMINAMATH_GPT_Ian_hourly_wage_l1248_124893

variable (hours_worked : ℕ)
variable (money_left : ℕ)
variable (hourly_wage : ℕ)

theorem Ian_hourly_wage :
  hours_worked = 8 ∧
  money_left = 72 ∧
  hourly_wage = 18 →
  2 * money_left = hours_worked * hourly_wage :=
by
  intros
  sorry

end NUMINAMATH_GPT_Ian_hourly_wage_l1248_124893


namespace NUMINAMATH_GPT_find_x_l1248_124885

variable (n : ℝ) (x : ℝ)

theorem find_x (h1 : n = 15.0) (h2 : 3 * n - x = 40) : x = 5.0 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1248_124885


namespace NUMINAMATH_GPT_find_angle_D_l1248_124843

theorem find_angle_D (A B C D : ℝ)
  (h1 : A + B = 180)
  (h2 : C = 2 * D)
  (h3 : B = C + 40) : D = 70 := by
  sorry

end NUMINAMATH_GPT_find_angle_D_l1248_124843


namespace NUMINAMATH_GPT_problem_l1248_124847

theorem problem {x y n : ℝ} 
  (h1 : 2 * x + y = 4) 
  (h2 : (x + y) / 3 = 1) 
  (h3 : x + 2 * y = n) : n = 5 := 
sorry

end NUMINAMATH_GPT_problem_l1248_124847


namespace NUMINAMATH_GPT_intersection_correct_l1248_124892

def A : Set ℝ := { x : ℝ | 1 ≤ x ∧ x ≤ 3 }
def B : Set ℝ := { x : ℝ | 2 < x ∧ x < 4 }

theorem intersection_correct : A ∩ B = { x : ℝ | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_correct_l1248_124892


namespace NUMINAMATH_GPT_no_fixed_points_l1248_124822

def f (x a : ℝ) : ℝ := x^2 + 2*a*x + 1

theorem no_fixed_points (a : ℝ) :
  (∀ x : ℝ, f x a ≠ x) ↔ (-1/2 < a ∧ a < 3/2) := by
    sorry

end NUMINAMATH_GPT_no_fixed_points_l1248_124822


namespace NUMINAMATH_GPT_train_speed_proof_l1248_124819

noncomputable def train_speed_kmh (length_train : ℝ) (time_crossing : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  let relative_speed := length_train / time_crossing
  let train_speed_ms := relative_speed - man_speed_ms
  train_speed_ms * (3600 / 1000)

theorem train_speed_proof :
  train_speed_kmh 150 8 7 = 60.5 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_proof_l1248_124819


namespace NUMINAMATH_GPT_quadratic_form_rewrite_l1248_124898

theorem quadratic_form_rewrite (x : ℝ) : 2 * x ^ 2 + 7 = 4 * x → 2 * x ^ 2 - 4 * x + 7 = 0 :=
by
    intro h
    linarith

end NUMINAMATH_GPT_quadratic_form_rewrite_l1248_124898


namespace NUMINAMATH_GPT_find_m_l1248_124886

theorem find_m (a b c d : ℕ) (m : ℕ) (a_n b_n c_n d_n: ℕ → ℕ)
  (ha : ∀ n, a_n n = a * n + b)
  (hb : ∀ n, b_n n = c * n + d)
  (hc : ∀ n, c_n n = a_n n * b_n n)
  (hd : ∀ n, d_n n = c_n (n + 1) - c_n n)
  (ha1b1 : m = a_n 1 * b_n 1)
  (hca2b2 : a_n 2 * b_n 2 = 4)
  (hca3b3 : a_n 3 * b_n 3 = 8)
  (hca4b4 : a_n 4 * b_n 4 = 16) :
  m = 4 := 
by sorry

end NUMINAMATH_GPT_find_m_l1248_124886


namespace NUMINAMATH_GPT_complement_union_eq_l1248_124879

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_eq_l1248_124879


namespace NUMINAMATH_GPT_ageOfX_l1248_124845

def threeYearsAgo (x y : ℕ) := x - 3 = 2 * (y - 3)
def sevenYearsHence (x y : ℕ) := (x + 7) + (y + 7) = 83

theorem ageOfX (x y : ℕ) (h1 : threeYearsAgo x y) (h2 : sevenYearsHence x y) : x = 45 := by
  sorry

end NUMINAMATH_GPT_ageOfX_l1248_124845


namespace NUMINAMATH_GPT_parallel_lines_direction_vector_l1248_124887

theorem parallel_lines_direction_vector (k : ℝ) :
  (∃ c : ℝ, (5, -3) = (c * -2, c * k)) ↔ k = 6 / 5 :=
by sorry

end NUMINAMATH_GPT_parallel_lines_direction_vector_l1248_124887


namespace NUMINAMATH_GPT_max_value_of_expression_l1248_124812

theorem max_value_of_expression (x y z : ℤ) 
  (h1 : x * y + x + y = 20) 
  (h2 : y * z + y + z = 6) 
  (h3 : x * z + x + z = 2) : 
  x^2 + y^2 + z^2 ≤ 84 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l1248_124812


namespace NUMINAMATH_GPT_num_possible_radii_l1248_124802

theorem num_possible_radii:
  ∃ (S : Finset ℕ), 
  (∀ r ∈ S, r < 60 ∧ (2 * r * π ∣ 120 * π)) ∧ 
  S.card = 11 := 
sorry

end NUMINAMATH_GPT_num_possible_radii_l1248_124802


namespace NUMINAMATH_GPT_geom_sequence_next_term_l1248_124863

def geom_seq (a r : ℕ → ℤ) (i : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n * r i

theorem geom_sequence_next_term (y : ℤ) (a : ℕ → ℤ) (r : ℕ → ℤ) (n : ℕ) : 
  geom_seq a r 0 →
  a 0 = 3 →
  a 1 = 9 * y^2 →
  a 2 = 27 * y^4 →
  a 3 = 81 * y^6 →
  r 0 = 3 * y^2 →
  a 4 = 243 * y^8 :=
by
  intro h_seq h1 h2 h3 h4 hr
  sorry

end NUMINAMATH_GPT_geom_sequence_next_term_l1248_124863


namespace NUMINAMATH_GPT_mrs_hilt_money_left_l1248_124875

theorem mrs_hilt_money_left (initial_money : ℕ) (cost_of_pencil : ℕ) (money_left : ℕ) (h1 : initial_money = 15) (h2 : cost_of_pencil = 11) : money_left = 4 :=
by
  sorry

end NUMINAMATH_GPT_mrs_hilt_money_left_l1248_124875


namespace NUMINAMATH_GPT_trigonometric_identity_l1248_124813

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) : 2 * Real.sin θ + Real.sin θ * Real.cos θ = 2 := by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1248_124813


namespace NUMINAMATH_GPT_cost_price_eq_l1248_124826

variables (x : ℝ)

def f (x : ℝ) : ℝ := x * (1 + 0.30)
def g (x : ℝ) : ℝ := f x * 0.80

theorem cost_price_eq (h : g x = 2080) : x * (1 + 0.30) * 0.80 = 2080 :=
by sorry

end NUMINAMATH_GPT_cost_price_eq_l1248_124826


namespace NUMINAMATH_GPT_min_value_of_sum_squares_l1248_124807

theorem min_value_of_sum_squares (x y z : ℝ) (h : 2 * x + 3 * y + 4 * z = 11) : 
  x^2 + y^2 + z^2 ≥ 121 / 29 := sorry

end NUMINAMATH_GPT_min_value_of_sum_squares_l1248_124807


namespace NUMINAMATH_GPT_matrix_expression_l1248_124865

variable {F : Type} [Field F] {n : Type} [Fintype n] [DecidableEq n]
variable (B : Matrix n n F)

-- Suppose B is invertible
variable [Invertible B]

-- Condition given in the problem
theorem matrix_expression (h : (B - 3 • (1 : Matrix n n F)) * (B - 5 • (1 : Matrix n n F)) = 0) :
  B + 10 • (B⁻¹) = 10 • (B⁻¹) + (32 / 3 : F) • (1 : Matrix n n F) :=
sorry

end NUMINAMATH_GPT_matrix_expression_l1248_124865


namespace NUMINAMATH_GPT_find_excluded_digit_l1248_124831

theorem find_excluded_digit (a b : ℕ) (d : ℕ) (h : a * b = 1024) (ha : a % 10 ≠ d) (hb : b % 10 ≠ d) : 
  ∃ r : ℕ, d = r ∧ r < 10 :=
by 
  sorry

end NUMINAMATH_GPT_find_excluded_digit_l1248_124831


namespace NUMINAMATH_GPT_number_of_sixes_l1248_124895

theorem number_of_sixes
  (total_runs : ℕ)
  (boundaries : ℕ)
  (percent_runs_by_running : ℚ)
  (runs_per_boundary : ℕ)
  (runs_per_six : ℕ)
  (runs_by_running : ℚ)
  (runs_by_boundaries : ℕ)
  (runs_by_sixes : ℕ)
  (number_of_sixes : ℕ)
  (h1 : total_runs = 120)
  (h2 : boundaries = 6)
  (h3 : percent_runs_by_running = 0.6)
  (h4 : runs_per_boundary = 4)
  (h5 : runs_per_six = 6)
  (h6 : runs_by_running = percent_runs_by_running * total_runs)
  (h7 : runs_by_boundaries = boundaries * runs_per_boundary)
  (h8 : runs_by_sixes = total_runs - (runs_by_running + runs_by_boundaries))
  (h9 : number_of_sixes = runs_by_sixes / runs_per_six)
  : number_of_sixes = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sixes_l1248_124895


namespace NUMINAMATH_GPT_mortar_shell_hits_the_ground_at_50_seconds_l1248_124853

noncomputable def mortar_shell_firing_equation (x : ℝ) : ℝ :=
  - (1 / 5) * x^2 + 10 * x

theorem mortar_shell_hits_the_ground_at_50_seconds : 
  ∃ x : ℝ, mortar_shell_firing_equation x = 0 ∧ x = 50 :=
by
  sorry

end NUMINAMATH_GPT_mortar_shell_hits_the_ground_at_50_seconds_l1248_124853


namespace NUMINAMATH_GPT_Tammy_second_day_speed_l1248_124827

variable (v t : ℝ)

/-- This statement represents Tammy's climbing situation -/
theorem Tammy_second_day_speed:
  (t + (t - 2) = 14) ∧
  (v * t + (v + 0.5) * (t - 2) = 52) →
  (v + 0.5 = 4) :=
by
  sorry

end NUMINAMATH_GPT_Tammy_second_day_speed_l1248_124827


namespace NUMINAMATH_GPT_expand_polynomial_l1248_124894

theorem expand_polynomial (x : ℂ) : 
  (1 + x^4) * (1 - x^5) * (1 + x^7) = 1 + x^4 - x^5 + x^7 + x^11 - x^9 - x^12 - x^16 := 
sorry

end NUMINAMATH_GPT_expand_polynomial_l1248_124894


namespace NUMINAMATH_GPT_range_of_t_l1248_124850

noncomputable def a_n (n : ℕ) (t : ℝ) : ℝ := -n + t
noncomputable def b_n (n : ℕ) : ℝ := 3^(n-3)
noncomputable def c_n (n : ℕ) (t : ℝ) : ℝ := 
  let a := a_n n t 
  let b := b_n n
  (a + b) / 2 + (|a - b|) / 2

theorem range_of_t (t : ℝ) (h : ∀ n : ℕ, n > 0 → c_n n t ≥ c_n 3 t) : 10/3 < t ∧ t < 5 :=
    sorry

end NUMINAMATH_GPT_range_of_t_l1248_124850


namespace NUMINAMATH_GPT_pipe_fills_cistern_l1248_124890

theorem pipe_fills_cistern (t : ℕ) (h : t = 5) : 11 * t = 55 :=
by
  sorry

end NUMINAMATH_GPT_pipe_fills_cistern_l1248_124890


namespace NUMINAMATH_GPT_frogs_need_new_pond_l1248_124859

theorem frogs_need_new_pond
  (num_frogs : ℕ) 
  (num_tadpoles : ℕ) 
  (num_survivor_tadpoles : ℕ) 
  (pond_capacity : ℕ) 
  (hc1 : num_frogs = 5)
  (hc2 : num_tadpoles = 3 * num_frogs)
  (hc3 : num_survivor_tadpoles = (2 * num_tadpoles) / 3)
  (hc4 : pond_capacity = 8):
  ((num_frogs + num_survivor_tadpoles) - pond_capacity) = 7 :=
by sorry

end NUMINAMATH_GPT_frogs_need_new_pond_l1248_124859


namespace NUMINAMATH_GPT_alice_safe_paths_l1248_124855

/-
Define the coordinate system and conditions.
-/

def total_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

def paths_through_dangerous_area : ℕ :=
  (total_paths 2 2) * (total_paths 2 1)

def safe_paths : ℕ :=
  total_paths 4 3 - paths_through_dangerous_area

theorem alice_safe_paths : safe_paths = 17 := by
  sorry

end NUMINAMATH_GPT_alice_safe_paths_l1248_124855


namespace NUMINAMATH_GPT_problem_statement_l1248_124876

variable {a : ℕ+ → ℝ} 

theorem problem_statement (h : ∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n) :
  (∀ n : ℕ+, a (n + 1) < a n) ∧ -- Sequence is decreasing (original proposition)
  (∀ (a : ℕ+ → ℝ), (∀ n : ℕ+, a (n + 1) < a n) → (∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n)) ∧ -- Inverse
  ((∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n) → (∀ n : ℕ+, a (n + 1) < a n)) ∧ -- Converse
  ((∀ (a : ℕ+ → ℝ), (∀ n : ℕ+, a (n + 1) < a n) → (∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n))) -- Contrapositive
:= by
  sorry

end NUMINAMATH_GPT_problem_statement_l1248_124876


namespace NUMINAMATH_GPT_trajectory_midpoint_l1248_124803

theorem trajectory_midpoint (P Q M : ℝ × ℝ)
  (hP : P.1^2 + P.2^2 = 1)
  (hQ : Q.1 = 3 ∧ Q.2 = 0)
  (hM : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (2 * M.1 - 3)^2 + 4 * M.2^2 = 1 :=
sorry

end NUMINAMATH_GPT_trajectory_midpoint_l1248_124803


namespace NUMINAMATH_GPT_base5_addition_l1248_124857

theorem base5_addition : 
  (14 : ℕ) + (132 : ℕ) = (101 : ℕ) :=
by {
  sorry
}

end NUMINAMATH_GPT_base5_addition_l1248_124857


namespace NUMINAMATH_GPT_speed_of_second_train_l1248_124883

theorem speed_of_second_train
  (distance : ℝ)
  (speed_fast : ℝ)
  (time_difference : ℝ)
  (v : ℝ)
  (h_distance : distance = 425.80645161290323)
  (h_speed_fast : speed_fast = 75)
  (h_time_difference : time_difference = 4)
  (h_v : v = distance / (distance / speed_fast + time_difference)) :
  v = 44 := 
sorry

end NUMINAMATH_GPT_speed_of_second_train_l1248_124883


namespace NUMINAMATH_GPT_find_positive_integers_l1248_124868

noncomputable def positive_integer_solutions_ineq (x : ℕ) : Prop :=
  x > 0 ∧ (x : ℝ) < 4

theorem find_positive_integers (x : ℕ) : 
  (x > 0 ∧ (↑x - 3)/3 < 7 - 5*(↑x)/3) ↔ positive_integer_solutions_ineq x :=
by
  sorry

end NUMINAMATH_GPT_find_positive_integers_l1248_124868


namespace NUMINAMATH_GPT_journey_speed_second_half_l1248_124846

theorem journey_speed_second_half (total_time : ℝ) (first_half_speed : ℝ) (total_distance : ℝ) (v : ℝ) : 
  total_time = 10 ∧ first_half_speed = 21 ∧ total_distance = 224 →
  v = 24 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_journey_speed_second_half_l1248_124846


namespace NUMINAMATH_GPT_digit_for_divisibility_by_9_l1248_124808

theorem digit_for_divisibility_by_9 (A : ℕ) (hA : A < 10) : 
  (∃ k : ℕ, 83 * 1000 + A * 10 + 5 = 9 * k) ↔ A = 2 :=
by
  sorry

end NUMINAMATH_GPT_digit_for_divisibility_by_9_l1248_124808


namespace NUMINAMATH_GPT_frozen_yogurt_price_l1248_124851

variable (F G S : ℝ) -- Define the variables F, G, S as real numbers

-- Define the conditions given in the problem
variable (h1 : 5 * F + 2 * G + 5 * S = 55)
variable (h2 : S = 5)
variable (h3 : G = 1 / 2 * F)

-- State the proof goal
theorem frozen_yogurt_price : F = 5 :=
by
  sorry

end NUMINAMATH_GPT_frozen_yogurt_price_l1248_124851


namespace NUMINAMATH_GPT_question1_question2_l1248_124836

-- Define required symbols and parameters
variables {x : ℝ} {b c : ℝ}

-- Statement 1: Proving b + c given the conditions on the inequality
theorem question1 (h : ∀ x, -1 < x ∧ x < 3 → 5*x^2 - b*x + c < 0) : b + c = -25 := sorry

-- Statement 2: Proving the solution set for the given inequality
theorem question2 (h : ∀ x, (2 * x - 5) / (x + 4) ≥ 0 → (x ≥ 5 / 2 ∨ x < -4)) : 
  {x | (2 * x - 5) / (x + 4) ≥ 0} = {x | x ≥ 5/2 ∨ x < -4} := sorry

end NUMINAMATH_GPT_question1_question2_l1248_124836


namespace NUMINAMATH_GPT_table_seating_problem_l1248_124830

theorem table_seating_problem 
  (n : ℕ) 
  (label : ℕ → ℕ) 
  (h1 : label 31 = 31) 
  (h2 : label (31 - 17 + n) = 14) 
  (h3 : label (31 + 16) = 7) 
  : n = 41 :=
sorry

end NUMINAMATH_GPT_table_seating_problem_l1248_124830


namespace NUMINAMATH_GPT_evaluate_expression_to_zero_l1248_124880

-- Assuming 'm' is an integer with specific constraints and providing a proof that the expression evaluates to 0 when m = -1
theorem evaluate_expression_to_zero (m : ℤ) (h1 : -2 ≤ m) (h2 : m ≤ 2) (h3 : m ≠ 0) (h4 : m ≠ 1) (h5 : m ≠ 2) (h6 : m ≠ -2) : 
  (m = -1) → ((m / (m - 2) - 4 / (m ^ 2 - 2 * m)) / (m + 2) / (m ^ 2 - m)) = 0 := 
by
  intro hm_eq_neg1
  sorry

end NUMINAMATH_GPT_evaluate_expression_to_zero_l1248_124880


namespace NUMINAMATH_GPT_handshakes_count_l1248_124816

def num_teams : ℕ := 4
def players_per_team : ℕ := 2
def total_players : ℕ := num_teams * players_per_team
def shakeable_players (total : ℕ) : ℕ := total * (total - players_per_team) / 2

theorem handshakes_count :
  shakeable_players total_players = 24 :=
by
  sorry

end NUMINAMATH_GPT_handshakes_count_l1248_124816


namespace NUMINAMATH_GPT_monotonic_when_a_is_neg1_find_extreme_points_l1248_124848

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/3) * x ^ 3 - (1/2) * (a^2 + a + 2) * x ^ 2 + a^2 * (a + 2) * x

theorem monotonic_when_a_is_neg1 :
  ∀ x : ℝ, f x (-1) ≤ f x (-1) :=
sorry

theorem find_extreme_points (a : ℝ) :
  if h : a = -1 ∨ a = 2 then
    True  -- The function is monotonically increasing, no extreme points
  else if h : a < -1 ∨ a > 2 then
    ∃ x_max x_min : ℝ, x_max = a + 2 ∧ x_min = a^2 ∧ (f x_max a ≥ f x a ∧ f x_min a ≤ f x a) 
  else
    ∃ x_max x_min : ℝ, x_max = a^2 ∧ x_min = a + 2 ∧ (f x_max a ≥ f x a ∧ f x_min a ≤ f x a) :=
sorry

end NUMINAMATH_GPT_monotonic_when_a_is_neg1_find_extreme_points_l1248_124848


namespace NUMINAMATH_GPT_total_salary_correct_l1248_124824

-- Define the daily salaries
def owner_salary : ℕ := 20
def manager_salary : ℕ := 15
def cashier_salary : ℕ := 10
def clerk_salary : ℕ := 5
def bagger_salary : ℕ := 3

-- Define the number of employees
def num_owners : ℕ := 1
def num_managers : ℕ := 3
def num_cashiers : ℕ := 5
def num_clerks : ℕ := 7
def num_baggers : ℕ := 9

-- Define the total salary calculation
def total_daily_salary : ℕ :=
  (num_owners * owner_salary) +
  (num_managers * manager_salary) +
  (num_cashiers * cashier_salary) +
  (num_clerks * clerk_salary) +
  (num_baggers * bagger_salary)

-- The theorem we need to prove
theorem total_salary_correct :
  total_daily_salary = 177 :=
by
  -- Proof can be filled in later
  sorry

end NUMINAMATH_GPT_total_salary_correct_l1248_124824


namespace NUMINAMATH_GPT_copy_pages_l1248_124871

theorem copy_pages (total_cents : ℕ) (cost_per_page : ℕ) (h1 : total_cents = 1500) (h2 : cost_per_page = 5) : 
  (total_cents / cost_per_page = 300) :=
sorry

end NUMINAMATH_GPT_copy_pages_l1248_124871


namespace NUMINAMATH_GPT_select_3_products_select_exactly_1_defective_select_at_least_1_defective_l1248_124809

noncomputable def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

namespace ProductInspection

def total_products : Nat := 100
def qualified_products : Nat := 98
def defective_products : Nat := 2

-- Proof Problem 1
theorem select_3_products (h : combination total_products 3 = 161700) : True := by
  trivial

-- Proof Problem 2
theorem select_exactly_1_defective (h : combination defective_products 1 * combination qualified_products 2 = 9506) : True := by
  trivial

-- Proof Problem 3
theorem select_at_least_1_defective (h : combination total_products 3 - combination qualified_products 3 = 9604) : True := by
  trivial

end ProductInspection

end NUMINAMATH_GPT_select_3_products_select_exactly_1_defective_select_at_least_1_defective_l1248_124809


namespace NUMINAMATH_GPT_solve_for_m_l1248_124841

def A := {x : ℝ | x^2 + 3*x - 10 ≤ 0}
def B (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem solve_for_m (m : ℝ) (h : B m ⊆ A) : m < 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_m_l1248_124841


namespace NUMINAMATH_GPT_men_in_group_l1248_124833

theorem men_in_group (A : ℝ) (n : ℕ) (h : n > 0) 
  (inc_avg : ↑n * A + 2 * 32 - (21 + 23) = ↑n * (A + 1)) : n = 20 :=
sorry

end NUMINAMATH_GPT_men_in_group_l1248_124833


namespace NUMINAMATH_GPT_lcm_of_fractions_l1248_124840

-- Definitions based on the problem's conditions
def numerators : List ℕ := [7, 8, 3, 5, 13, 15, 22, 27]
def denominators : List ℕ := [10, 9, 8, 12, 14, 100, 45, 35]

-- LCM and GCD functions for lists of natural numbers
def list_lcm (l : List ℕ) : ℕ := l.foldr lcm 1
def list_gcd (l : List ℕ) : ℕ := l.foldr gcd 0

-- Main proposition
theorem lcm_of_fractions : list_lcm numerators / list_gcd denominators = 13860 :=
by {
  -- to be proven
  sorry
}

end NUMINAMATH_GPT_lcm_of_fractions_l1248_124840


namespace NUMINAMATH_GPT_lcm_nuts_bolts_l1248_124811

theorem lcm_nuts_bolts : Nat.lcm 13 8 = 104 := 
sorry

end NUMINAMATH_GPT_lcm_nuts_bolts_l1248_124811
