import Mathlib

namespace allocation_methods_count_l1971_197188

theorem allocation_methods_count :
  let doctors := 2
  let nurses := 4
  let schools := 2
  let doctors_per_school := 1
  let nurses_per_school := 2
  ∃ (allocation_methods : ℕ), allocation_methods = 12 := 
by
  let doctors := 2
  let nurses := 4
  let schools := 2
  let doctors_per_school := 1
  let nurses_per_school := 2
  use doctors * Nat.choose nurses 2
  sorry

end allocation_methods_count_l1971_197188


namespace intersection_A_B_l1971_197100

def A : Set ℝ := { x | 2 < x ∧ x ≤ 4 }
def B : Set ℝ := { x | -1 < x ∧ x < 3 }

theorem intersection_A_B : A ∩ B = { x | 2 < x ∧ x < 3 } :=
sorry

end intersection_A_B_l1971_197100


namespace third_side_length_l1971_197197

theorem third_side_length (x : ℝ) (h1 : 2 + 4 > x) (h2 : 4 + x > 2) (h3 : x + 2 > 4) : x = 4 :=
by {
  sorry
}

end third_side_length_l1971_197197


namespace max_groups_l1971_197164

theorem max_groups (boys girls : ℕ) (h1 : boys = 120) (h2 : girls = 140) : Nat.gcd boys girls = 20 := 
  by
  rw [h1, h2]
  -- Proof steps would be here
  sorry

end max_groups_l1971_197164


namespace erica_riding_time_is_65_l1971_197152

-- Definition of Dave's riding time
def dave_time : ℕ := 10

-- Definition of Chuck's riding time based on Dave's time
def chuck_time (dave_time : ℕ) : ℕ := 5 * dave_time

-- Definition of Erica's additional riding time calculated as 30% of Chuck's time
def erica_additional_time (chuck_time : ℕ) : ℕ := (30 * chuck_time) / 100

-- Definition of Erica's total riding time as Chuck's time plus her additional time
def erica_total_time (chuck_time : ℕ) (erica_additional_time : ℕ) : ℕ := chuck_time + erica_additional_time

-- The proof problem: Erica's total riding time should be 65 minutes.
theorem erica_riding_time_is_65 : erica_total_time (chuck_time dave_time) (erica_additional_time (chuck_time dave_time)) = 65 :=
by
  -- The proof is skipped here
  sorry

end erica_riding_time_is_65_l1971_197152


namespace find_single_digit_number_l1971_197108

-- Define the given conditions:
def single_digit (A : ℕ) := A < 10
def rounded_down_tens (x : ℕ) (result: ℕ) := (x / 10) * 10 = result

-- Lean statement of the problem:
theorem find_single_digit_number (A : ℕ) (H1 : single_digit A) (H2 : rounded_down_tens (A * 1000 + 567) 2560) : A = 2 :=
sorry

end find_single_digit_number_l1971_197108


namespace maximum_value_of_a_l1971_197137

theorem maximum_value_of_a {x y a : ℝ} (hx : x > 1 / 3) (hy : y > 1) :
  (∀ x y, x > 1 / 3 → y > 1 → 9 * x^2 / (a^2 * (y - 1)) + y^2 / (a^2 * (3 * x - 1)) ≥ 1)
  ↔ a ≤ 2 * Real.sqrt 2 :=
sorry

end maximum_value_of_a_l1971_197137


namespace kids_french_fries_cost_l1971_197198

noncomputable def cost_burger : ℝ := 5
noncomputable def cost_fries : ℝ := 3
noncomputable def cost_soft_drink : ℝ := 3
noncomputable def cost_special_burger_meal : ℝ := 9.50
noncomputable def cost_kids_burger : ℝ := 3
noncomputable def cost_kids_juice_box : ℝ := 2
noncomputable def cost_kids_meal : ℝ := 5
noncomputable def savings : ℝ := 10

noncomputable def total_adult_meal_individual : ℝ := 2 * cost_burger + 2 * cost_fries + 2 * cost_soft_drink
noncomputable def total_adult_meal_deal : ℝ := 2 * cost_special_burger_meal

noncomputable def total_kids_meal_individual (F : ℝ) : ℝ := 2 * cost_kids_burger + 2 * F + 2 * cost_kids_juice_box
noncomputable def total_kids_meal_deal : ℝ := 2 * cost_kids_meal

noncomputable def total_cost_individual (F : ℝ) : ℝ := total_adult_meal_individual + total_kids_meal_individual F
noncomputable def total_cost_deal : ℝ := total_adult_meal_deal + total_kids_meal_deal

theorem kids_french_fries_cost : ∃ F : ℝ, total_cost_individual F - total_cost_deal = savings ∧ F = 3.50 := 
by
  use 3.50
  sorry

end kids_french_fries_cost_l1971_197198


namespace current_population_l1971_197173

def initial_population : ℕ := 4200
def percentage_died : ℕ := 10
def percentage_left : ℕ := 15

theorem current_population (pop : ℕ) (died left : ℕ) 
  (h1 : pop = initial_population) 
  (h2 : died = pop * percentage_died / 100) 
  (h3 : left = (pop - died) * percentage_left / 100) 
  (h4 : ∀ remaining, remaining = pop - died - left) 
  : (pop - died - left) = 3213 := 
by sorry

end current_population_l1971_197173


namespace max_M_correct_l1971_197102

variable (A : ℝ) (x y : ℝ)

axiom A_pos : A > 0

noncomputable def max_M : ℝ :=
if A ≤ 4 then 2 + A / 2 else 2 * Real.sqrt A

theorem max_M_correct : 
  (∀ x y : ℝ, 0 < x → 0 < y → 1/x + 1/y + A/(x + y) ≥ max_M A / Real.sqrt (x * y)) ∧ 
  (A ≤ 4 → max_M A = 2 + A / 2) ∧ 
  (A > 4 → max_M A = 2 * Real.sqrt A) :=
sorry

end max_M_correct_l1971_197102


namespace negation_statement_l1971_197175

variables {a b c : ℝ}

theorem negation_statement (h : a * b * c = 0) : ¬(a = 0 ∨ b = 0 ∨ c = 0) :=
sorry

end negation_statement_l1971_197175


namespace exists_a_max_value_of_four_l1971_197144

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.cos x)^2 + 2 * a * Real.sin x + 3 * a - 1

theorem exists_a_max_value_of_four :
  ∃ a : ℝ, (a = 1) ∧ ∀ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), f a x ≤ 4 := 
sorry

end exists_a_max_value_of_four_l1971_197144


namespace werewolf_eats_per_week_l1971_197107
-- First, we import the necessary libraries

-- We define the conditions using Lean definitions

-- The vampire drains 3 people a week
def vampire_drains_per_week : Nat := 3

-- The total population of the village
def village_population : Nat := 72

-- The number of weeks both can live off the population
def weeks : Nat := 9

-- Prove the number of people the werewolf eats per week (W) given the conditions
theorem werewolf_eats_per_week :
  ∃ W : Nat, vampire_drains_per_week * weeks + weeks * W = village_population ∧ W = 5 :=
by
  sorry

end werewolf_eats_per_week_l1971_197107


namespace eval_expression_l1971_197104

theorem eval_expression : -30 + 12 * (8 / 4)^2 = 18 :=
by
  sorry

end eval_expression_l1971_197104


namespace base_representing_350_as_four_digit_number_with_even_final_digit_l1971_197149

theorem base_representing_350_as_four_digit_number_with_even_final_digit {b : ℕ} :
  b ^ 3 ≤ 350 ∧ 350 < b ^ 4 ∧ (∃ d1 d2 d3 d4, 350 = d1 * b^3 + d2 * b^2 + d3 * b + d4 ∧ d4 % 2 = 0) ↔ b = 6 :=
by sorry

end base_representing_350_as_four_digit_number_with_even_final_digit_l1971_197149


namespace pairs_of_socks_calculation_l1971_197115

variable (num_pairs_socks : ℤ)
variable (cost_per_pair : ℤ := 950) -- in cents
variable (cost_shoes : ℤ := 9200) -- in cents
variable (money_jack_has : ℤ := 4000) -- in cents
variable (money_needed : ℤ := 7100) -- in cents
variable (total_money_needed : ℤ := money_jack_has + money_needed)

theorem pairs_of_socks_calculation (x : ℤ) (h : cost_per_pair * x + cost_shoes = total_money_needed) : x = 2 :=
by
  sorry

end pairs_of_socks_calculation_l1971_197115


namespace lower_denomination_cost_l1971_197160

-- Conditions
def total_stamps : ℕ := 20
def total_cost_cents : ℕ := 706
def high_denomination_stamps : ℕ := 18
def high_denomination_cost : ℕ := 37
def low_denomination_stamps : ℕ := total_stamps - high_denomination_stamps

-- Theorem proving the cost of the lower denomination stamp.
theorem lower_denomination_cost :
  ∃ (x : ℕ), (high_denomination_stamps * high_denomination_cost) + (low_denomination_stamps * x) = total_cost_cents
  ∧ x = 20 :=
by
  use 20
  sorry

end lower_denomination_cost_l1971_197160


namespace proof_problem_l1971_197161

theorem proof_problem (a b : ℝ) (h1 : (5 * a + 2)^(1/3) = 3) (h2 : (3 * a + b - 1)^(1/2) = 4) :
  a = 5 ∧ b = 2 ∧ (3 * a - b + 3)^(1/2) = 4 :=
by
  sorry

end proof_problem_l1971_197161


namespace integer_pairs_summing_to_six_l1971_197167

theorem integer_pairs_summing_to_six :
  ∃ m n : ℤ, m + n + m * n = 6 ∧ ((m = 0 ∧ n = 6) ∨ (m = 6 ∧ n = 0)) :=
by
  sorry

end integer_pairs_summing_to_six_l1971_197167


namespace not_divisor_60_l1971_197145

variable (k : ℤ)
def n : ℤ := k * (k + 1) * (k + 2)

theorem not_divisor_60 
  (h₁ : ∃ k, n = k * (k + 1) * (k + 2) ∧ 5 ∣ n) : ¬(60 ∣ n) := 
sorry

end not_divisor_60_l1971_197145


namespace find_f_1_div_2007_l1971_197157

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_1_div_2007 :
  f 0 = 0 ∧
  (∀ x, f x + f (1 - x) = 1) ∧
  (∀ x, f (x / 5) = f x / 2) ∧
  (∀ x1 x2, 0 ≤ x1 → x1 < x2 → x2 ≤ 1 → f x1 ≤ f x2) →
  f (1 / 2007) = 1 / 32 :=
sorry

end find_f_1_div_2007_l1971_197157


namespace Erik_money_left_l1971_197171

theorem Erik_money_left 
  (init_money : ℝ)
  (loaf_of_bread : ℝ) (n_loaves_of_bread : ℝ)
  (carton_of_orange_juice : ℝ) (n_cartons_of_orange_juice : ℝ)
  (dozen_eggs : ℝ) (n_dozens_of_eggs : ℝ)
  (chocolate_bar : ℝ) (n_chocolate_bars : ℝ)
  (pound_apples : ℝ) (n_pounds_apples : ℝ)
  (pound_grapes : ℝ) (n_pounds_grapes : ℝ)
  (discount_bread_and_eggs : ℝ) (discount_other_items : ℝ)
  (sales_tax : ℝ) :
  n_loaves_of_bread = 3 →
  loaf_of_bread = 3 →
  n_cartons_of_orange_juice = 3 →
  carton_of_orange_juice = 6 →
  n_dozens_of_eggs = 2 →
  dozen_eggs = 4 →
  n_chocolate_bars = 5 →
  chocolate_bar = 2 →
  n_pounds_apples = 4 →
  pound_apples = 1.25 →
  n_pounds_grapes = 1.5 →
  pound_grapes = 2.5 →
  discount_bread_and_eggs = 0.1 →
  discount_other_items = 0.05 →
  sales_tax = 0.06 →
  init_money = 86 →
  (init_money - 
     (n_loaves_of_bread * loaf_of_bread * (1 - discount_bread_and_eggs) + 
      n_cartons_of_orange_juice * carton_of_orange_juice * (1 - discount_other_items) + 
      n_dozens_of_eggs * dozen_eggs * (1 - discount_bread_and_eggs) + 
      n_chocolate_bars * chocolate_bar * (1 - discount_other_items) + 
      n_pounds_apples * pound_apples * (1 - discount_other_items) + 
      n_pounds_grapes * pound_grapes * (1 - discount_other_items)) * (1 + sales_tax)) = 32.78 :=
by
  sorry

end Erik_money_left_l1971_197171


namespace solve_f_435_l1971_197151

variable (f : ℝ → ℝ)

-- Conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_period : ∀ x : ℝ, f (3 - x) = f x

-- To Prove
theorem solve_f_435 : f 435 = 0 :=
by
  sorry

end solve_f_435_l1971_197151


namespace tan_435_eq_2_add_sqrt_3_l1971_197196

theorem tan_435_eq_2_add_sqrt_3 :
  Real.tan (435 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  sorry

end tan_435_eq_2_add_sqrt_3_l1971_197196


namespace car_tank_capacity_is_12_gallons_l1971_197181

noncomputable def truck_tank_capacity : ℕ := 20
noncomputable def truck_tank_half_full : ℕ := truck_tank_capacity / 2
noncomputable def car_tank_third_full (car_tank_capacity : ℕ) : ℕ := car_tank_capacity / 3
noncomputable def total_gallons_added : ℕ := 18

theorem car_tank_capacity_is_12_gallons (car_tank_capacity : ℕ) 
    (h1 : truck_tank_half_full + (car_tank_third_full car_tank_capacity) + 18 = truck_tank_capacity + car_tank_capacity) 
    (h2 : total_gallons_added = 18) : car_tank_capacity = 12 := 
by
  sorry

end car_tank_capacity_is_12_gallons_l1971_197181


namespace calc_tan_fraction_l1971_197128

theorem calc_tan_fraction :
  (1 - Real.tan (15 * Real.pi / 180)) / (1 + Real.tan (15 * Real.pi / 180)) = Real.tan (30 * Real.pi / 180) :=
by
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h_tan_30 : Real.tan (30 * Real.pi / 180) = Real.sqrt 3 / 3 := by sorry
  sorry

end calc_tan_fraction_l1971_197128


namespace car_speed_first_hour_l1971_197118

theorem car_speed_first_hour (x : ℝ) (h_second_hour_speed : x + 80 / 2 = 85) : x = 90 :=
sorry

end car_speed_first_hour_l1971_197118


namespace jamal_green_marbles_l1971_197106

theorem jamal_green_marbles
  (Y B K T : ℕ)
  (hY : Y = 12)
  (hB : B = 10)
  (hK : K = 1)
  (h_total : 1 / T = 1 / 28) :
  T - (Y + B + K) = 5 :=
by
  -- sorry, proof goes here
  sorry

end jamal_green_marbles_l1971_197106


namespace standard_equation_of_circle_tangent_to_x_axis_l1971_197143

theorem standard_equation_of_circle_tangent_to_x_axis :
  ∀ (x y : ℝ), ((x + 3) ^ 2 + (y - 4) ^ 2 = 16) :=
by
  -- Definitions based on the conditions
  let center_x := -3
  let center_y := 4
  let radius := 4

  sorry

end standard_equation_of_circle_tangent_to_x_axis_l1971_197143


namespace tan_alpha_eq_neg_sqrt_15_l1971_197177

/-- Given α in the interval (0, π) and the equation tan(2α) = sin(α) / (2 + cos(α)), prove that tan(α) = -√15. -/
theorem tan_alpha_eq_neg_sqrt_15 (α : ℝ) (h1 : 0 < α ∧ α < π) 
  (h2 : Real.tan (2 * α) = Real.sin α / (2 + Real.cos α)) : 
  Real.tan α = -Real.sqrt 15 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end tan_alpha_eq_neg_sqrt_15_l1971_197177


namespace solve_for_x_l1971_197136

theorem solve_for_x (x : ℝ) (h : 3 * (x - 5) = 3 * (18 - 5)) : x = 18 :=
by
  sorry

end solve_for_x_l1971_197136


namespace radius_of_inscribed_circle_l1971_197178

theorem radius_of_inscribed_circle (a b c : ℝ) (r : ℝ) 
  (ha : a = 5) (hb : b = 10) (hc : c = 20)
  (h : 1 / r = 1 / a + 1 / b + 1 / c + 2 * Real.sqrt ((1 / (a * b)) + (1 / (a * c)) + (1 / (b * c)))) :
  r = 20 * (7 - Real.sqrt 10) / 39 :=
by
  -- Statements and conditions are setup, but the proof is omitted.
  sorry

end radius_of_inscribed_circle_l1971_197178


namespace value_of_fraction_l1971_197147

open Real

theorem value_of_fraction (a : ℝ) (h : a^2 + a - 1 = 0) : (1 - a) / a + a / (1 + a) = 1 := 
by { sorry }

end value_of_fraction_l1971_197147


namespace a_pow_b_iff_a_minus_1_b_positive_l1971_197166

theorem a_pow_b_iff_a_minus_1_b_positive (a b : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) : 
  (a^b > 1) ↔ ((a - 1) * b > 0) := 
sorry

end a_pow_b_iff_a_minus_1_b_positive_l1971_197166


namespace sin_120_eq_sqrt3_div_2_l1971_197192

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_120_eq_sqrt3_div_2_l1971_197192


namespace suitable_sampling_method_l1971_197105

noncomputable def is_stratified_sampling_suitable (mountainous hilly flat low_lying sample_size : ℕ) (yield_dependent_on_land_type : Bool) : Bool :=
  if yield_dependent_on_land_type && mountainous + hilly + flat + low_lying > 0 then true else false

theorem suitable_sampling_method :
  is_stratified_sampling_suitable 8000 12000 24000 4000 480 true = true :=
by
  sorry

end suitable_sampling_method_l1971_197105


namespace largest_root_in_interval_l1971_197189

theorem largest_root_in_interval :
  ∃ (r : ℝ), (2 < r ∧ r < 3) ∧ (∃ (a_2 a_1 a_0 : ℝ), 
    |a_2| ≤ 3 ∧ |a_1| ≤ 3 ∧ |a_0| ≤ 3 ∧ a_2 + a_1 + a_0 = -6 ∧ r^3 + a_2 * r^2 + a_1 * r + a_0 = 0) :=
sorry

end largest_root_in_interval_l1971_197189


namespace fred_earned_correctly_l1971_197124

-- Assuming Fred's earnings from different sources
def fred_earned_newspapers := 16 -- dollars
def fred_earned_cars := 74 -- dollars

-- Total earnings over the weekend
def fred_earnings := fred_earned_newspapers + fred_earned_cars

-- Given condition that Fred earned 90 dollars over the weekend
def fred_earnings_given := 90 -- dollars

-- The theorem stating that Fred's total earnings match the given earnings
theorem fred_earned_correctly : fred_earnings = fred_earnings_given := by
  sorry

end fred_earned_correctly_l1971_197124


namespace average_speed_of_car_l1971_197121

theorem average_speed_of_car : 
  let distance1 := 30
  let speed1 := 60
  let distance2 := 35
  let speed2 := 70
  let distance3 := 36
  let speed3 := 80
  let distance4 := 20
  let speed4 := 55
  let total_distance := distance1 + distance2 + distance3 + distance4
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let time3 := distance3 / speed3
  let time4 := distance4 / speed4
  let total_time := time1 + time2 + time3 + time4
  let average_speed := total_distance / total_time
  average_speed = 66.70 := sorry

end average_speed_of_car_l1971_197121


namespace math_problem_l1971_197114

theorem math_problem (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : (a + b)^9 + a^6 = 2 :=
sorry

end math_problem_l1971_197114


namespace total_shaded_area_approx_l1971_197103

noncomputable def area_of_shaded_regions (r1 r2 : ℝ) :=
  let area_smaller_circle := 3 * 6 - (1 / 2) * Real.pi * r1^2
  let area_larger_circle := 6 * 12 - (1 / 2) * Real.pi * r2^2
  area_smaller_circle + area_larger_circle

theorem total_shaded_area_approx :
  abs (area_of_shaded_regions 3 6 - 19.4) < 0.05 :=
by
  sorry

end total_shaded_area_approx_l1971_197103


namespace explicit_form_of_function_l1971_197159

theorem explicit_form_of_function (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x * f x + f x * f y + y - 1) = f (x * f x + x * y) + y - 1) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end explicit_form_of_function_l1971_197159


namespace k_domain_all_reals_l1971_197140

noncomputable def domain_condition (k : ℝ) : Prop :=
  9 + 28 * k < 0

noncomputable def k_values : Set ℝ :=
  {k : ℝ | domain_condition k}

theorem k_domain_all_reals :
  k_values = {k : ℝ | k < -9 / 28} :=
by
  sorry

end k_domain_all_reals_l1971_197140


namespace total_crayons_l1971_197117

-- Define the number of crayons Billy has
def billy_crayons : ℝ := 62.0

-- Define the number of crayons Jane has
def jane_crayons : ℝ := 52.0

-- Formulate the theorem to prove the total number of crayons
theorem total_crayons : billy_crayons + jane_crayons = 114.0 := by
  sorry

end total_crayons_l1971_197117


namespace domain_f_l1971_197180

noncomputable def f (x : ℝ) : ℝ := -2 / (Real.sqrt (x + 5)) + Real.log (2^x + 1)

theorem domain_f :
  {x : ℝ | (-5 ≤ x)} = {x : ℝ | f x ∈ Set.univ} := sorry

end domain_f_l1971_197180


namespace number_of_factors_l1971_197162

theorem number_of_factors : 
  ∃ (count : ℕ), count = 45 ∧
    (∀ n : ℕ, (1 ≤ n ∧ n ≤ 500) → 
      ∃ a b : ℤ, (x - a) * (x - b) = x^2 + 2 * x - n) :=
by
  sorry

end number_of_factors_l1971_197162


namespace trajectory_eqn_l1971_197146

-- Definition of points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Conditions given in the problem
def PA_squared (P : ℝ × ℝ) : ℝ := (P.1 + 1)^2 + P.2^2
def PB_squared (P : ℝ × ℝ) : ℝ := (P.1 - 1)^2 + P.2^2

-- The main statement to prove
theorem trajectory_eqn (P : ℝ × ℝ) (h : PA_squared P = 3 * PB_squared P) : 
  P.1^2 + P.2^2 - 4 * P.1 + 1 = 0 :=
by 
  sorry

end trajectory_eqn_l1971_197146


namespace find_X_l1971_197174

def tax_problem (X I T : ℝ) (income : ℝ) (total_tax : ℝ) :=
  (income = 56000) ∧ (total_tax = 8000) ∧ (T = 0.12 * X + 0.20 * (I - X))

theorem find_X :
  ∃ X : ℝ, ∀ I T : ℝ, tax_problem X I T 56000 8000 → X = 40000 := 
  by
    sorry

end find_X_l1971_197174


namespace min_value_expr_l1971_197184

theorem min_value_expr (x y : ℝ) : 
  ∃ x y : ℝ, (x, y) = (4, 0) ∧ (∀ x y : ℝ, x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y ≥ -22) :=
by
  sorry

end min_value_expr_l1971_197184


namespace exist_two_divisible_by_n_l1971_197158

theorem exist_two_divisible_by_n (n : ℤ) (a : Fin (n.toNat + 1) → ℤ) :
  ∃ (i j : Fin (n.toNat + 1)), i ≠ j ∧ (a i - a j) % n = 0 :=
by
  sorry

end exist_two_divisible_by_n_l1971_197158


namespace factor_expression_l1971_197120

theorem factor_expression (x : ℝ) :
  x^3 - 6 * x^2 + 9 * x = x * (x - 3)^2 := 
  sorry

end factor_expression_l1971_197120


namespace aunt_money_calculation_l1971_197168

variable (total_money_received aunt_money : ℕ)
variable (bank_amount grandfather_money : ℕ := 150)

theorem aunt_money_calculation (h1 : bank_amount = 45) (h2 : bank_amount = total_money_received / 5) (h3 : total_money_received = aunt_money + grandfather_money) :
  aunt_money = 75 :=
by
  -- The proof is captured in these statements:
  sorry

end aunt_money_calculation_l1971_197168


namespace find_point_on_y_axis_l1971_197179

/-- 
Given points A (1, 2, 3) and B (2, -1, 4), and a point P on the y-axis 
such that the distances |PA| and |PB| are equal, 
prove that the coordinates of point P are (0, -7/6, 0).
 -/
theorem find_point_on_y_axis
  (A B : ℝ × ℝ × ℝ)
  (hA : A = (1, 2, 3))
  (hB : B = (2, -1, 4))
  (P : ℝ × ℝ × ℝ)
  (hP : ∃ y : ℝ, P = (0, y, 0)) :
  dist A P = dist B P → P = (0, -7/6, 0) :=
by
  sorry

end find_point_on_y_axis_l1971_197179


namespace ratio_of_heights_eq_three_twentieths_l1971_197170

noncomputable def base_circumference : ℝ := 32 * Real.pi
noncomputable def original_height : ℝ := 60
noncomputable def shorter_volume : ℝ := 768 * Real.pi

theorem ratio_of_heights_eq_three_twentieths
  (base_circumference : ℝ)
  (original_height : ℝ)
  (shorter_volume : ℝ)
  (h' : ℝ)
  (ratio : ℝ) :
  base_circumference = 32 * Real.pi →
  original_height = 60 →
  shorter_volume = 768 * Real.pi →
  (1 / 3 * Real.pi * (base_circumference / (2 * Real.pi))^2 * h') = shorter_volume →
  ratio = h' / original_height →
  ratio = 3 / 20 := 
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end ratio_of_heights_eq_three_twentieths_l1971_197170


namespace statement_B_statement_D_l1971_197123

noncomputable def f (x : ℝ) := 2 * Real.sin x * (Real.cos x + Real.sqrt 3 * Real.sin x) - Real.sqrt 3 + 1

theorem statement_B (x₁ x₂ : ℝ) (h1 : -π / 12 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < 5 * π / 12) :
  f x₁ < f x₂ := sorry

theorem statement_D (x₁ x₂ x₃ : ℝ) (h1 : π / 3 ≤ x₁) (h2 : x₁ ≤ π / 2) (h3 : π / 3 ≤ x₂) (h4 : x₂ ≤ π / 2) (h5 : π / 3 ≤ x₃) (h6 : x₃ ≤ π / 2) :
  f x₁ + f x₂ - f x₃ > 2 := sorry

end statement_B_statement_D_l1971_197123


namespace sale_price_60_l1971_197155

theorem sale_price_60 (original_price : ℕ) (discount_percentage : ℝ) (sale_price : ℝ) 
  (h1 : original_price = 100) 
  (h2 : discount_percentage = 0.40) :
  sale_price = (original_price : ℝ) * (1 - discount_percentage) :=
by
  sorry

end sale_price_60_l1971_197155


namespace interval_of_segmentation_l1971_197148

-- Define the population size and sample size as constants.
def population_size : ℕ := 2000
def sample_size : ℕ := 40

-- State the theorem for the interval of segmentation.
theorem interval_of_segmentation :
  population_size / sample_size = 50 :=
sorry

end interval_of_segmentation_l1971_197148


namespace total_winter_clothing_l1971_197111

def num_scarves (boxes : ℕ) (scarves_per_box : ℕ) : ℕ := boxes * scarves_per_box
def num_mittens (boxes : ℕ) (mittens_per_box : ℕ) : ℕ := boxes * mittens_per_box
def num_hats (boxes : ℕ) (hats_per_box : ℕ) : ℕ := boxes * hats_per_box
def num_jackets (boxes : ℕ) (jackets_per_box : ℕ) : ℕ := boxes * jackets_per_box

theorem total_winter_clothing :
    num_scarves 4 8 + num_mittens 3 6 + num_hats 2 5 + num_jackets 1 3 = 63 :=
by
  -- The proof will use the given definitions and calculate the total
  sorry

end total_winter_clothing_l1971_197111


namespace increase_80_by_150_percent_l1971_197129

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end increase_80_by_150_percent_l1971_197129


namespace scale_model_height_l1971_197113

/-- 
Given a scale model ratio and the actual height of the skyscraper in feet,
we can deduce the height of the model in inches.
-/
theorem scale_model_height
  (scale_ratio : ℕ := 25)
  (actual_height_feet : ℕ := 1250) :
  (actual_height_feet / scale_ratio) * 12 = 600 :=
by 
  sorry

end scale_model_height_l1971_197113


namespace necessary_not_sufficient_l1971_197150

-- Definitions and conditions based on the problem statement
def x_ne_1 (x : ℝ) : Prop := x ≠ 1
def polynomial_ne_zero (x : ℝ) : Prop := (x^2 - 3 * x + 2) ≠ 0

-- The theorem statement
theorem necessary_not_sufficient (x : ℝ) : 
  (∀ x, polynomial_ne_zero x → x_ne_1 x) ∧ ¬ (∀ x, x_ne_1 x → polynomial_ne_zero x) :=
by 
  intros
  sorry

end necessary_not_sufficient_l1971_197150


namespace solution_l1971_197194

-- Conditions
def x : ℚ := 3/5
def y : ℚ := 5/3

-- Proof problem
theorem solution : (1/3) * x^8 * y^9 = 5/9 := sorry

end solution_l1971_197194


namespace max_k_value_l1971_197142

theorem max_k_value :
  ∃ A B C k : ℕ, 
  (A ≠ 0) ∧ 
  (A < 10) ∧ 
  (B < 10) ∧ 
  (C < 10) ∧
  (10 * A + B) * k = 100 * A + 10 * C + B ∧
  (∀ k' : ℕ, 
     ((A ≠ 0) ∧ (A < 10) ∧ (B < 10) ∧ (C < 10) ∧
     (10 * A + B) * k' = 100 * A + 10 * C + B) 
     → k' ≤ 19) ∧
  k = 19 :=
sorry

end max_k_value_l1971_197142


namespace unattainable_value_l1971_197172

theorem unattainable_value (x : ℝ) (hx : x ≠ -4/3) : 
  ¬ ∃ y : ℝ, y = (1 - x) / (3 * x + 4) ∧ y = -1/3 :=
by
  sorry

end unattainable_value_l1971_197172


namespace crushing_load_l1971_197122

theorem crushing_load (T H C : ℝ) (L : ℝ) 
  (h1 : T = 5) (h2 : H = 10) (h3 : C = 3)
  (h4 : L = C * 25 * T^4 / H^2) : 
  L = 468.75 :=
by
  sorry

end crushing_load_l1971_197122


namespace at_least_two_participants_solved_exactly_five_l1971_197110

open Nat Real

variable {n : ℕ}  -- Number of participants
variable {pij : ℕ → ℕ → ℕ} -- Number of contestants who correctly answered both the i-th and j-th problems

-- Conditions as definitions in Lean 4
def conditions (n : ℕ) (pij : ℕ → ℕ → ℕ) : Prop :=
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 6 → pij i j > (2 * n) / 5) ∧
  (∀ k, ¬ (∀ i, 1 ≤ i ∧ i ≤ 6 → pij k i = 1))

-- Main theorem statement
theorem at_least_two_participants_solved_exactly_five (n : ℕ) (pij : ℕ → ℕ → ℕ) (h : conditions n pij) : ∃ k₁ k₂, k₁ ≠ k₂ ∧ (∃ i, 1 ≤ i ∧ i ≤ 6 ∧ pij k₁ i = 1) ∧ (∃ i, 1 ≤ i ∧ i ≤ 6 ∧ pij k₂ i = 1) := sorry

end at_least_two_participants_solved_exactly_five_l1971_197110


namespace range_of_a_l1971_197109

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 2

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ≥ -1 → f a x ≥ a) : -3 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l1971_197109


namespace pen_and_pencil_total_cost_l1971_197139

theorem pen_and_pencil_total_cost :
  ∀ (pen pencil : ℕ), pen = 4 → pen = 2 * pencil → pen + pencil = 6 :=
by
  intros pen pencil
  intro h1
  intro h2
  sorry

end pen_and_pencil_total_cost_l1971_197139


namespace sandy_more_tokens_than_siblings_l1971_197119

-- Define the initial conditions
def initial_tokens : ℕ := 3000000
def initial_transaction_fee_percent : ℚ := 0.10
def value_increase_percent : ℚ := 0.20
def additional_tokens : ℕ := 500000
def additional_transaction_fee_percent : ℚ := 0.07
def sandy_keep_percent : ℚ := 0.40
def siblings : ℕ := 7
def sibling_transaction_fee_percent : ℚ := 0.05

-- Define the main theorem to prove
theorem sandy_more_tokens_than_siblings :
  let received_initial_tokens := initial_tokens * (1 - initial_transaction_fee_percent)
  let increased_tokens := received_initial_tokens * (1 + value_increase_percent)
  let received_additional_tokens := additional_tokens * (1 - additional_transaction_fee_percent)
  let total_tokens := increased_tokens + received_additional_tokens
  let sandy_tokens := total_tokens * sandy_keep_percent
  let remaining_tokens := total_tokens * (1 - sandy_keep_percent)
  let each_sibling_tokens := remaining_tokens / siblings * (1 - sibling_transaction_fee_percent)
  sandy_tokens - each_sibling_tokens = 1180307.1428 := sorry

end sandy_more_tokens_than_siblings_l1971_197119


namespace measure_of_angle_B_l1971_197112

theorem measure_of_angle_B (a b c R : ℝ) (A B C : ℝ)
  (h1 : a = 2 * R * Real.sin A)
  (h2 : b = 2 * R * Real.sin B)
  (h3 : c = 2 * R * Real.sin C)
  (h4 : 2 * R * (Real.sin A ^ 2 - Real.sin B ^ 2) = (Real.sqrt 2 * a - c) * Real.sin C) :
  B = Real.pi / 4 :=
by
  sorry

end measure_of_angle_B_l1971_197112


namespace ellipse_standard_equation_l1971_197185

theorem ellipse_standard_equation :
  ∀ (a b c : ℝ), a = 9 → c = 6 → b = Real.sqrt (a^2 - c^2) →
  (b ≠ 0 ∧ a ≠ 0 → (∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)) :=
by
  sorry

end ellipse_standard_equation_l1971_197185


namespace find_first_term_l1971_197190

noncomputable def firstTermOfGeometricSeries (a r : ℝ) : Prop :=
  (a / (1 - r) = 30) ∧ (a^2 / (1 - r^2) = 120)

theorem find_first_term :
  ∃ a r : ℝ, firstTermOfGeometricSeries a r ∧ a = 120 / 17 :=
by
  sorry

end find_first_term_l1971_197190


namespace keith_initial_cards_l1971_197125

theorem keith_initial_cards (new_cards : ℕ) (cards_after_incident : ℕ) (total_cards : ℕ) :
  new_cards = 8 →
  cards_after_incident = 46 →
  total_cards = 2 * cards_after_incident →
  (total_cards - new_cards) = 84 :=
by
  intros
  sorry

end keith_initial_cards_l1971_197125


namespace num_20_paise_coins_l1971_197135

theorem num_20_paise_coins (x y : ℕ) (h1 : x + y = 344) (h2 : 20 * x + 25 * y = 7100) : x = 300 :=
by
  sorry

end num_20_paise_coins_l1971_197135


namespace number_of_days_at_Tom_house_l1971_197127

-- Define the constants and conditions
def total_people := 6
def plates_per_person_per_day := 6
def total_plates := 144

-- Prove that the number of days they were at Tom's house is 4
theorem number_of_days_at_Tom_house : total_plates / (total_people * plates_per_person_per_day) = 4 :=
  sorry

end number_of_days_at_Tom_house_l1971_197127


namespace a_can_be_any_real_l1971_197191

theorem a_can_be_any_real (a b c d e : ℝ) (h1 : b ≠ 0) (h2 : d ≠ 0) (h3 : e ≠ 0) :
  ∃ a : ℝ, true :=
by sorry

end a_can_be_any_real_l1971_197191


namespace base_k_number_eq_binary_l1971_197156

theorem base_k_number_eq_binary (k : ℕ) (h : k^2 + 3 * k + 2 = 30) : k = 4 :=
sorry

end base_k_number_eq_binary_l1971_197156


namespace problem_A_problem_B_problem_C_problem_D_l1971_197193

theorem problem_A : 2 * Real.sqrt 3 + 3 * Real.sqrt 2 ≠ 5 * Real.sqrt 5 := by
  sorry

theorem problem_B : 3 * Real.sqrt 3 * (3 * Real.sqrt 2) ≠ 3 * Real.sqrt 6 := by
  sorry

theorem problem_C : (Real.sqrt 27 / Real.sqrt 3) = 3 := by
  sorry

theorem problem_D : 2 * Real.sqrt 2 - Real.sqrt 2 ≠ 2 := by
  sorry

end problem_A_problem_B_problem_C_problem_D_l1971_197193


namespace sum_of_coordinates_B_l1971_197130

theorem sum_of_coordinates_B
  (x y : ℤ)
  (Mx My : ℤ)
  (Ax Ay : ℤ)
  (M : Mx = 2 ∧ My = -3)
  (A : Ax = -4 ∧ Ay = -5)
  (midpoint_x : (x + Ax) / 2 = Mx)
  (midpoint_y : (y + Ay) / 2 = My) :
  x + y = 7 :=
by
  sorry

end sum_of_coordinates_B_l1971_197130


namespace PQ_PR_QR_div_l1971_197176

theorem PQ_PR_QR_div (p q r : ℝ)
    (midQR : p = 0) (midPR : q = 0) (midPQ : r = 0) :
    (4 * (q ^ 2 + r ^ 2) + 4 * (p ^ 2 + r ^ 2) + 4 * (p ^ 2 + q ^ 2)) / (p ^ 2 + q ^ 2 + r ^ 2) = 8 :=
by {
    sorry
}

end PQ_PR_QR_div_l1971_197176


namespace swimming_pool_length_correct_l1971_197199

noncomputable def swimming_pool_length (V_removed: ℝ) (W: ℝ) (H: ℝ) (gal_to_cuft: ℝ): ℝ :=
  V_removed / (W * H / gal_to_cuft)

theorem swimming_pool_length_correct:
  swimming_pool_length 3750 25 0.5 7.48052 = 40.11 :=
by
  sorry

end swimming_pool_length_correct_l1971_197199


namespace solve_xyz_l1971_197169

variable {x y z : ℝ}

theorem solve_xyz (h1 : (x + y + z) * (xy + xz + yz) = 35) (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11) : x * y * z = 8 := 
by
  sorry

end solve_xyz_l1971_197169


namespace market_value_of_stock_l1971_197116

def face_value : ℝ := 100
def dividend_percentage : ℝ := 0.13
def yield : ℝ := 0.08

theorem market_value_of_stock : 
  (dividend_percentage * face_value / yield) * 100 = 162.50 :=
by
  sorry

end market_value_of_stock_l1971_197116


namespace potatoes_cost_l1971_197165

-- Defining our constants and conditions
def pounds_per_person : ℝ := 1.5
def number_of_people : ℝ := 40
def pounds_per_bag : ℝ := 20
def cost_per_bag : ℝ := 5

-- The main goal: to prove the total cost is 15.
theorem potatoes_cost : (number_of_people * pounds_per_person) / pounds_per_bag * cost_per_bag = 15 :=
by sorry

end potatoes_cost_l1971_197165


namespace marble_ratio_is_two_to_one_l1971_197138

-- Conditions
def dan_blue_marbles : ℕ := 5
def mary_blue_marbles : ℕ := 10

-- Ratio definition
def marble_ratio : ℚ := mary_blue_marbles / dan_blue_marbles

-- Theorem statement
theorem marble_ratio_is_two_to_one : marble_ratio = 2 :=
by 
  -- Prove the statement here
  sorry

end marble_ratio_is_two_to_one_l1971_197138


namespace complex_expression_is_none_of_the_above_l1971_197132

-- We define the problem in Lean, stating that the given complex expression is not equal to any of the simplified forms
theorem complex_expression_is_none_of_the_above (x : ℝ) :
  ( ( ((x+1)^2*(x^2-x+2)^2) / (x^3+1)^2 )^2 * ( ((x-1)^2*(x^2+x+2)^2) / (x^3-2)^2 )^2 ≠ (x+1)^4 ) ∧
  ( ( ((x+1)^2*(x^2-x+2)^2) / (x^3+1)^2 )^2 * ( ((x-1)^2*(x^2+x+2)^2) / (x^3-2)^2 )^2 ≠ (x^3+1)^4 ) ∧
  ( ( ((x+1)^2*(x^2-x+2)^2) / (x^3+1)^2 )^2 * ( ((x-1)^2*(x^2+x+2)^2) / (x^3-2)^2 )^2 ≠ (x-1)^4 ) :=
sorry

end complex_expression_is_none_of_the_above_l1971_197132


namespace number_of_ways_to_label_decagon_equal_sums_l1971_197182

open Nat

-- Formal definition of the problem
def sum_of_digits : Nat := (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)

-- The problem statement: Prove there are 3840 ways to label digits ensuring the given condition
theorem number_of_ways_to_label_decagon_equal_sums :
  ∃ (n : Nat), n = 3840 ∧ ∀ (A B C D E F G H I J K L : Nat), 
    (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (A ≠ F) ∧ (A ≠ G) ∧ (A ≠ H) ∧ (A ≠ I) ∧ (A ≠ J) ∧ (A ≠ K) ∧ (A ≠ L) ∧
    (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (B ≠ F) ∧ (B ≠ G) ∧ (B ≠ H) ∧ (B ≠ I) ∧ (B ≠ J) ∧ (B ≠ K) ∧ (B ≠ L) ∧
    (C ≠ D) ∧ (C ≠ E) ∧ (C ≠ F) ∧ (C ≠ G) ∧ (C ≠ H) ∧ (C ≠ I) ∧ (C ≠ J) ∧ (C ≠ K) ∧ (C ≠ L) ∧
    (D ≠ E) ∧ (D ≠ F) ∧ (D ≠ G) ∧ (D ≠ H) ∧ (D ≠ I) ∧ (D ≠ J) ∧ (D ≠ K) ∧ (D ≠ L) ∧
    (E ≠ F) ∧ (E ≠ G) ∧ (E ≠ H) ∧ (E ≠ I) ∧ (E ≠ J) ∧ (E ≠ K) ∧ (E ≠ L) ∧
    (F ≠ G) ∧ (F ≠ H) ∧ (F ≠ I) ∧ (F ≠ J) ∧ (F ≠ K) ∧ (F ≠ L) ∧
    (G ≠ H) ∧ (G ≠ I) ∧ (G ≠ J) ∧ (G ≠ K) ∧ (G ≠ L) ∧
    (H ≠ I) ∧ (H ≠ J) ∧ (H ≠ K) ∧ (H ≠ L) ∧
    (I ≠ J) ∧ (I ≠ K) ∧ (I ≠ L) ∧
    (J ≠ K) ∧ (J ≠ L) ∧
    (K ≠ L) ∧
    (A + L + F = B + L + G) ∧ (B + L + G = C + L + H) ∧ 
    (C + L + H = D + L + I) ∧ (D + L + I = E + L + J) ∧ 
    (E + L + J = F + L + K) ∧ (F + L + K = A + L + F) :=
sorry

end number_of_ways_to_label_decagon_equal_sums_l1971_197182


namespace polygon_sides_l1971_197195

theorem polygon_sides (h : ∀ (θ : ℕ), θ = 108) : ∃ n : ℕ, n = 5 :=
by
  sorry

end polygon_sides_l1971_197195


namespace remainder_of_towers_l1971_197141

open Nat

def count_towers (m : ℕ) : ℕ :=
  match m with
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | 4 => 18
  | 5 => 54
  | 6 => 162
  | _ => 0

theorem remainder_of_towers : (count_towers 6) % 100 = 62 :=
  by
  sorry

end remainder_of_towers_l1971_197141


namespace grade_above_B_l1971_197153

theorem grade_above_B (total_students : ℕ) (percentage_below_B : ℕ) (students_above_B : ℕ) :
  total_students = 60 ∧ percentage_below_B = 40 ∧ students_above_B = total_students * (100 - percentage_below_B) / 100 →
  students_above_B = 36 :=
by
  sorry

end grade_above_B_l1971_197153


namespace acute_angle_30_l1971_197126

theorem acute_angle_30 (α : ℝ) (h : Real.cos (π / 6) * Real.sin α = Real.sqrt 3 / 4) : α = π / 6 := 
by 
  sorry

end acute_angle_30_l1971_197126


namespace total_possible_match_sequences_l1971_197154

theorem total_possible_match_sequences :
  let num_teams := 2
  let team_size := 7
  let possible_sequences := 2 * (Nat.choose (2 * team_size - 1) (team_size - 1))
  possible_sequences = 3432 :=
by
  sorry

end total_possible_match_sequences_l1971_197154


namespace ratio_of_probabilities_l1971_197183

noncomputable def balls_toss (balls bins : ℕ) : Nat := by
  sorry

def prob_A : ℚ := by
  sorry
  
def prob_B : ℚ := by
  sorry

theorem ratio_of_probabilities (balls : ℕ) (bins : ℕ) 
  (h_balls : balls = 20) (h_bins : bins = 5) (p q : ℚ) 
  (h_p : p = prob_A) (h_q : q = prob_B) :
  (p / q) = 4 := by
  sorry

end ratio_of_probabilities_l1971_197183


namespace tim_final_soda_cans_l1971_197134

-- Definitions based on given conditions
def initialSodaCans : ℕ := 22
def cansTakenByJeff : ℕ := 6
def remainingCans (t0 j : ℕ) : ℕ := t0 - j
def additionalCansBought (remaining : ℕ) : ℕ := remaining / 2

-- Function to calculate final number of soda cans
def finalSodaCans (t0 j : ℕ) : ℕ :=
  let remaining := remainingCans t0 j
  remaining + additionalCansBought remaining

-- Theorem to prove the final number of soda cans
theorem tim_final_soda_cans : finalSodaCans initialSodaCans cansTakenByJeff = 24 :=
by
  sorry

end tim_final_soda_cans_l1971_197134


namespace percentage_invalid_votes_l1971_197186

theorem percentage_invalid_votes
  (total_votes : ℕ)
  (votes_for_A : ℕ)
  (candidate_A_percentage : ℝ)
  (total_votes_count : total_votes = 560000)
  (votes_for_A_count : votes_for_A = 404600)
  (candidate_A_percentage_count : candidate_A_percentage = 0.85) :
  ∃ (x : ℝ), (x / 100) * total_votes = total_votes - votes_for_A / candidate_A_percentage ∧ x = 15 :=
by
  sorry

end percentage_invalid_votes_l1971_197186


namespace digits_difference_l1971_197163

theorem digits_difference (d A B : ℕ) (hd : d > 7) (h : d^2 * A + d * B + d^2 * A + d * A = 1 * d^2 + 7 * d + 2) : 
  A - B = 4 := 
sorry

end digits_difference_l1971_197163


namespace sum_of_opposite_numbers_is_zero_l1971_197101

theorem sum_of_opposite_numbers_is_zero {a b : ℝ} (h : a + b = 0) : a + b = 0 := 
h

end sum_of_opposite_numbers_is_zero_l1971_197101


namespace triangle_angle_contradiction_l1971_197131

theorem triangle_angle_contradiction (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : α < 60) (h3 : β < 60) (h4 : γ < 60) : false := 
sorry

end triangle_angle_contradiction_l1971_197131


namespace additional_songs_added_l1971_197133

theorem additional_songs_added (original_songs : ℕ) (song_duration : ℕ) (total_duration : ℕ) :
  original_songs = 25 → song_duration = 3 → total_duration = 105 → 
  (total_duration - original_songs * song_duration) / song_duration = 10 :=
by
  intros h1 h2 h3
  sorry

end additional_songs_added_l1971_197133


namespace maximize_rectangle_area_l1971_197187

theorem maximize_rectangle_area (l w : ℝ) (h : l + w ≥ 40) : l * w ≤ 400 :=
by sorry

end maximize_rectangle_area_l1971_197187
